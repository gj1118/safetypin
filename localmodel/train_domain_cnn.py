"""
Train domain phishing classifier using Character-level CNN (CharCNN).
This model learns character-level patterns to detect phishing domains.

Run: python train_domain_cnn.py
"""

import os
import json
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
NUM_EPOCHS = 15
LEARNING_RATE = 0.002
MAX_LENGTH = 100  # Max characters in domain

print(f"Using device: {DEVICE}")

OUTPUT_DIR = "output/domain_classifier_cnn"


class CharCNN(nn.Module):
    """Character-level CNN for domain classification"""
    
    def __init__(self, num_chars=128, num_classes=2, dropout=0.4):
        super().__init__()
        
        # Character embedding
        self.embedding = nn.Embedding(num_chars, 32)
        
        # Multiple filter sizes for capturing different patterns
        self.convs = nn.ModuleList([
            nn.Conv1d(32, 64, kernel_size=k, padding=k//2)
            for k in [3, 4, 5]
        ])
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch, seq_len)
        
        # Embed characters
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        
        # Apply convolutions with different filter sizes
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))  # (batch, out_channels, seq_len)
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # (batch, out_channels)
            conv_outputs.append(pooled)
        
        # Concatenate all filter outputs
        x = torch.cat(conv_outputs, dim=1)  # (batch, 64 * 3)
        x = self.dropout(x)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class DomainDataset(Dataset):
    """Dataset for domain classification"""
    
    def __init__(self, domains, labels, char_to_idx, max_length=MAX_LENGTH):
        self.domains = domains
        self.labels = labels
        self.char_to_idx = char_to_idx
        self.max_length = max_length
        
    def __len__(self):
        return len(self.domains)
    
    def __getitem__(self, idx):
        domain = self.domains[idx]
        label = self.labels[idx]
        
        # Convert characters to indices
        indices = [self.char_to_idx.get(c, 0) for c in domain[:self.max_length]]
        
        # Pad or truncate
        if len(indices) < self.max_length:
            indices += [0] * (self.max_length - len(indices))
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def load_domain_data(filepath="data/domain_reputation_augmented.json"):
    """Load domain data from JSON"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    domains = []
    labels = []
    for domain, label in data.items():
        # Clean domain
        domain = domain.lower()
        domain = re.sub(r'^https?://', '', domain)
        domain = domain.split('/')[0]
        
        domains.append(domain)
        labels.append(1 if label == "phishing" else 0)
    
    print(f"Loaded {len(domains)} domains")
    phishing_count = sum(labels)
    legitimate_count = len(labels) - phishing_count
    print(f"  Phishing: {phishing_count}, Legitimate: {legitimate_count}")
    
    return domains, labels


def build_char_vocab(domains):
    """Build character vocabulary from domains"""
    chars = set()
    for domain in domains:
        chars.update(domain)
    
    # Reserve 0 for padding, 1 for unknown
    char_to_idx = {c: i + 2 for i, c in enumerate(sorted(chars))}
    char_to_idx['<PAD>'] = 0
    char_to_idx['<UNK>'] = 1
    
    print(f"Vocabulary size: {len(char_to_idx)}")
    return char_to_idx


def train():
    print("=" * 50)
    print("Training CharCNN Domain Phishing Classifier")
    print("=" * 50)
    
    # Load augmented data
    domains, labels = load_domain_data("data/domain_reputation_augmented.json")
    
    # Build vocabulary
    char_to_idx = build_char_vocab(domains)
    
    # Train/validation split
    train_domains, val_domains, train_labels, val_labels = train_test_split(
        domains, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\nTraining samples: {len(train_domains)}")
    print(f"Validation samples: {len(val_domains)}")
    
    # Create datasets
    train_dataset = DomainDataset(train_domains, train_labels, char_to_idx)
    val_dataset = DomainDataset(val_domains, val_labels, char_to_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = CharCNN(num_chars=len(char_to_idx), num_classes=2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    best_model_state = None
    patience = 5
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
        
        val_acc = 100. * val_correct / val_total
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Acc: {train_acc:.1f}%, Val Acc: {val_acc:.1f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  New best: {best_val_acc:.1f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    
    print("\n" + "=" * 50)
    print("Final Results")
    print("=" * 50)
    print(f"Best Validation Accuracy: {best_val_acc:.1f}%")
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["legitimate", "phishing"]))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(f"  True Legitimate: {cm[0,0]}, False Phishing: {cm[0,1]}")
    print(f"  False Legitimate: {cm[1,0]}, True Phishing: {cm[1,1]}")
    
    # Save model
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'char_to_idx': char_to_idx,
        'max_length': MAX_LENGTH,
    }, f"{OUTPUT_DIR}/model.pt")
    
    print(f"\nModel saved to {OUTPUT_DIR}/")
    
    return model, char_to_idx


if __name__ == "__main__":
    train()
