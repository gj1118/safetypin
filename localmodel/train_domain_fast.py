"""
Simplified CharCNN for faster training.
"""

import os
import json
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
NUM_EPOCHS = 10
LEARNING_RATE = 0.002
MAX_LENGTH = 64

print(f"Using device: {DEVICE}")

OUTPUT_DIR = "output/domain_classifier_cnn"


class CharCNN(nn.Module):
    def __init__(self, num_chars=128, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(num_chars, 32)
        self.conv1 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(2)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def load_data(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    domains, labels = [], []
    for domain, label in data.items():
        domain = re.sub(r'^https?://', '', domain.lower()).split('/')[0]
        domains.append(domain)
        labels.append(1 if label == "phishing" else 0)
    
    return domains, labels


def encode_domains(domains, char_to_idx, max_len):
    encoded = []
    for domain in domains:
        indices = [char_to_idx.get(c, 1) for c in domain[:max_len]]
        if len(indices) < max_len:
            indices += [0] * (max_len - len(indices))
        encoded.append(indices)
    return np.array(encoded)


def main():
    print("Loading data...")
    domains, labels = load_data("data/domain_reputation_augmented.json")
    print(f"Total: {len(domains)}, Phishing: {sum(labels)}, Legit: {len(labels) - sum(labels)}")
    
    # Build vocab
    chars = set()
    for d in domains:
        chars.update(d)
    char_to_idx = {c: i + 2 for i, c in enumerate(sorted(chars))}
    char_to_idx['<PAD>'] = 0
    char_to_idx['<UNK>'] = 1
    print(f"Vocab: {len(char_to_idx)}")
    
    # Encode
    X = encode_domains(domains, char_to_idx, MAX_LENGTH)
    y = np.array(labels)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    train_ds = TensorDataset(torch.LongTensor(X_train), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.LongTensor(X_val), torch.LongTensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    
    model = CharCNN(len(char_to_idx)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    best_state = None
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
        
        # Eval
        model.eval()
        preds, true = [], []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(DEVICE)
                out = model(inputs)
                preds.extend(out.argmax(1).cpu().numpy())
                true.extend(targets.numpy())
        
        acc = accuracy_score(true, preds)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Val Acc: {acc*100:.1f}%")
        
        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict().copy()
    
    model.load_state_dict(best_state)
    
    print(f"\nBest: {best_acc*100:.1f}%")
    print(classification_report(true, preds, target_names=["legitimate", "phishing"]))
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'char_to_idx': char_to_idx,
        'max_length': MAX_LENGTH,
    }, f"{OUTPUT_DIR}/model.pt")
    print(f"Saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
