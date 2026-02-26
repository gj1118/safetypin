"""
Export CharCNN domain classifier to ONNX format for faster inference.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

OUTPUT_DIR = "output/domain_classifier_cnn"
ONNX_PATH = "output/text_classifier_domain.onnx"


class CharCNN(nn.Module):
    def __init__(self, num_chars=128, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(num_chars, 32)
        self.conv1 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # (batch, 32, seq)
        x = F.relu(self.conv1(x))  # (batch, 64, seq)
        x = F.relu(self.conv2(x))  # (batch, 64, seq)
        x = self.pool(x)  # (batch, 64, seq//2)
        x = x.mean(dim=2)  # Global average pooling
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def train_and_export():
    """Train a new model with ONNX-compatible architecture and export it"""
    print("Training ONNX-compatible domain classifier...")
    
    import json
    import re
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, TensorDataset
    
    # Load data
    with open("data/domain_reputation_augmented.json", 'r') as f:
        data = json.load(f)
    
    domains, labels = [], []
    for domain, label in data.items():
        domain = re.sub(r'^https?://', '', domain.lower()).split('/')[0]
        domains.append(domain)
        labels.append(1 if label == "phishing" else 0)
    
    print(f"Total: {len(domains)}")
    
    # Build vocab
    chars = set()
    for d in domains:
        chars.update(d)
    char_to_idx = {c: i + 2 for i, c in enumerate(sorted(chars))}
    char_to_idx['<PAD>'] = 0
    char_to_idx['<UNK>'] = 1
    
    max_length = 64
    
    # Encode
    def encode(d):
        indices = [char_to_idx.get(c, 1) for c in d[:max_length]]
        if len(indices) < max_length:
            indices += [0] * (max_length - len(indices))
        return indices
    
    X = np.array([encode(d) for d in domains])
    y = np.array(labels)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    train_ds = TensorDataset(torch.LongTensor(X_train), torch.LongTensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    
    model = CharCNN(len(char_to_idx)).to('cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(15):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/10")
    
    # Save model with ONNX-compatible architecture
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'char_to_idx': char_to_idx,
        'max_length': max_length,
    }, f"{OUTPUT_DIR}/model.pt")
    print(f"Model saved to {OUTPUT_DIR}/model.pt")
    
    # Export to ONNX
    export_to_onnx()


def export_to_onnx():
    print("Exporting domain classifier to ONNX...")
    
    # Load model
    checkpoint = torch.load(f"{OUTPUT_DIR}/model.pt", map_location='cpu', weights_only=False)
    char_to_idx = checkpoint['char_to_idx']
    max_length = checkpoint['max_length']
    
    model = CharCNN(len(char_to_idx))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randint(0, len(char_to_idx), (1, max_length))
    
    # Export to ONNX (use default opset version to avoid conversion errors)
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Exported to {ONNX_PATH}")
    
    # Save metadata
    metadata = {
        'char_to_idx': char_to_idx,
        'max_length': max_length,
        'vocab_size': len(char_to_idx)
    }
    
    import json
    with open(f"{OUTPUT_DIR}/metadata.json", 'w') as f:
        json.dump(metadata, f)
    
    print(f"Metadata saved to {OUTPUT_DIR}/metadata.json")
    print(f"Export complete!")
    
    return ONNX_PATH


if __name__ == "__main__":
    export_to_onnx()
