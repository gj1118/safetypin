"""
Export trained PyTorch image classifier to ONNX format for efficient inference.
Run this after training with: python export_image.py
"""

import torch
import torch.nn as nn
from torchvision import models
import os

# Configuration
IMAGE_SIZE = 128  # Match training configuration
os.makedirs("output/image", exist_ok=True)

class EfficientImageClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super().__init__()

        # Use ResNet18 for memory efficiency (matches training)
        self.backbone = models.resnet18(weights=None)  # No pretrained weights needed for export

        # Simpler classifier head for memory efficiency (matches training)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

print("Loading trained model...")
model = EfficientImageClassifier()

# Load from the new training output location
model_path = 'output/image/image_classifier.pth'
if not os.path.exists(model_path):
    # Fallback to old location
    model_path = 'image_classifier.pth'

model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

print(f"Exporting model to ONNX format (input size: {IMAGE_SIZE}x{IMAGE_SIZE})...")

# Export with the correct input size (128x128 instead of 224x224)
torch.onnx.export(
    model,
    torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE),  # Match training input size
    'output/image/image_classifier.onnx',
    input_names=['input'],
    output_names=['output'],
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    dynamo=False
)

print('✅ Image classifier exported successfully!')
print(f'   - Location: output/image/image_classifier.onnx')
print(f'   - Input size: {IMAGE_SIZE}x{IMAGE_SIZE}')
print(f'   - Model: ResNet18-Efficient')
print('\nYour server can now use this optimized ONNX model for fast inference.')
