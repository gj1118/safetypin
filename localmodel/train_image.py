"""
Train image classifier using modern ResNet with enhanced data augmentation.
Optimized for child safety image classification (good/bad).
Run: python train_image_improved.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration - Optimized for memory efficiency
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16  # Reduced from 32 to save memory
NUM_EPOCHS = 25
INITIAL_LR = 1e-4
MIN_LR = 1e-6
PATIENCE = 5
WEIGHT_DECAY = 1e-4
IMAGE_SIZE = 128  # Reduced from 224 to significantly save RAM

print(f"Using device: {DEVICE}")
print(f"Training configuration:")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Initial LR: {INITIAL_LR}")
print(f"  Device: {DEVICE}")


class ImageDataset(Dataset):
    def __init__(self, samples, transform=None, augment_unsafe=True):
        self.samples = samples
        self.transform = transform
        self.augment_unsafe = augment_unsafe

        print(f"Dataset loaded with {len(self.samples)} images")
        bad_count = sum(1 for _, label in self.samples if label == 0)
        good_count = sum(1 for _, label in self.samples if label == 1)
        print(f"  Bad (unsafe): {bad_count}")
        print(f"  Good (safe): {good_count}")

        if bad_count < good_count and augment_unsafe:
            print(f"  Note: Will apply stronger augmentation to 'bad' images for balance")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        try:
            image = Image.open(path)

            # Handle transparency properly
            if image.mode in ('RGBA', 'LA', 'P'):
                # Create white background for transparent images
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    # Convert palette to RGBA first if it has transparency
                    if 'transparency' in image.info:
                        image = image.convert('RGBA')
                    else:
                        image = image.convert('RGB')

                if image.mode in ('RGBA', 'LA'):
                    background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else image.split()[-1])
                    image = background
                else:
                    image = image.convert("RGB")
            else:
                image = image.convert("RGB")

        except Exception as e:
            logger.warning(f"Failed to load image {path}: {e}")
            # Create a gray placeholder image
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color='gray')

        if self.transform:
            image = self.transform(image)

        return image, label


class EfficientImageClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super().__init__()

        # Use ResNet18 for memory efficiency (much smaller than ResNet50)
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Freeze early layers to prevent overfitting on small datasets
        for param in list(self.backbone.parameters())[:-10]:
            param.requires_grad = False

        # Simpler classifier head for memory efficiency
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


def create_transforms():
    """Create memory-efficient transforms with smaller image sizes"""

    # Memory-efficient training transforms
    train_transform = transforms.Compose([
        transforms.Resize((144, 144)),  # Smaller initial size
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),  # 128x128 final size
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),  # Reduced rotation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),  # Lighter augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Clean validation transforms
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # 128x128
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def load_image_data():
    """Load image data from data/images/good and data/images/bad directories"""
    samples = []
    image_dir = "data/images"

    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

    # Load bad (unsafe) images
    bad_dir = os.path.join(image_dir, "bad")
    if os.path.exists(bad_dir):
        bad_files = [f for f in os.listdir(bad_dir) if f.lower().endswith(supported_formats)]
        for fname in bad_files:
            samples.append((os.path.join(bad_dir, fname), 0))
        print(f"Found {len(bad_files)} bad images")

    # Load good (safe) images
    good_dir = os.path.join(image_dir, "good")
    if os.path.exists(good_dir):
        good_files = [f for f in os.listdir(good_dir) if f.lower().endswith(supported_formats)]
        for fname in good_files:
            samples.append((os.path.join(good_dir, fname), 1))
        print(f"Found {len(good_files)} good images")

    return samples


def evaluate_model(model, data_loader, criterion, device):
    """Comprehensive model evaluation"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            predicted = outputs.argmax(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Store predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = correct / total
    avg_loss = total_loss / len(data_loader)

    return accuracy, avg_loss, all_predictions, all_labels, all_probs


def print_training_summary(epoch, num_epochs, train_loss, train_acc, val_loss, val_acc, lr, is_best=False):
    """Print formatted training summary"""
    status = " ⭐ BEST" if is_best else ""
    print(f"Epoch [{epoch:2d}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:6.2f}% | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:6.2f}% | "
          f"LR: {lr:.6f}{status}")


def train():
    print("="*60)
    print("Training Modern Image Classifier for Child Safety")
    print("="*60)

    # Load and validate data
    all_samples = load_image_data()
    if len(all_samples) == 0:
        print("❌ No image data found!")
        print("Please add images to:")
        print("  - data/images/good/  (safe images)")
        print("  - data/images/bad/   (unsafe images)")
        return

    if len(all_samples) < 10:
        print("⚠️  Warning: Very few images found. Consider adding more data for better performance.")

    # Split data with stratification
    labels_for_split = [sample[1] for sample in all_samples]
    train_samples, val_samples = train_test_split(
        all_samples, test_size=0.2, random_state=42, stratify=labels_for_split
    )

    print(f"\n📊 Data Split:")
    print(f"  Training samples: {len(train_samples)}")
    print(f"  Validation samples: {len(val_samples)}")

    # Create transforms and datasets
    train_transform, val_transform = create_transforms()
    train_dataset = ImageDataset(train_samples, train_transform)
    val_dataset = ImageDataset(val_samples, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Initialize model, criterion, optimizer
    model = EfficientImageClassifier(num_classes=2, dropout_rate=0.3).to(DEVICE)

    # Calculate class weights to handle imbalance and reduce false positives
    # Give higher weight to "bad" class to compensate for fewer examples
    # But also add penalty for false positives on "good" class
    good_weight = 0.8  # Slightly lower weight for good (reduce false negatives)
    bad_weight = 1.5   # Higher weight for bad (but not too high to avoid false positives)
    class_weights = torch.tensor([bad_weight, good_weight], dtype=torch.float32).to(DEVICE)
    print(f"Using class weights: bad={bad_weight}, good={good_weight}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=PATIENCE,
                                min_lr=MIN_LR)

    # Training tracking
    best_val_accuracy = 0
    best_model_state = None
    patience_counter = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print(f"\n🚀 Starting training...")
    training_start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()

        # Training phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_accuracy = 100 * correct / total
        train_loss = total_loss / len(train_loader)

        # Validation phase
        val_accuracy, val_loss, val_preds, val_labels, val_probs = evaluate_model(
            model, val_loader, criterion, DEVICE
        )
        val_accuracy_pct = val_accuracy * 100

        # Learning rate scheduling
        scheduler.step(val_accuracy)
        current_lr = optimizer.param_groups[0]['lr']

        # Track metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy_pct)

        # Check for best model
        is_best = val_accuracy > best_val_accuracy
        if is_best:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        # Print progress
        print_training_summary(epoch + 1, NUM_EPOCHS, train_loss, train_accuracy,
                             val_loss, val_accuracy_pct, current_lr, is_best)

        # Early stopping
        if patience_counter >= PATIENCE * 2:
            print(f"\n⏹️  Early stopping triggered after {patience_counter} epochs without improvement")
            break

    # Training complete
    training_time = time.time() - training_start_time
    print(f"\n✅ Training completed in {training_time:.2f} seconds")

    # Load best model and save
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Create output directory and save model
    os.makedirs("output/image", exist_ok=True)
    torch.save(model.state_dict(), "output/image/image_classifier.pth")

    # Save model info
    model_info = {
        "architecture": "ResNet18-Efficient",
        "num_classes": 2,
        "input_size": IMAGE_SIZE,
        "best_accuracy": float(best_val_accuracy),
        "total_epochs": epoch + 1,
        "training_samples": len(train_samples),
        "validation_samples": len(val_samples)
    }

    import json
    with open("output/image/model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)

    print(f"\n📊 Final Results:")
    print(f"  Best validation accuracy: {best_val_accuracy*100:.2f}%")
    print(f"  Model saved to: output/image/image_classifier.pth")

    # Final evaluation with detailed metrics
    val_accuracy, _, val_preds, val_labels, val_probs = evaluate_model(model, val_loader, criterion, DEVICE)

    print(f"\n📋 Final Classification Report:")
    print(classification_report(val_labels, val_preds, target_names=["unsafe", "safe"], digits=4))

    print(f"\n🔍 Confusion Matrix:")
    cm = confusion_matrix(val_labels, val_preds)
    print("       Predicted")
    print("       Unsafe  Safe")
    print(f"Unsafe   {cm[0,0]:4d}   {cm[0,1]:4d}")
    print(f"Safe     {cm[1,0]:4d}   {cm[1,1]:4d}")

    # Calculate confidence statistics
    confidences = [max(prob) for prob in val_probs]
    print(f"\n📈 Confidence Statistics:")
    print(f"  Average confidence: {np.mean(confidences):.3f}")
    print(f"  Min confidence: {np.min(confidences):.3f}")
    print(f"  Max confidence: {np.max(confidences):.3f}")

    logger.info(f"Training completed successfully. Best accuracy: {best_val_accuracy*100:.2f}%")


if __name__ == "__main__":
    train()