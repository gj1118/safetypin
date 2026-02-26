"""
Train text classifier with reason prediction using DistilBERT.
Multi-output model that predicts both classification (good/bad) and reason category.
Run: python train_text.py
"""

import os
import json
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from urllib.parse import urlparse

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
NUM_EPOCHS = 15
LEARNING_RATE = 3e-5  # Back to original
USE_DOMAIN_FEATURES = False  # Disable domain features for now

print(f"Using device: {DEVICE}")

# Reason categories mapping
REASON_CATEGORIES = {
    "positive_interaction": 0,
    "educational_content": 1,
    "bullying_harassment": 2,
    "threats_violence": 3,
    "dangerous_weapons": 4,
    "inappropriate_behavior": 5,
    "secrecy_manipulation": 6,
    "personal_info_request": 7,
    "contact_info_sharing": 8,
    "predatory_behavior": 9,
    "inappropriate_content": 10,
    "animals_wildlife": 11,
    "sports_recreation": 12,
    "music_entertainment": 13,
    "food_cooking": 14,
    "art_creative": 15,
    "health_medical": 16,
    "travel_geography": 17,
    "news_current_events": 18,
    "self_harm": 19,
    "substance_abuse": 20,
    "gambling": 21,
    "scam_fraud": 22,
    "hate_speech": 23,
    "spam": 24
}

REASON_NAMES = {v: k for k, v in REASON_CATEGORIES.items()}

# Load domain reputation mapping
def load_domain_reputation(filepath="data/domain_reputation.json"):
    """Load domain reputation mapping from phishing dataset"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Domain reputation file not found at {filepath}")
        print("Run download_phishing_data.py first to download phishing dataset")
        return {}

def extract_urls_from_text(text):
    """Extract URLs from text content"""
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        r'|(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:/[^\s]*)?'
    )
    urls = url_pattern.findall(text)
    return [url.strip() for url in urls if len(url) > 5]

def extract_domain(url):
    """Extract domain from URL"""
    try:
        parsed = urlparse(url if url.startswith(('http://', 'https://')) else f'http://{url}')
        return parsed.netloc.lower()
    except:
        return url.lower()

def get_platform_type(domain):
    """Classify platform type for risk assessment"""
    if not domain:
        return "unknown"

    social_platforms = ['facebook.com', 'instagram.com', 'twitter.com', 'tiktok.com',
                       'snapchat.com', 'discord.com', 'reddit.com', 'youtube.com', 'twitch.tv']
    news_media = ['bbc.com', 'cnn.com', 'reuters.com', 'ap.org', 'npr.org']
    educational = ['wikipedia.org', 'wikimedia.org']

    for platform in social_platforms:
        if platform in domain:
            return "social_platform"

    for news in news_media:
        if news in domain:
            return "news_media"

    for edu in educational:
        if edu in domain:
            return "educational"

    if domain.endswith('.edu') or domain.endswith('.gov'):
        return "institutional"

    return "unknown"

def analyze_domain_context(text, domain_reputation):
    """Analyze domain context and behavioral patterns"""
    urls = extract_urls_from_text(text)

    if not urls:
        return {
            'has_urls': False,
            'domain_risk_score': 0.0,
            'platform_type': 'none',
            'confidence_threshold': 0.75  # Default threshold for text-only
        }

    domain_risks = []
    platform_types = []

    for url in urls:
        domain = extract_domain(url)
        reputation = domain_reputation.get(domain, 'unknown')
        platform_type = get_platform_type(domain)

        # Calculate risk score based on reputation
        if reputation == 'phishing':
            risk_score = 0.9  # Very high risk
        elif reputation == 'legitimate':
            risk_score = 0.2  # Low risk
        else:
            risk_score = 0.5  # Unknown domain

        domain_risks.append(risk_score)
        platform_types.append(platform_type)

    avg_domain_risk = np.mean(domain_risks)
    primary_platform_type = max(set(platform_types), key=platform_types.count)

    # Set confidence threshold based on context
    if primary_platform_type == 'social_platform':
        confidence_threshold = 0.6  # Lower threshold for social platforms
    elif primary_platform_type in ['educational', 'news_media', 'institutional']:
        confidence_threshold = 0.9  # Higher threshold for trusted content
    elif avg_domain_risk > 0.7:
        confidence_threshold = 0.5  # Very low threshold for suspicious domains
    else:
        confidence_threshold = 0.75  # Default

    return {
        'has_urls': True,
        'domain_risk_score': avg_domain_risk,
        'platform_type': primary_platform_type,
        'confidence_threshold': confidence_threshold,
        'urls_found': len(urls)
    }

# Load domain reputation
print("Loading domain reputation data...")
DOMAIN_REPUTATION = load_domain_reputation()


class TextWithReasonsDataset(Dataset):
    def __init__(self, samples, tokenizer, domain_reputation, max_length=256):
        self.samples = samples
        self.tokenizer = tokenizer
        self.domain_reputation = domain_reputation
        self.max_length = max_length
        print(f"Dataset loaded with {len(self.samples)} samples")

        bad_count = sum(1 for _, label, _ in self.samples if label == 0)
        good_count = sum(1 for _, label, _ in self.samples if label == 1)
        print(f"  Bad: {bad_count}, Good: {good_count}")

        reason_counts = {}
        for _, _, reason in self.samples:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        print("Reason distribution:")
        for reason, count in sorted(reason_counts.items()):
            print(f"  {REASON_NAMES[reason]}: {count}")

        # Analyze domain context for training samples
        self._analyze_domain_context()

    def _analyze_domain_context(self):
        """Analyze domain context for all samples"""
        domain_stats = {'with_urls': 0, 'without_urls': 0, 'platform_types': {}}

        for text, _, _ in self.samples:
            context = analyze_domain_context(text, self.domain_reputation)
            if context['has_urls']:
                domain_stats['with_urls'] += 1
                platform_type = context['platform_type']
                domain_stats['platform_types'][platform_type] = domain_stats['platform_types'].get(platform_type, 0) + 1
            else:
                domain_stats['without_urls'] += 1

        print(f"Domain context analysis:")
        print(f"  Samples with URLs: {domain_stats['with_urls']}")
        print(f"  Samples without URLs: {domain_stats['without_urls']}")
        print(f"  Platform type distribution: {domain_stats['platform_types']}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label, reason = self.samples[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Analyze domain context for this sample
        domain_context = analyze_domain_context(text, self.domain_reputation)

        # Create domain features tensor
        domain_features = torch.tensor([
            1.0 if domain_context['has_urls'] else 0.0,  # has_urls (binary)
            domain_context['domain_risk_score'],  # domain_risk_score (0-1)
            float(get_platform_type_encoding(domain_context['platform_type']))  # platform_type (0-5)
        ], dtype=torch.float)

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "domain_features": domain_features,
            "labels": torch.tensor(label, dtype=torch.long),
            "reasons": torch.tensor(reason, dtype=torch.long)
        }

def get_platform_type_encoding(platform_type):
    """Get numerical encoding for platform type"""
    platform_types = {
        'none': 0, 'social_platform': 1, 'news_media': 2,
        'educational': 3, 'institutional': 4, 'unknown': 5
    }
    return platform_types.get(platform_type, platform_types['unknown'])


class DistilBertWithReasons(nn.Module):
    def __init__(self, num_labels=2, num_reasons=11, num_platform_types=6):
        super().__init__()
        self.num_labels = num_labels
        self.num_reasons = num_reasons
        self.num_platform_types = num_platform_types

        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.6)

        # Domain context features (3 features: has_urls, domain_risk_score, platform_type_encoded)
        domain_feature_size = 3

        # Feature size (BERT only or BERT + domain context)
        if USE_DOMAIN_FEATURES:
            feature_size = self.distilbert.config.hidden_size + domain_feature_size
        else:
            feature_size = self.distilbert.config.hidden_size

        # Classification head (good/bad) - back to simpler original architecture
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_labels)
        )

        # Reason prediction head - back to simpler original architecture
        self.reason_classifier = nn.Sequential(
            nn.Linear(feature_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_reasons)
        )

        # Platform type encoding
        self.platform_types = {
            'none': 0, 'social_platform': 1, 'news_media': 2,
            'educational': 3, 'institutional': 4, 'unknown': 5
        }

    def forward(self, input_ids, attention_mask, domain_features=None, labels=None, reasons=None):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
        pooled_output = self.dropout(pooled_output)

        # Combine BERT features with domain features (if enabled)
        if USE_DOMAIN_FEATURES and domain_features is not None:
            combined_features = torch.cat([pooled_output, domain_features], dim=1)
        else:
            # Use only BERT features (original model behavior)
            combined_features = pooled_output

        # Classification logits with domain context
        classification_logits = self.classifier(combined_features)

        # Reason logits with domain context
        reason_logits = self.reason_classifier(combined_features)

        total_loss = None
        if labels is not None and reasons is not None:
            loss_fct = nn.CrossEntropyLoss()
            classification_loss = loss_fct(classification_logits, labels)
            reason_loss = loss_fct(reason_logits, reasons)
            # Weight the losses properly (classification is primary, reason is secondary)
            total_loss = 0.9 * classification_loss + 0.1 * reason_loss

        return {
            "loss": total_loss,
            "classification_logits": classification_logits,
            "reason_logits": reason_logits
        }

    def encode_platform_type(self, platform_type):
        """Encode platform type as numerical value"""
        return self.platform_types.get(platform_type, self.platform_types['unknown'])


def evaluate_model(model, data_loader, device):
    model.eval()
    all_class_predictions = []
    all_class_labels = []
    all_reason_predictions = []
    all_reason_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            domain_features = batch["domain_features"].to(device)
            labels = batch["labels"].to(device)
            reasons = batch["reasons"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                          domain_features=domain_features, labels=labels, reasons=reasons)
            loss = outputs["loss"]
            total_loss += loss.item()

            # Classification predictions
            _, class_predicted = outputs["classification_logits"].max(1)
            all_class_predictions.extend(class_predicted.cpu().numpy())
            all_class_labels.extend(labels.cpu().numpy())

            # Reason predictions
            _, reason_predicted = outputs["reason_logits"].max(1)
            all_reason_predictions.extend(reason_predicted.cpu().numpy())
            all_reason_labels.extend(reasons.cpu().numpy())

    class_accuracy = accuracy_score(all_class_labels, all_class_predictions)
    reason_accuracy = accuracy_score(all_reason_labels, all_reason_predictions)
    return class_accuracy, reason_accuracy, total_loss / len(data_loader), all_class_predictions, all_class_labels, all_reason_predictions, all_reason_labels


def train():
    print("="*50)
    print("Training DistilBERT Multi-Output Classifier")
    print("="*50)

    # Load data
    with open("data/train.json") as f:
        data = json.load(f)

    samples = [
        (item["text"], 0 if item["label"] == "bad" else 1, REASON_CATEGORIES[item["reason"]])
        for item in data
    ]

    # Train/validation split
    train_samples, val_samples = train_test_split(samples, test_size=0.2, random_state=42,
                                                  stratify=[s[1] for s in samples])

    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    train_dataset = TextWithReasonsDataset(train_samples, tokenizer, DOMAIN_REPUTATION, max_length=256)
    val_dataset = TextWithReasonsDataset(val_samples, tokenizer, DOMAIN_REPUTATION, max_length=256)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = DistilBertWithReasons(num_labels=2, num_reasons=len(REASON_CATEGORIES)).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    best_class_accuracy = 0
    best_model_state = None
    patience = 10
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 30)

        # Training
        model.train()
        total_loss = 0
        class_correct = 0
        reason_correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            domain_features = batch["domain_features"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            reasons = batch["reasons"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                          domain_features=domain_features, labels=labels, reasons=reasons)
            loss = outputs["loss"]
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            # Calculate accuracies
            _, class_predicted = outputs["classification_logits"].max(1)
            _, reason_predicted = outputs["reason_logits"].max(1)
            class_correct += class_predicted.eq(labels).sum().item()
            reason_correct += reason_predicted.eq(reasons).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 5 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        train_class_accuracy = 100. * class_correct / total
        train_reason_accuracy = 100. * reason_correct / total
        print(f"Training - Loss: {total_loss:.4f}, Class Acc: {train_class_accuracy:.1f}%, Reason Acc: {train_reason_accuracy:.1f}%")

        # Validation
        val_class_acc, val_reason_acc, val_loss, val_class_preds, val_class_labels, val_reason_preds, val_reason_labels = evaluate_model(model, val_loader, DEVICE)
        print(f"Validation - Loss: {val_loss:.4f}, Class Acc: {val_class_acc*100:.1f}%, Reason Acc: {val_reason_acc*100:.1f}%")

        # Save best model based on classification accuracy
        if val_class_acc > best_class_accuracy:
            best_class_accuracy = val_class_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"New best classification accuracy: {val_class_acc*100:.1f}%")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

        # Step learning rate scheduler
        scheduler.step()

    # Load best model and save
    model.load_state_dict(best_model_state)
    os.makedirs("output/bert_text_with_reasons", exist_ok=True)

    # Save the model state dict and tokenizer
    torch.save(model.state_dict(), "output/bert_text_with_reasons/model.pth")
    tokenizer.save_pretrained("output/bert_text_with_reasons")

    # Save reason categories mapping
    with open("output/bert_text_with_reasons/reason_categories.json", "w") as f:
        json.dump(REASON_CATEGORIES, f, indent=2)

    print(f"\nTraining completed!")
    print(f"Best classification accuracy: {best_class_accuracy*100:.1f}%")
    print("Model saved to output/bert_text_with_reasons")

    # Final evaluation with classification report
    val_class_acc, val_reason_acc, _, val_class_preds, val_class_labels, val_reason_preds, val_reason_labels = evaluate_model(model, val_loader, DEVICE)

    print("\nFinal Classification Report (Good/Bad):")
    print(classification_report(val_class_labels, val_class_preds, target_names=["bad", "good"]))

    print("\nFinal Reason Classification Report:")
    # Get only the unique reason indices that actually appear in validation data
    unique_reasons = sorted(set(list(val_reason_labels) + list(val_reason_preds)))
    reason_names_list = [REASON_NAMES[i] for i in unique_reasons]
    print(classification_report(val_reason_labels, val_reason_preds, target_names=reason_names_list))


if __name__ == "__main__":
    train()