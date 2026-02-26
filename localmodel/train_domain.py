"""
Train domain phishing classifier using TF-IDF + Logistic Regression.
Run: python train_domain.py
"""

import os
import json
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
import pickle

OUTPUT_DIR = "output/domain_classifier"


def load_domain_data(filepath="data/domain_reputation.json"):
    """Load domain reputation data from JSON"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    domains = []
    labels = []
    for domain, label in data.items():
        domains.append(domain)
        labels.append(1 if label == "phishing" else 0)
    
    print(f"Loaded {len(domains)} domains")
    phishing_count = sum(labels)
    legitimate_count = len(labels) - phishing_count
    print(f"  Phishing: {phishing_count}, Legitimate: {legitimate_count}")
    
    return domains, labels


def extract_features(domain):
    """Extract handcrafted features from domain"""
    features = {}
    
    # Length features
    features['length'] = len(domain)
    features['num_digits'] = sum(c.isdigit() for c in domain)
    features['num_hyphens'] = domain.count('-')
    features['num_dots'] = domain.count('.')
    features['num_underscores'] = domain.count('_')
    
    # Ratio features
    features['digit_ratio'] = features['num_digits'] / max(len(domain), 1)
    features['hyphen_ratio'] = features['num_hyphens'] / max(len(domain), 1)
    
    # Character type features
    features['has_ip'] = 1 if re.match(r'^\d{1,3}(\.\d{1,3}){3}', domain) else 0
    features['has_port'] = 1 if ':' in domain else 0
    
    # TLD features
    tld = domain.split('.')[-1].lower() if '.' in domain else ''
    suspicious_tlds = ['xyz', 'top', 'club', 'win', 'work', 'tk', 'ml', 'ga', 'cf', 'gq', 'pw', 'cc', 'ws']
    features['suspicious_tld'] = 1 if tld in suspicious_tlds else 0
    features['is_common_tld'] = 1 if tld in ['com', 'org', 'net', 'edu', 'gov'] else 0
    
    # Subdomain depth
    features['subdomain_depth'] = domain.count('.')
    
    # Known patterns
    features['has_www'] = 1 if domain.startswith('www.') else 0
    features['has_numbers_in_domain'] = 1 if re.search(r'[a-z]+\d+[a-z]', domain) else 0
    
    # Entropy (randomness indicator)
    char_counts = {}
    for c in domain:
        char_counts[c] = char_counts.get(c, 0) + 1
    entropy = -sum((c/len(domain)) * np.log2(c/len(domain)) for c in char_counts.values() if c > 0)
    features['entropy'] = entropy
    
    return features


def create_feature_matrix(domains):
    """Create feature matrix with TF-IDF + handcrafted features"""
    # TF-IDF with character n-grams
    print("Creating TF-IDF features...")
    tfidf = TfidfVectorizer(
        analyzer='char',
        ngram_range=(2, 5),
        max_features=1000,
        lowercase=True
    )
    tfidf_features = tfidf.fit_transform(domains)
    print(f"  TF-IDF features: {tfidf_features.shape[1]}")
    
    # Handcrafted features
    print("Extracting handcrafted features...")
    handcrafted = []
    for domain in domains:
        feats = extract_features(domain)
        handcrafted.append([
            feats['length'],
            feats['num_digits'],
            feats['num_hyphens'],
            feats['num_dots'],
            feats['num_underscores'],
            feats['digit_ratio'],
            feats['hyphen_ratio'],
            feats['has_ip'],
            feats['has_port'],
            feats['suspicious_tld'],
            feats['is_common_tld'],
            feats['subdomain_depth'],
            feats['has_www'],
            feats['has_numbers_in_domain'],
            feats['entropy']
        ])
    
    handcrafted = np.array(handcrafted)
    handcrafted = np.atleast_2d(handcrafted)
    print(f"  Handcrafted features: {handcrafted.shape[1]}")
    
    # Scale handcrafted features
    scaler = StandardScaler()
    handcrafted_scaled = scaler.fit_transform(handcrafted)
    
    # Combine features
    features = hstack([tfidf_features, csr_matrix(handcrafted_scaled)])
    print(f"Total features: {features.shape[1]}")
    
    return features, tfidf, scaler


def train():
    print("=" * 50)
    print("Training Domain Phishing Classifier")
    print("=" * 50)
    
    # Load data
    domains, labels = load_domain_data("data/domain_reputation.json")
    
    # Create features
    features, tfidf, scaler = create_feature_matrix(domains)
    labels = np.array(labels)
    
    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\nTraining samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    
    # Train model
    print("\nTraining Logistic Regression...")
    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        C=1.0,
        random_state=42,
        solver='lbfgs'
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    
    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_val, val_preds)
    
    print(f"\nTraining Accuracy: {train_acc*100:.1f}%")
    print(f"Validation Accuracy: {val_acc*100:.1f}%")
    
    print("\nValidation Classification Report:")
    print(classification_report(y_val, val_preds, target_names=["legitimate", "phishing"]))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_val, val_preds)
    print(f"  True Legitimate: {cm[0,0]}, False Phishing: {cm[0,1]}")
    print(f"  False Legitimate: {cm[1,0]}, True Phishing: {cm[1,1]}")
    
    # Feature importance (top n-grams)
    print("\nTop 10 important n-grams for phishing detection:")
    feature_names = tfidf.get_feature_names_out().tolist() + [
        'length', 'num_digits', 'num_hyphens', 'num_dots', 'num_underscores',
        'digit_ratio', 'hyphen_ratio', 'has_ip', 'has_port', 'suspicious_tld',
        'is_common_tld', 'subdomain_depth', 'has_www', 'has_numbers_in_domain', 'entropy'
    ]
    
    # Get coefficients for phishing class
    coefs = model.coef_[0]
    top_indices = np.argsort(coefs)[-10:][::-1]
    for idx in top_indices:
        print(f"  {feature_names[idx]}: {coefs[idx]:.3f}")
    
    # Save model
    print(f"\nSaving model to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(f"{OUTPUT_DIR}/model.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    with open(f"{OUTPUT_DIR}/tfidf.pkl", 'wb') as f:
        pickle.dump(tfidf, f)
    
    with open(f"{OUTPUT_DIR}/scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Training completed!")
    print(f"Model saved to {OUTPUT_DIR}/")
    
    return model, tfidf, scaler


if __name__ == "__main__":
    train()
