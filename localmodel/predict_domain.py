"""
Domain phishing classifier prediction module.
Usage:
    from predict_domain import predict_domain
    
    result = predict_domain("paypal-fake.com")
    # Returns: {"domain": "...", "is_phishing": True, "confidence": 0.85}
"""

import os
import re
import pickle
import numpy as np
from scipy.sparse import csr_matrix, hstack

MODEL_DIR = "output/domain_classifier"

_model = None
_tfidf = None
_scaler = None


def _load_model():
    """Lazy load model on first use"""
    global _model, _tfidf, _scaler
    
    if _model is None:
        model_path = os.path.join(MODEL_DIR, "model.pkl")
        tfidf_path = os.path.join(MODEL_DIR, "tfidf.pkl")
        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Run 'python train_domain.py' first."
            )
        
        with open(model_path, 'rb') as f:
            _model = pickle.load(f)
        
        with open(tfidf_path, 'rb') as f:
            _tfidf = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            _scaler = pickle.load(f)
        
        print(f"Loaded domain classifier from {MODEL_DIR}")
    
    return _model, _tfidf, _scaler


def _extract_features(domain):
    """Extract handcrafted features from domain"""
    features = []
    
    length = len(domain)
    num_digits = sum(c.isdigit() for c in domain)
    num_hyphens = domain.count('-')
    num_dots = domain.count('.')
    num_underscores = domain.count('_')
    
    features.extend([
        length,
        num_digits,
        num_hyphens,
        num_dots,
        num_underscores,
        num_digits / max(length, 1),
        num_hyphens / max(length, 1),
        1 if re.match(r'^\d{1,3}(\.\d{1,3}){3}', domain) else 0,
        1 if ':' in domain else 0,
        1 if domain.split('.')[-1].lower() in ['xyz', 'top', 'club', 'win', 'work', 'tk', 'ml', 'ga', 'cf', 'gq', 'pw', 'cc', 'ws'] else 0,
        1 if domain.split('.')[-1].lower() in ['com', 'org', 'net', 'edu', 'gov'] else 0,
        domain.count('.'),
        1 if domain.startswith('www.') else 0,
        1 if re.search(r'[a-z]+\d+[a-z]', domain) else 0,
    ])
    
    # Entropy
    char_counts = {}
    for c in domain:
        char_counts[c] = char_counts.get(c, 0) + 1
    entropy = -sum((c/length) * np.log2(c/length) for c in char_counts.values() if c > 0)
    features.append(entropy)
    
    return np.array(features).reshape(1, -1)


def predict_domain(domain):
    """
    Predict if a domain is phishing or legitimate.
    
    Args:
        domain: Domain string to classify (e.g., "paypal-fake.com")
    
    Returns:
        dict with keys:
            - domain: The input domain
            - is_phishing: Boolean, True if predicted as phishing
            - confidence: Float between 0 and 1, probability of prediction
            - probabilities: Dict with "legitimate" and "phishing" probabilities
    """
    model, tfidf, scaler = _load_model()
    
    # Clean domain
    domain = domain.lower().strip()
    if not domain.startswith(('http://', 'https://')):
        domain = domain
    else:
        # Extract domain from URL
        if '://' in domain:
            domain = domain.split('://')[1].split('/')[0]
    
    # Transform domain to features
    tfidf_features = tfidf.transform([domain])
    handcrafted = scaler.transform(_extract_features(domain))
    
    features = hstack([tfidf_features, csr_matrix(handcrafted)])
    
    # Predict
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    return {
        "domain": domain,
        "is_phishing": bool(prediction),
        "confidence": float(max(probabilities)),
        "probabilities": {
            "legitimate": float(probabilities[0]),
            "phishing": float(probabilities[1])
        }
    }


def predict_batch(domains):
    """
    Predict phishing for multiple domains.
    
    Args:
        domains: List of domain strings
    
    Returns:
        List of dicts with prediction results
    """
    return [predict_domain(d) for d in domains]


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        domain = sys.argv[1]
    else:
        domain = input("Enter domain to check: ")
    
    result = predict_domain(domain)
    
    print(f"\nDomain: {result['domain']}")
    print(f"Is Phishing: {result['is_phishing']}")
    print(f"Confidence: {result['confidence']*100:.1f}%")
    print(f"  Legitimate: {result['probabilities']['legitimate']*100:.1f}%")
    print(f"  Phishing: {result['probabilities']['phishing']*100:.1f}%")
