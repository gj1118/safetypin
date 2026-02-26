"""
Hybrid domain classifier using ONNX model + domain reputation lookup.
"""

import os
import re
import json
import numpy as np
import onnxruntime as ort

MODEL_DIR = "output/domain_classifier_cnn"
ONNX_PATH = "output/text_classifier_domain.onnx"

_session = None
_char_to_idx = None
_max_length = None
_reputation_data = None


def _load_model():
    """Lazy load ONNX model"""
    global _session, _char_to_idx, _max_length
    
    if _session is None:
        # Load ONNX session
        _session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
        
        # Load metadata
        with open(f"{MODEL_DIR}/metadata.json") as f:
            metadata = json.load(f)
        _char_to_idx = metadata['char_to_idx']
        _max_length = metadata['max_length']
        
        print(f"Loaded ONNX domain classifier from {ONNX_PATH}")
    
    return _session, _char_to_idx, _max_length


def _load_reputation():
    """Load domain reputation data"""
    global _reputation_data
    
    if _reputation_data is None:
        for filename in ["data/domain_reputation_augmented.json", "data/domain_reputation.json"]:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    _reputation_data = json.load(f)
                print(f"Loaded reputation data from {filename}")
                break
    
    return _reputation_data


def _clean_domain(domain):
    """Clean and normalize domain"""
    domain = domain.lower().strip()
    domain = re.sub(r'^https?://', '', domain)
    domain = domain.split('/')[0]
    domain = domain.split(':')[0]
    return domain


def _encode_domain(domain, char_to_idx, max_len):
    """Encode domain to numpy array"""
    indices = [char_to_idx.get(c, 1) for c in domain[:max_len]]
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))
    return np.array([indices], dtype=np.int64)


def _check_reputation(domain):
    """Check domain in reputation database"""
    reputation = _load_reputation()
    if reputation is None:
        return False, None
    
    if domain in reputation:
        return True, reputation[domain] == "phishing"
    
    if domain.startswith('www.'):
        domain_no_www = domain[4:]
        if domain_no_www in reputation:
            return True, reputation[domain_no_www] == "phishing"
    
    return False, None


def predict_domain(domain, use_reputation=True):
    """
    Predict if a domain is phishing.
    
    Args:
        domain: Domain to check
        use_reputation: Whether to use reputation database lookup
    
    Returns:
        dict with:
            - domain: cleaned domain
            - is_phishing: boolean prediction
            - confidence: overall confidence
            - model_prediction: from ONNX model
            - model_confidence: ONNX model confidence
            - reputation_match: if domain found in reputation DB
            - reputation_result: "phishing", "legitimate", or None
    """
    session, char_to_idx, max_length = _load_model()
    domain = _clean_domain(domain)
    
    # Get ONNX model prediction
    encoded = _encode_domain(domain, char_to_idx, max_length)
    output = session.run(None, {'input': encoded})[0][0]
    
    # Apply softmax
    exp_output = np.exp(output - np.max(output))
    probs = exp_output / exp_output.sum()
    
    model_conf = float(probs[1])  # Probability of phishing
    model_pred = 1 if model_conf > 0.5 else 0
    
    result = {
        "domain": domain,
        "model_prediction": bool(model_pred),
        "model_confidence": model_conf,
        "model_probabilities": {
            "legitimate": float(probs[0]),
            "phishing": float(probs[1])
        }
    }
    
    # Add reputation check
    if use_reputation:
        is_known, rep_result = _check_reputation(domain)
        result["reputation_match"] = is_known
        result["reputation_result"] = rep_result
        
        if is_known and rep_result is not None:
            result["is_phishing"] = rep_result
            result["confidence"] = 0.99 if rep_result else 0.95
            result["prediction_source"] = "reputation_database"
        else:
            result["is_phishing"] = bool(model_pred)
            result["confidence"] = max(model_conf, 1 - model_conf)
            result["prediction_source"] = "onnx_model"
    else:
        result["is_phishing"] = bool(model_pred)
        result["confidence"] = max(model_conf, 1 - model_conf)
        result["prediction_source"] = "onnx_model"
    
    return result


def predict_batch(domains):
    """Predict for multiple domains"""
    return [predict_domain(d) for d in domains]


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        domains = sys.argv[1:]
    else:
        domains = [
            "paypal.com",
            "apple.com",
            "paypal-verify-account.com",
            "google.com",
            "apple-id-verify.com",
        ]
    
    print("Testing ONNX domain classifier...\n")
    for domain in domains:
        result = predict_domain(domain)
        print(f"Domain: {result['domain']}")
        print(f"  Source: {result['prediction_source']}")
        print(f"  Is Phishing: {result['is_phishing']}")
        print(f"  Confidence: {result['confidence']*100:.1f}%")
        if result.get('reputation_match'):
            print(f"  Reputation: {result['reputation_result']}")
        print(f"  Model Probabilities: Legit {result['model_probabilities']['legitimate']*100:.1f}%, Phishing {result['model_probabilities']['phishing']*100:.1f}%")
        print()
