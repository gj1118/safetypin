"""
Classifier server with DistilBERT text + ONNX image.
Enhanced with security, validation, and monitoring.
"""

import os
import base64
import io
import time
import logging
import json
from collections import defaultdict, deque
import numpy as np
from flask import Flask, request, jsonify, g
from PIL import Image
import torch
import onnxruntime as ort
from torchvision import transforms
from transformers import DistilBertTokenizer
from functools import wraps

# HTML classifier import
try:
    from html_classifier import HTMLPageClassifier
    HTML_CLASSIFIER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"HTML classifier not available: {e}")
    HTML_CLASSIFIER_AVAILABLE = False

# Three-stage safety analyzer import
try:
    from two_stage_safety_analyzer import ThreeStageSafetyAnalyzer
    THREE_STAGE_ANALYZER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Three-stage analyzer not available: {e}")
    THREE_STAGE_ANALYZER_AVAILABLE = False

# PDF classifier import
try:
    from pdf_classifier import PDFClassifier
    PDF_CLASSIFIER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"PDF classifier not available: {e}")
    PDF_CLASSIFIER_AVAILABLE = False

# Domain classifier import
try:
    from predict_domain_cnn import predict_domain as predict_domain_cnn
    DOMAIN_CLASSIFIER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Domain classifier not available: {e}")
    DOMAIN_CLASSIFIER_AVAILABLE = False

app = Flask(__name__)

# Configuration
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_TEXT_LENGTH = 2000  # Characters
RATE_LIMIT_REQUESTS = 100  # Requests per minute per IP
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for classification

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('classifier_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Rate limiting storage
request_counts = defaultdict(deque)


def rate_limit(max_requests=RATE_LIMIT_REQUESTS):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_ip = request.remote_addr
            now = time.time()
            minute_ago = now - 60

            # Clean old requests
            while request_counts[client_ip] and request_counts[client_ip][0] < minute_ago:
                request_counts[client_ip].popleft()

            # Check rate limit
            if len(request_counts[client_ip]) >= max_requests:
                logger.warning(f"Rate limit exceeded for IP: {client_ip}")
                return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429

            # Add current request
            request_counts[client_ip].append(now)
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def validate_image_data(image_data):
    """Validate image input"""
    if not image_data:
        return False, "No image data provided"

    try:
        # Handle data URL format
        if isinstance(image_data, str) and image_data.startswith("data:image"):
            image_data = image_data.split(",")[1]

        # Decode and validate size
        image_bytes = base64.b64decode(image_data)
        if len(image_bytes) > MAX_IMAGE_SIZE:
            return False, f"Image too large. Maximum size: {MAX_IMAGE_SIZE/1024/1024:.1f}MB"

        # Validate image format
        image = Image.open(io.BytesIO(image_bytes))
        if image.format not in ['JPEG', 'PNG', 'JPG']:
            return False, "Unsupported image format. Use JPEG or PNG"

        return True, None
    except Exception as e:
        return False, f"Invalid image data: {str(e)}"


def validate_text_data(text):
    """Validate text input"""
    if not text or not isinstance(text, str):
        return False, "No text data provided"

    if len(text) > MAX_TEXT_LENGTH:
        return False, f"Text too long. Maximum length: {MAX_TEXT_LENGTH} characters"

    # Basic content validation (no scripts, etc.)
    if any(tag in text.lower() for tag in ['<script', '<iframe', '<object', '<embed']):
        return False, "Invalid text content detected"

    return True, None


try:
    print("Loading image classifier...")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    image_session = ort.InferenceSession(
        "output/image/image_classifier.onnx",
        sess_options,
        providers=["CPUExecutionProvider"]
    )

    image_transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Match training size for memory efficiency
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    logger.info("Image classifier loaded successfully")
    IMAGE_MODEL_AVAILABLE = True
except Exception as e:
    logger.error(f"Failed to load image classifier: {e}")
    IMAGE_MODEL_AVAILABLE = False
    image_session = None



try:
    print("Loading text classifier (ONNX)...")

    # Load ONNX text classifier
    if os.path.exists("output/text_classifier.onnx"):
        # Setup ONNX session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        text_session = ort.InferenceSession("output/text_classifier.onnx", sess_options)

        # Load tokenizer
        model_path = "output/bert_text_with_reasons"
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)

        # Load reason categories
        with open(f"{model_path}/reason_categories.json") as f:
            reason_categories = json.load(f)
        reason_names = {v: k for k, v in reason_categories.items()}

        logger.info("ONNX text classifier loaded successfully")
        TEXT_MODEL_AVAILABLE = True
        TEXT_MODEL_HAS_REASONS = True
        text_model = None  # Not needed for ONNX

    else:
        logger.error("ONNX text classifier not found at output/text_classifier.onnx")
        TEXT_MODEL_AVAILABLE = False
        TEXT_MODEL_HAS_REASONS = False
        text_session = None
        tokenizer = None
        reason_names = {}

except Exception as e:
    logger.error(f"Failed to load text classifier: {e}")
    TEXT_MODEL_AVAILABLE = False
    TEXT_MODEL_HAS_REASONS = False
    text_session = None
    tokenizer = None
    reason_names = {}


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


def analyze_domain_reputation(url):
    """Analyze domain reputation using loaded domain data"""
    from urllib.parse import urlparse
    try:
        domain = urlparse(url).netloc.lower()
        # Remove www prefix for consistency
        if domain.startswith('www.'):
            domain = domain[4:]

        domain_info = domain_reputation.get(domain, {})

        if domain_info:
            domain_risk = domain_info.get('risk_score', 0.5)
            domain_category = domain_info.get('category', 'unknown')
            return {
                'domain': domain,
                'risk_score': domain_risk,
                'category': domain_category,
                'in_database': True
            }
        else:
            # Unknown domain - neutral risk
            return {
                'domain': domain,
                'risk_score': 0.5,
                'category': 'unknown',
                'in_database': False
            }
    except Exception as e:
        logger.error(f"Domain analysis error: {e}")
        return {
            'domain': 'unknown',
            'risk_score': 0.5,
            'category': 'unknown',
            'in_database': False
        }


def classify_text_with_server_models(text):
    """Classify text using server's loaded ONNX models"""
    try:
        # Check if models are loaded
        if text_session is None or tokenizer is None:
            logger.error("Text models not loaded")
            return {
                'classification': 'good',
                'confidence': 0.5,
                'reason': 'models_not_loaded',
                'text_length': len(text)
            }

        # Tokenize text - fix for numpy compatibility
        inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')

        # Convert to numpy arrays for ONNX (correct types: input_ids=int64, attention_mask=float32)
        onnx_inputs = {
            "input_ids": inputs['input_ids'].numpy().astype(np.int64),
            "attention_mask": inputs['attention_mask'].numpy().astype(np.float32)
        }

        outputs = text_session.run(None, onnx_inputs)
        safety_logits, reason_logits = outputs[0], outputs[1]

        # Get predictions
        safety_probs = softmax(safety_logits[0])
        reason_probs = softmax(reason_logits[0])

        safety_pred = int(np.argmax(safety_probs))
        reason_pred = int(np.argmax(reason_probs))

        classification = 'good' if safety_pred == 0 else 'bad'
        confidence = float(safety_probs[safety_pred])
        reason = reason_names.get(reason_pred, 'unknown')

        return {
            'classification': classification,
            'confidence': confidence,
            'reason': reason,
            'text_length': len(text)
        }
    except Exception as e:
        logger.error(f"Text classification error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'classification': 'good',
            'confidence': 0.5,
            'reason': 'classification_error',
            'text_length': len(text)
        }


def get_element_info_server(element):
    """Extract identifying information from HTML element for server functions"""
    info = {
        'tag': element.name,
        'id': element.get('id', ''),
        'class': ' '.join(element.get('class', [])),
        'data_attributes': {}
    }

    # Extract data attributes
    for attr_name, attr_value in element.attrs.items():
        if attr_name.startswith('data-'):
            info['data_attributes'][attr_name] = attr_value

    # Build CSS selector for easy targeting
    selector_parts = [element.name]
    if info['id']:
        selector_parts.append(f"#{info['id']}")
    if info['class']:
        selector_parts.append(f".{info['class'].replace(' ', '.')}")

    info['css_selector'] = ''.join(selector_parts)

    return info

def analyze_images_with_server_models(html_content, base_url):
    """Analyze images using server's loaded ONNX image model with element tracking"""
    image_results = []
    good_images = 0
    bad_images = 0
    bad_image_elements = []

    try:
        from bs4 import BeautifulSoup
        import requests
        from PIL import Image
        from io import BytesIO
        import urllib.parse

        soup = BeautifulSoup(html_content, 'html.parser')
        img_tags = soup.find_all('img')

        for img_tag in img_tags:
            try:
                # Get image URL
                img_url = img_tag.get('src') or img_tag.get('data-src') or img_tag.get('data-lazy-src')
                if not img_url:
                    continue

                # Handle relative URLs
                if img_url.startswith('//'):
                    img_url = 'https:' + img_url
                elif img_url.startswith('/'):
                    img_url = urllib.parse.urljoin(base_url, img_url)
                elif not img_url.startswith(('http://', 'https://')):
                    img_url = urllib.parse.urljoin(base_url, img_url)

                # Skip SVG and other non-image files
                if img_url.lower().endswith(('.svg', '.ico', '.gif')):
                    continue

                # Download image
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                response = requests.get(img_url, headers=headers, timeout=5, verify=False)

                if response.status_code == 200:
                    # Load image
                    image = Image.open(BytesIO(response.content))

                    # Convert to RGB if needed
                    if image.mode != 'RGB':
                        image = image.convert('RGB')

                    # Apply transforms
                    image_tensor = image_transform(image).unsqueeze(0).numpy()

                    # Run ONNX inference
                    outputs = image_session.run(None, {"input": image_tensor})
                    logits = outputs[0][0]

                    # Apply softmax to get proper probabilities
                    exp_logits = np.exp(logits)
                    probabilities = exp_logits / np.sum(exp_logits)

                    # Get prediction with correct class mapping
                    predicted_class = int(np.argmax(probabilities))
                    confidence = float(np.max(probabilities))  # Now using probabilities 0-1

                    # Correct class mapping: 0=bad, 1=good
                    classification = 'bad' if predicted_class == 0 else 'good'

                    # Get element info for tracking
                    element_info = get_element_info_server(img_tag)

                    result = {
                        'image_url': img_url,
                        'classification': classification,
                        'confidence': confidence,
                        'image_size': len(response.content),
                        'content_type': response.headers.get('content-type', 'unknown'),
                        'element_info': element_info
                    }

                    image_results.append(result)

                    if classification == 'good':
                        good_images += 1
                    else:
                        bad_images += 1
                        # Track bad image elements for easy targeting
                        bad_image_elements.append({
                            'element_info': element_info,
                            'url': img_url,
                            'alt_text': img_tag.get('alt', ''),
                            'confidence': confidence,
                            'classification': classification
                        })

            except Exception as e:
                logger.error(f"Error processing image {img_url}: {e}")
                continue

    except Exception as e:
        logger.error(f"Image analysis error: {e}")

    return {
        'has_harmful_images': bad_images > 0,
        'images_analyzed': len(image_results),
        'total_images': len(image_results),
        'good_images': good_images,
        'bad_images': bad_images,
        'bad_image_elements': bad_image_elements,
        'image_results': image_results
    }


try:
    if HTML_CLASSIFIER_AVAILABLE and IMAGE_MODEL_AVAILABLE and TEXT_MODEL_AVAILABLE:
        print("Initializing HTML page classifier...")
        html_classifier = HTMLPageClassifier()
        logger.info("HTML page classifier loaded successfully")
        HTML_CLASSIFIER_READY = True
    else:
        html_classifier = None
        HTML_CLASSIFIER_READY = False
        logger.warning("HTML classifier not ready - missing dependencies")
except Exception as e:
    logger.error(f"Failed to initialize HTML classifier: {e}")
    html_classifier = None
    HTML_CLASSIFIER_READY = False

# Load domain reputation data for server-side analysis
domain_reputation = {}
try:
    with open("data/domain_reputation.json", "r") as f:
        domain_reputation = json.load(f)
    print(f"Loaded {len(domain_reputation)} domain reputation entries")
except Exception as e:
    logger.warning(f"Could not load domain reputation data: {e}")

# Initialize three-stage safety analyzer
three_stage_analyzer = None
THREE_STAGE_ANALYZER_READY = False

try:
    if THREE_STAGE_ANALYZER_AVAILABLE and IMAGE_MODEL_AVAILABLE and TEXT_MODEL_AVAILABLE:
        print("Server-side three-stage analysis ready with domain reputation")
        logger.info("Three-stage analyzer functionality available via server models")
        THREE_STAGE_ANALYZER_READY = True
    else:
        missing_deps = []
        if not THREE_STAGE_ANALYZER_AVAILABLE:
            missing_deps.append("THREE_STAGE_ANALYZER")
        if not IMAGE_MODEL_AVAILABLE:
            missing_deps.append("IMAGE_MODEL")
        if not TEXT_MODEL_AVAILABLE:
            missing_deps.append("TEXT_MODEL")
        logger.warning(f"Three-stage analyzer not ready - missing dependencies: {', '.join(missing_deps)}")
except Exception as e:
    logger.error(f"Failed to initialize three-stage analyzer: {e}")
    import traceback
    traceback.print_exc()

try:
    if PDF_CLASSIFIER_AVAILABLE and IMAGE_MODEL_AVAILABLE and TEXT_MODEL_AVAILABLE:
        print("Initializing PDF classifier...")
        pdf_classifier = PDFClassifier()
        logger.info("PDF classifier loaded successfully")
        PDF_CLASSIFIER_READY = True
    else:
        pdf_classifier = None
        PDF_CLASSIFIER_READY = False
        logger.warning("PDF classifier not ready - missing dependencies")
except Exception as e:
    logger.error(f"Failed to initialize PDF classifier: {e}")
    pdf_classifier = None
    PDF_CLASSIFIER_READY = False

if IMAGE_MODEL_AVAILABLE and TEXT_MODEL_AVAILABLE:
    print("All basic models loaded successfully!")
else:
    print(f"Model loading status - Image: {IMAGE_MODEL_AVAILABLE}, Text: {TEXT_MODEL_AVAILABLE}")

if HTML_CLASSIFIER_READY:
    print("HTML page classifier ready!")
else:
    print("HTML page classifier not available")

if PDF_CLASSIFIER_READY:
    print("PDF classifier ready!")
else:
    print("PDF classifier not available")

if THREE_STAGE_ANALYZER_READY:
    print("Three-stage safety analyzer ready!")
else:
    print("Three-stage safety analyzer not available")


@app.route("/health", methods=["GET"])
def health():
    """Enhanced health check with model status"""
    status = {
        "status": "ok" if (IMAGE_MODEL_AVAILABLE or TEXT_MODEL_AVAILABLE) else "degraded",
        "models": {
            "image": "available" if IMAGE_MODEL_AVAILABLE else "unavailable",
            "text": "available" if TEXT_MODEL_AVAILABLE else "unavailable",
            "html": "available" if HTML_CLASSIFIER_READY else "unavailable",
            "pdf": "available" if PDF_CLASSIFIER_READY else "unavailable"
        },
        "timestamp": time.time()
    }
    logger.info(f"Health check requested: {status}")
    return jsonify(status)


@app.route("/classify", methods=["POST"])
def classify_image():
    """Classify image - returns good/bad with enhanced validation"""
    start_time = time.time()
    client_ip = request.remote_addr

    if not IMAGE_MODEL_AVAILABLE:
        logger.error("Image classification requested but model not available")
        return jsonify({"error": "Image classification service unavailable"}), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        image_input = data.get("image")

        if not image_input:
            return jsonify({"error": "No 'image' field provided"}), 400

        # Auto-detect if input is URL or base64
        if isinstance(image_input, str) and image_input.startswith(('http://', 'https://')):
            # Process as URL
            try:
                # Skip unsupported formats
                if image_input.lower().endswith(('.svg', '.ico')):
                    return jsonify({"error": "SVG and ICO formats are not supported"}), 400

                # Download image
                import requests
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                response = requests.get(image_input, headers=headers, timeout=10, verify=False)

                if response.status_code != 200:
                    return jsonify({"error": f"Failed to download image: HTTP {response.status_code}"}), 400

                image = Image.open(io.BytesIO(response.content)).convert("RGB")
                logger.info(f"Successfully downloaded image from URL: {image_input}")

            except Exception as e:
                logger.error(f"Error downloading image from URL {image_input}: {e}")
                return jsonify({"error": f"Failed to download or process image from URL: {str(e)}"}), 400

        else:
            # Process as base64 data
            # Validate input
            is_valid, error_msg = validate_image_data(image_input)
            if not is_valid:
                logger.warning(f"Invalid image data from {client_ip}: {error_msg}")
                return jsonify({"error": error_msg}), 400

            # Process base64 image
            if isinstance(image_input, str) and image_input.startswith("data:image"):
                image_input = image_input.split(",")[1]
            image_bytes = base64.b64decode(image_input)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        image_tensor = image_transform(image)
        image_np = image_tensor.unsqueeze(0).numpy()

        outputs = image_session.run(None, {"input": image_np})
        probs = softmax(outputs[0][0])
        predicted = np.argmax(probs)
        confidence = float(probs[predicted])

        label = "bad" if predicted == 0 else "good"

        # Log classification result
        processing_time = time.time() - start_time
        logger.info(f"Image classified - IP: {client_ip}, Result: {label}, Confidence: {confidence:.3f}, Time: {processing_time:.3f}s")

        # Add low confidence warning
        result = {
            "classification": label,
            "confidence": confidence,
            "type": "image",
            "processing_time": processing_time
        }

        if confidence < CONFIDENCE_THRESHOLD:
            result["warning"] = "Low confidence classification - manual review recommended"
            logger.warning(f"Low confidence image classification: {confidence:.3f} for IP: {client_ip}")

        return jsonify(result)

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Image classification error from {client_ip}: {str(e)}, Time: {processing_time:.3f}s")
        return jsonify({"error": "Classification failed - please try again"}), 500


@app.route("/classify_text", methods=["POST"])
def classify_text():
    """Classify text - returns good/bad with enhanced validation"""
    start_time = time.time()
    client_ip = request.remote_addr

    if not TEXT_MODEL_AVAILABLE:
        logger.error("Text classification requested but model not available")
        return jsonify({"error": "Text classification service unavailable"}), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        text = data.get("text", "")

        # Validate input
        is_valid, error_msg = validate_text_data(text)
        if not is_valid:
            logger.warning(f"Invalid text data from {client_ip}: {error_msg}")
            return jsonify({"error": error_msg}), 400

        # Process text with ONNX
        inputs = tokenizer(text, return_tensors="np", truncation=True, max_length=256, padding="max_length")

        # Run ONNX inference - token IDs should be int64, attention mask can be float32
        onnx_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.float32)
        }

        class_logits, reason_logits = text_session.run(None, onnx_inputs)

        # Process outputs
        class_probs = softmax(class_logits[0])
        reason_probs = softmax(reason_logits[0])
        class_predicted = np.argmax(class_probs)
        reason_predicted = np.argmax(reason_probs)
        confidence = float(class_probs[class_predicted])
        predicted_reason = reason_names.get(reason_predicted, "unknown")

        label = "bad" if class_predicted == 0 else "good"

        # Log classification result
        processing_time = time.time() - start_time
        text_preview = text[:50] + "..." if len(text) > 50 else text
        logger.info(f"Text classified - IP: {client_ip}, Result: {label}, Confidence: {confidence:.3f}, Time: {processing_time:.3f}s, Text: '{text_preview}'")

        # Add low confidence warning
        result = {
            "classification": label,
            "confidence": confidence,
            "type": "text",
            "processing_time": processing_time,
            "text_length": len(text)
        }

        # Add reason if available
        if TEXT_MODEL_HAS_REASONS and predicted_reason:
            result["reason"] = predicted_reason
            # Convert to human readable format
            result["reason_description"] = predicted_reason.replace("_", " ").title()

        if confidence < CONFIDENCE_THRESHOLD:
            result["warning"] = "Low confidence classification - manual review recommended"
            logger.warning(f"Low confidence text classification: {confidence:.3f} for IP: {client_ip}")

        return jsonify(result)

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Text classification error from {client_ip}: {str(e)}, Time: {processing_time:.3f}s")
        return jsonify({"error": "Classification failed - please try again"}), 500


@app.route("/classify_both", methods=["POST"])
def classify_both():
    """Classify both image and text - returns combined result"""
    data = request.get_json()
    image_data = data.get("image")
    text = data.get("text", "")
    
    results = {}
    
    # Classify image if provided
    if image_data:
        try:
            if isinstance(image_data, str) and image_data.startswith("data:image"):
                image_data = image_data.split(",")[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            image_tensor = image_transform(image)
            image_np = image_tensor.unsqueeze(0).numpy()
            
            outputs = image_session.run(None, {"input": image_np})
            probs = softmax(outputs[0][0])
            predicted = np.argmax(probs)
            
            results["image"] = {
                "classification": "bad" if predicted == 0 else "good",
                "confidence": float(probs[predicted])
            }
        except Exception as e:
            results["image"] = {"error": str(e)}
    
    # Classify text if provided
    if text:
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            
            with torch.no_grad():
                outputs = text_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
                predicted = np.argmax(probs)
            
            results["text"] = {
                "classification": "bad" if predicted == 0 else "good",
                "confidence": float(probs[predicted])
            }
        except Exception as e:
            results["text"] = {"error": str(e)}
    
    # Combined result - if either is bad, mark as bad
    final_classification = "good"
    if "image" in results and "error" not in results.get("image", {}):
        if results["image"]["classification"] == "bad":
            final_classification = "bad"
    if "text" in results and "error" not in results.get("text", {}):
        if results["text"]["classification"] == "bad":
            final_classification = "bad"
    
    results["final"] = {
        "classification": final_classification,
        "has_image": "image" in results,
        "has_text": "text" in results
    }
    
    return jsonify(results)


@app.route("/classify_html", methods=["POST"])
def classify_html():
    """Classify HTML page content - analyzes both text and images"""
    start_time = time.time()
    client_ip = request.remote_addr

    if not HTML_CLASSIFIER_READY:
        logger.error("HTML classification requested but classifier not available")
        return jsonify({"error": "HTML classification service unavailable"}), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        html_content = data.get("html", "")
        base_url = data.get("base_url", "")

        # Validate input
        if not html_content or not isinstance(html_content, str):
            logger.warning(f"Invalid HTML data from {client_ip}")
            return jsonify({"error": "No HTML content provided"}), 400

        if len(html_content) > 1024 * 1024:  # 1MB limit
            logger.warning(f"HTML content too large from {client_ip}: {len(html_content)} bytes")
            return jsonify({"error": "HTML content too large (max 1MB)"}), 400

        # Classify HTML page
        results = html_classifier.classify_html_page(html_content, base_url)

        # Add processing metadata
        processing_time = time.time() - start_time
        results["processing_time"] = processing_time
        results["type"] = "html"

        # Log results
        logger.info(f"HTML classified - IP: {client_ip}, Result: {results['page_classification']}, "
                   f"Time: {processing_time:.3f}s, Text: {results['text_analysis']['classification']}, "
                   f"Bad images: {results['image_analysis']['bad_images']}")

        return jsonify(results)

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"HTML classification error from {client_ip}: {str(e)}, Time: {processing_time:.3f}s")
        return jsonify({"error": "HTML classification failed - please try again"}), 500


@app.route("/classify_url", methods=["POST"])
def classify_url():
    """Classify web page by URL - downloads and analyzes entire page"""
    start_time = time.time()
    client_ip = request.remote_addr

    # Force use of three-stage analyzer if available (prioritize over HTML classifier)
    if not THREE_STAGE_ANALYZER_READY and not HTML_CLASSIFIER_READY:
        logger.error("URL classification requested but no classifier available")
        return jsonify({"error": "URL classification service unavailable"}), 503

    logger.info(f"Using three-stage analyzer: {THREE_STAGE_ANALYZER_READY}, HTML classifier: {HTML_CLASSIFIER_READY}")

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        url = data.get("url", "")

        # Validate input
        if not url or not isinstance(url, str):
            logger.warning(f"Invalid URL data from {client_ip}")
            return jsonify({"error": "No URL provided"}), 400

        # Basic URL validation
        if not url.startswith(('http://', 'https://')):
            return jsonify({"error": "URL must start with http:// or https://"}), 400

        # STAGE 0: Check domain using trained classifier
        # If domain is clearly phishing (>90%), skip image/text analysis
        DOMAIN_THRESHOLD = 0.90
        
        if DOMAIN_CLASSIFIER_AVAILABLE:
            try:
                from urllib.parse import urlparse
                domain = urlparse(url).netloc.lower()
                if domain.startswith('www.'):
                    domain = domain[4:]
                
                domain_result = predict_domain_cnn(domain)
                
                logger.info(f"Domain classifier result for {domain}: phishing={domain_result['is_phishing']}, confidence={domain_result['confidence']:.2f}")
                
                # If domain is definitely phishing (>90% confidence), skip other classifiers
                if domain_result['is_phishing'] and domain_result['confidence'] >= DOMAIN_THRESHOLD:
                    processing_time = time.time() - start_time
                    results = {
                        "page_classification": "bad",
                        "page_confidence": domain_result['confidence'],
                        "page_reasons": [f"domain_detected_as_phishing"],
                        "domain_analysis": {
                            "domain": domain,
                            "risk_score": 1.0,
                            "category": "phishing",
                            "in_database": domain_result.get('reputation_match', False),
                            "domain_source": domain_result.get('prediction_source', 'char_cnn_model')
                        },
                        "text_analysis": {"classification": "good", "confidence": 0.5, "reason": "skipped"},
                        "image_analysis": {"has_harmful_images": False, "images_analyzed": 0},
                        "page_info": {},
                        "processing_time": processing_time,
                        "type": "url",
                        "requested_url": url,
                        "classifiers_skipped": True,
                        "skip_reason": "domain_high_confidence_phishing"
                    }
                    logger.info(f"URL {url} classified as bad - domain {domain} detected as phishing with {domain_result['confidence']*100:.0f}% confidence (skipped image/text classifiers)")
                    return jsonify(results)
                    
            except Exception as e:
                logger.warning(f"Domain classification failed: {e}")

        # Use improved server-side three-stage analysis if available
        if THREE_STAGE_ANALYZER_READY:
            # Fetch webpage content for analysis
            import requests
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=10, verify=False)

            if response.status_code == 200:
                # Extract text content for analysis
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                text_content = soup.get_text(strip=True)[:2000]  # Limit text length

                # Stage 1: Analyze domain reputation
                domain_analysis = analyze_domain_reputation(url)

                # Stage 2: Classify text content
                text_analysis = classify_text_with_server_models(text_content)

                # Stage 2.5: Analyze images
                image_analysis = analyze_images_with_server_models(response.text, url)

                # Stage 3: Apply decision logic with priorities
                final_classification = text_analysis['classification']
                final_confidence = text_analysis['confidence']
                final_reason = text_analysis['reason']

                # HIGHEST PRIORITY: Harmful images override everything
                if image_analysis['has_harmful_images']:
                    final_classification = 'bad'
                    final_confidence = 0.95  # Very high confidence
                    final_reason = 'harmful_images_detected'
                    logger.info(f"Harmful images detected: {image_analysis['bad_images']} bad images found")

                # MEDIUM PRIORITY: Legitimate domain overrides false positive text classification (but not harmful images)
                elif (domain_analysis['risk_score'] < 0.3 and
                      text_analysis['classification'] == 'bad' and
                      text_analysis['reason'] == 'contact_info_sharing'):
                    # Domain is legitimate (low risk score) but text flagged as contact sharing
                    # This is likely a false positive from website metadata
                    final_classification = 'good'
                    final_confidence = 0.75  # High confidence in legitimate domain override
                    final_reason = 'legitimate_domain_override'
                    logger.info(f"Domain override applied for {domain_analysis['domain']}: {text_analysis['reason']} -> {final_reason}")

                # Convert to format similar to HTML classifier
                results = {
                    "page_classification": final_classification,
                    "page_confidence": final_confidence,
                    "page_reasons": [f"Final reason: {final_reason}"],
                    "text_analysis": text_analysis,
                    "domain_analysis": domain_analysis,
                    "image_analysis": image_analysis,
                    "page_info": {
                        "title": soup.title.string if soup.title else "",
                        "meta_description": soup.find('meta', attrs={'name': 'description'})['content'] if soup.find('meta', attrs={'name': 'description'}) else "",
                        "total_images_found": len(soup.find_all('img'))
                    }
                }

            else:
                return jsonify({"error": f"Failed to fetch webpage: HTTP {response.status_code}"}), 400
        else:
            # Fallback to original HTML classifier
            results = html_classifier.classify_url(url)

        # Add processing metadata
        processing_time = time.time() - start_time
        results["processing_time"] = processing_time
        results["type"] = "url"
        results["requested_url"] = url

        # Log results
        if "page_classification" in results:
            logger.info(f"URL classified - IP: {client_ip}, URL: {url}, Result: {results['page_classification']}, "
                       f"Time: {processing_time:.3f}s")
        else:
            logger.error(f"URL classification failed - IP: {client_ip}, URL: {url}, Error: {results.get('error', 'Unknown')}")

        return jsonify(results)

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"URL classification error from {client_ip}: {str(e)}, Time: {processing_time:.3f}s")
        return jsonify({"error": "URL classification failed - please try again"}), 500


@app.route("/classify_pdf", methods=["POST"])
def classify_pdf():
    """Classify PDF file - analyzes both text content and images"""
    start_time = time.time()
    client_ip = request.remote_addr

    if not PDF_CLASSIFIER_READY:
        logger.error("PDF classification requested but classifier not available")
        return jsonify({"error": "PDF classification service unavailable"}), 503

    try:
        # Check if file was uploaded
        if 'file' in request.files:
            file = request.files['file']

            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400

            # Validate file type
            if not file.filename.lower().endswith('.pdf'):
                return jsonify({"error": "File must be a PDF"}), 400

            # Read file content
            pdf_bytes = file.read()

            # Size validation (max 10MB)
            if len(pdf_bytes) > 10 * 1024 * 1024:
                logger.warning(f"PDF file too large from {client_ip}: {len(pdf_bytes)} bytes")
                return jsonify({"error": "PDF file too large (max 10MB)"}), 400

            # Classify PDF
            results = pdf_classifier.classify_pdf_bytes(pdf_bytes, file.filename)

        else:
            # Handle base64 encoded PDF
            data = request.get_json()
            if not data:
                return jsonify({"error": "No file data provided"}), 400

            pdf_data = data.get("pdf_data", "")
            filename = data.get("filename", "document.pdf")

            if not pdf_data:
                return jsonify({"error": "No PDF data provided"}), 400

            try:
                # Decode base64 PDF data
                pdf_bytes = base64.b64decode(pdf_data)

                # Size validation (max 10MB)
                if len(pdf_bytes) > 10 * 1024 * 1024:
                    logger.warning(f"PDF data too large from {client_ip}: {len(pdf_bytes)} bytes")
                    return jsonify({"error": "PDF data too large (max 10MB)"}), 400

                # Classify PDF
                results = pdf_classifier.classify_pdf_bytes(pdf_bytes, filename)

            except Exception as e:
                return jsonify({"error": f"Invalid PDF data: {str(e)}"}), 400

        # Add processing metadata
        processing_time = time.time() - start_time
        results["processing_time"] = processing_time
        results["type"] = "pdf"

        # Log results
        logger.info(f"PDF classified - IP: {client_ip}, File: {results['file_info']['file_name']}, "
                   f"Result: {results['pdf_classification']}, Time: {processing_time:.3f}s, "
                   f"Text: {results['text_analysis']['classification']}, "
                   f"Bad images: {results['image_analysis']['bad_images']}")

        return jsonify(results)

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"PDF classification error from {client_ip}: {str(e)}, Time: {processing_time:.3f}s")
        return jsonify({"error": "PDF classification failed - please try again"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
