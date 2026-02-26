"""
HTML Page Classifier for Child Safety
Combines text content analysis and image analysis to classify entire web pages.
Uses existing trained text and image models.
"""

import os
import re
import json
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import base64
from io import BytesIO
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Tuple, Optional
import torch
import numpy as np
from PIL import Image
from bs4 import BeautifulSoup
from transformers import DistilBertTokenizer
import onnxruntime as ort
from torchvision import transforms
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HTMLPageClassifier:
    def __init__(self,
                 text_model_path="output/text_classifier.onnx",
                 image_model_path="output/image/image_classifier.onnx",
                 max_images=10,
                 image_timeout=5):
        """
        Initialize HTML page classifier with trained models.

        Args:
            text_model_path: Path to trained text classifier
            image_model_path: Path to trained image classifier (ONNX)
            max_images: Maximum number of images to analyze per page
            image_timeout: Timeout for downloading images (seconds)
        """
        self.max_images = max_images
        self.image_timeout = image_timeout

        # Load text classifier (ONNX)
        self.text_session, self.tokenizer, self.reason_categories = self._load_text_model(text_model_path)

        # Load image classifier
        self.image_session, self.image_transform = self._load_image_model(image_model_path)

        logger.info("HTML Page Classifier initialized successfully")

    def _load_text_model(self, model_path):
        """Load the trained text classifier with reasons (ONNX)"""
        try:
            # Load ONNX text model
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            text_session = ort.InferenceSession(model_path, sess_options, providers=["CPUExecutionProvider"])

            # Load tokenizer from the bert model directory
            tokenizer = DistilBertTokenizer.from_pretrained("output/bert_text_with_reasons")

            # Load reason categories
            with open("output/bert_text_with_reasons/reason_categories.json") as f:
                reason_categories = json.load(f)

            logger.info("Text classifier (ONNX) loaded successfully")
            return text_session, tokenizer, reason_categories

        except Exception as e:
            logger.error(f"Failed to load text classifier: {e}")
            return None, None, {}

    def _load_image_model(self, model_path):
        """Load the trained image classifier (ONNX)"""
        try:
            # Load ONNX model
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            session = ort.InferenceSession(model_path, sess_options, providers=["CPUExecutionProvider"])

            # Image preprocessing (128x128 to match training)
            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            logger.info("Image classifier loaded successfully")
            return session, transform

        except Exception as e:
            logger.error(f"Failed to load image classifier: {e}")
            return None, None

    def extract_content_from_html(self, html_content: str, base_url: str = "") -> Dict:
        """
        Extract text content and image URLs from HTML, tracking element info.

        Args:
            html_content: Raw HTML content
            base_url: Base URL for resolving relative links

        Returns:
            Dict containing extracted text, image URLs, and element tracking info
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style", "noscript"]):
                script.decompose()

            # Extract text content with element tracking
            text_elements = []
            for element in soup.find_all(text=True):
                parent = element.parent
                if parent and parent.name not in ['script', 'style', 'noscript']:
                    element_info = self._get_element_info(parent)
                    text_elements.append({
                        'text': element.strip(),
                        'element_info': element_info
                    })

            # Clean up text
            text_content = ' '.join([elem['text'] for elem in text_elements if elem['text']])
            text_content = re.sub(r'\s+', ' ', text_content).strip()

            # Extract image URLs with element info
            image_data = []
            for img in soup.find_all('img', src=True):
                img_url = img['src']

                # Convert relative URLs to absolute
                if base_url and not img_url.startswith(('http://', 'https://')):
                    img_url = urljoin(base_url, img_url)

                # Filter out common non-content images
                if not any(skip in img_url.lower() for skip in ['icon', 'logo', 'avatar', 'button', 'arrow']):
                    element_info = self._get_element_info(img)
                    image_data.append({
                        'url': img_url,
                        'element_info': element_info,
                        'alt_text': img.get('alt', '')
                    })

            # Extract other potentially relevant content
            title = soup.title.string if soup.title else ""
            meta_description = ""
            if soup.find('meta', attrs={'name': 'description'}):
                meta_description = soup.find('meta', attrs={'name': 'description'}).get('content', '')

            # Combine all text content
            full_text = f"{title} {meta_description} {text_content}".strip()

            return {
                'text_content': full_text,
                'text_elements': text_elements,
                'image_data': image_data[:self.max_images],  # Limit number of images
                'image_urls': [img['url'] for img in image_data[:self.max_images]],
                'title': title,
                'meta_description': meta_description,
                'num_images': len(image_data)
            }

        except Exception as e:
            logger.error(f"Failed to extract content from HTML: {e}")
            return {
                'text_content': "",
                'text_elements': [],
                'image_data': [],
                'image_urls': [],
                'title': "",
                'meta_description': "",
                'num_images': 0
            }

    def classify_text_content(self, text: str, text_elements: List[Dict] = None) -> Dict:
        """Classify text content using trained ONNX text model with element tracking"""
        if not self.text_session or not text.strip():
            return {'classification': 'good', 'confidence': 0.5, 'reason': 'unknown'}

        try:
            # Tokenize text - same as server.py
            inputs = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')

            # Convert to numpy arrays for ONNX (correct types: input_ids=int64, attention_mask=float32)
            onnx_inputs = {
                "input_ids": inputs['input_ids'].numpy().astype(np.int64),
                "attention_mask": inputs['attention_mask'].numpy().astype(np.float32)
            }

            # Run ONNX inference
            outputs = self.text_session.run(None, onnx_inputs)
            safety_logits, reason_logits = outputs[0], outputs[1]

            # Get predictions using softmax
            safety_probs = self._softmax(safety_logits[0])
            reason_probs = self._softmax(reason_logits[0])

            safety_pred = int(np.argmax(safety_probs))
            reason_pred = int(np.argmax(reason_probs))

            # Note: ONNX model has different label mapping (0=good, 1=bad)
            classification = 'good' if safety_pred == 0 else 'bad'
            confidence = float(safety_probs[safety_pred])

            # Get reason name
            reason_names = {v: k for k, v in self.reason_categories.items()}
            predicted_reason = reason_names.get(reason_pred, "unknown")

            result = {
                'classification': classification,
                'confidence': confidence,
                'reason': predicted_reason,
                'reason_description': predicted_reason.replace("_", " ").title()
            }

            # If text is bad and we have element info, try to identify problematic elements
            if classification == 'bad' and text_elements and confidence > 0.6:
                bad_text_elements = self._identify_bad_text_elements(text_elements, predicted_reason)
                if bad_text_elements:
                    result['bad_text_elements'] = bad_text_elements

            return result

        except Exception as e:
            logger.error(f"Text classification failed: {e}")
            import traceback
            traceback.print_exc()
            return {'classification': 'good', 'confidence': 0.5, 'reason': 'error'}

    def download_and_classify_image(self, image_url: str, element_info: Dict = None) -> Dict:
        """Download and classify a single image with element tracking"""
        if not self.image_session:
            return {'classification': 'good', 'confidence': 0.5, 'error': 'Image model not loaded'}

        try:
            # Download image with timeout
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(image_url, headers=headers, timeout=self.image_timeout, stream=True, verify=False)
            response.raise_for_status()

            # Check if content is actually an image
            content_type = response.headers.get('content-type', '').lower()
            if not content_type.startswith('image/'):
                return {'classification': 'good', 'confidence': 0.5, 'error': 'Not an image'}

            # Open and process image
            image = Image.open(BytesIO(response.content)).convert("RGB")
            image_tensor = self.image_transform(image).unsqueeze(0).numpy()

            # Run inference
            outputs = self.image_session.run(None, {"input": image_tensor})
            probs = self._softmax(outputs[0][0])
            predicted = np.argmax(probs)
            confidence = float(probs[predicted])

            label = "bad" if predicted == 0 else "good"

            result = {
                'classification': label,
                'confidence': confidence,
                'url': image_url,
                'size': len(response.content)
            }

            # Include element info if provided
            if element_info:
                result['element_info'] = element_info

            return result

        except Exception as e:
            logger.warning(f"Failed to classify image {image_url}: {e}")
            result = {'classification': 'good', 'confidence': 0.5, 'error': str(e)}
            if element_info:
                result['element_info'] = element_info
            return result

    def _get_element_info(self, element) -> Dict:
        """Extract identifying information from HTML element"""
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

    def _identify_bad_text_elements(self, text_elements: List[Dict], predicted_reason: str) -> List[Dict]:
        """Try to identify which text elements likely contain problematic content"""
        bad_elements = []

        # Keywords associated with different reason categories
        reason_keywords = {
            'bullying_harassment': ['bully', 'harass', 'stupid', 'loser', 'hate', 'kill yourself'],
            'threats_violence': ['kill', 'hurt', 'violence', 'threat', 'weapon', 'attack'],
            'dangerous_weapons': ['gun', 'knife', 'weapon', 'bomb', 'explosive'],
            'inappropriate_behavior': ['inappropriate', 'sexual', 'nude', 'naked'],
            'secrecy_manipulation': ['secret', 'don\'t tell', 'keep quiet', 'between us'],
            'personal_info_request': ['address', 'phone', 'password', 'social security'],
            'contact_info_sharing': ['email', 'phone', 'meet me', 'address'],
            'predatory_behavior': ['meet', 'alone', 'secret', 'special friend'],
            'inappropriate_content': ['adult', 'mature', 'explicit', 'nsfw']
        }

        keywords = reason_keywords.get(predicted_reason, [])

        for element in text_elements:
            text = element['text'].lower()
            if text and any(keyword in text for keyword in keywords):
                # Score based on keyword matches
                matches = [kw for kw in keywords if kw in text]
                bad_elements.append({
                    'element_info': element['element_info'],
                    'text_snippet': element['text'][:100] + '...' if len(element['text']) > 100 else element['text'],
                    'matched_keywords': matches,
                    'reason': predicted_reason
                })

        return bad_elements

    def _softmax(self, x):
        """Apply softmax to numpy array"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def classify_html_page(self, html_content: str, base_url: str = "") -> Dict:
        """
        Classify entire HTML page by analyzing text content and images.

        Args:
            html_content: Raw HTML content
            base_url: Base URL for resolving relative links

        Returns:
            Classification results with detailed analysis
        """
        logger.info(f"Analyzing HTML page: {base_url}")

        # Extract content
        extracted = self.extract_content_from_html(html_content, base_url)

        # Classify text content with element tracking
        text_result = self.classify_text_content(extracted['text_content'], extracted['text_elements'])

        # Classify images with element tracking
        image_results = []
        bad_images = 0
        bad_image_elements = []

        for img_data in extracted['image_data']:
            img_result = self.download_and_classify_image(img_data['url'], img_data['element_info'])
            image_results.append(img_result)

            if img_result['classification'] == 'bad':
                bad_images += 1
                bad_image_elements.append({
                    'element_info': img_data['element_info'],
                    'url': img_data['url'],
                    'alt_text': img_data.get('alt_text', ''),
                    'confidence': img_result['confidence']
                })

        # Combine results to determine overall page classification
        page_classification = self._determine_page_classification(text_result, image_results, bad_images)

        # Compile comprehensive results
        results = {
            'page_classification': page_classification['classification'],
            'page_confidence': page_classification['confidence'],
            'page_reasons': page_classification['reasons'],

            'text_analysis': {
                'classification': text_result['classification'],
                'confidence': text_result['confidence'],
                'reason': text_result.get('reason', 'unknown'),
                'reason_description': text_result.get('reason_description', 'Unknown'),
                'content_length': len(extracted['text_content'])
            },

            'image_analysis': {
                'total_images': len(image_results),
                'bad_images': bad_images,
                'good_images': len(image_results) - bad_images,
                'bad_image_elements': bad_image_elements,
                'image_results': image_results
            },

            'page_info': {
                'title': extracted['title'],
                'meta_description': extracted['meta_description'],
                'total_images_found': extracted['num_images'],
                'images_analyzed': len(image_results)
            }
        }

        logger.info(f"Page analysis complete: {page_classification['classification']} "
                   f"(Text: {text_result['classification']}, Bad images: {bad_images})")

        return results

    def _determine_page_classification(self, text_result: Dict, image_results: List[Dict], bad_images: int) -> Dict:
        """Determine overall page classification based on text and image analysis"""

        reasons = []

        # If text content is bad, page is bad
        if text_result['classification'] == 'bad':
            reasons.append(f"Bad text content: {text_result.get('reason_description', 'Unknown reason')}")
            return {
                'classification': 'bad',
                'confidence': max(text_result['confidence'], 0.8),
                'reasons': reasons
            }

        # If any images are bad, page is bad
        if bad_images > 0:
            reasons.append(f"Contains {bad_images} inappropriate image{'s' if bad_images > 1 else ''}")

            # Calculate confidence based on number of bad images and their confidence
            bad_image_confidences = [img['confidence'] for img in image_results
                                   if img['classification'] == 'bad']
            avg_bad_confidence = np.mean(bad_image_confidences) if bad_image_confidences else 0.7

            return {
                'classification': 'bad',
                'confidence': min(avg_bad_confidence + (bad_images * 0.1), 0.95),
                'reasons': reasons
            }

        # If text is good and no bad images, page is good
        reasons.append("Clean text content and safe images")

        # Calculate overall confidence
        all_confidences = [text_result['confidence']]
        all_confidences.extend([img['confidence'] for img in image_results
                              if img['classification'] == 'good'])

        overall_confidence = np.mean(all_confidences) if all_confidences else 0.5

        return {
            'classification': 'good',
            'confidence': overall_confidence,
            'reasons': reasons
        }

    def classify_url(self, url: str) -> Dict:
        """Download and classify HTML page from URL"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=10, verify=False)
            response.raise_for_status()

            return self.classify_html_page(response.text, url)

        except Exception as e:
            logger.error(f"Failed to fetch URL {url}: {e}")
            return {
                'page_classification': 'error',
                'error': str(e),
                'url': url
            }


def main():
    """Example usage of HTMLPageClassifier"""

    # Initialize classifier
    classifier = HTMLPageClassifier()

    # Example HTML content
    example_html = """
    <html>
    <head>
        <title>Example Page</title>
        <meta name="description" content="This is a test page">
    </head>
    <body>
        <h1>Welcome to our site!</h1>
        <p>This is some sample content. Contact me at test@example.com for more info!</p>
        <img src="https://example.com/image1.jpg" alt="Sample image">
        <img src="https://example.com/image2.jpg" alt="Another image">
    </body>
    </html>
    """

    # Classify HTML content
    results = classifier.classify_html_page(example_html, "https://example.com")

    # Print results
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()