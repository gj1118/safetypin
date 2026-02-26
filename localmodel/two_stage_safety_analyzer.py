"""
Three-Stage Safety Analyzer
Stage 1: Text classification (proven effective model)
Stage 2: Domain analysis (conditional, only when URLs present)
Stage 3: Image analysis (conditional, only when images present)
Combines results intelligently for final safety decision

Image analysis has HIGHEST PRIORITY - harmful images override all other analysis.
This ensures cases like Wikipedia koala page are correctly flagged as BAD.
"""

import re
import json
import torch
import requests
import numpy as np
import onnxruntime as ort
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse
from transformers import DistilBertTokenizer, DistilBertModel
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import logging
from io import BytesIO
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


class DistilBertWithReasons(nn.Module):
    """Simple, effective text classifier (without domain features)"""
    def __init__(self, num_labels=2, num_reasons=11):
        super().__init__()
        self.num_labels = num_labels
        self.num_reasons = num_reasons

        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.6)

        # Simple, proven architecture
        self.classifier = nn.Sequential(
            nn.Linear(self.distilbert.config.hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_labels)
        )

        self.reason_classifier = nn.Sequential(
            nn.Linear(self.distilbert.config.hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_reasons)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
        pooled_output = self.dropout(pooled_output)

        classification_logits = self.classifier(pooled_output)
        reason_logits = self.reason_classifier(pooled_output)

        return classification_logits, reason_logits


class ImageAnalyzer:
    """Image classification analyzer using ONNX model"""

    def __init__(self, image_model_path="output/image/image_classifier.onnx"):
        try:
            # Load ONNX image model
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.image_session = ort.InferenceSession(image_model_path, sess_options)

            # Image preprocessing
            self.image_transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            print("ONNX image classifier loaded successfully")
            self.image_classifier_available = True

        except Exception as e:
            print(f"Error loading image classifier: {e}")
            self.image_classifier_available = False

    def softmax(self, x):
        """Apply softmax to logits"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def classify_image_from_url(self, image_url: str, timeout: int = 5) -> Dict[str, Any]:
        """Download and classify image from URL"""
        if not self.image_classifier_available:
            return {'classification': 'good', 'confidence': 0.5, 'error': 'Image model not available'}

        try:
            # Download image
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(image_url, headers=headers, timeout=timeout, stream=True, verify=False)
            response.raise_for_status()

            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if not content_type.startswith('image/'):
                return {'classification': 'good', 'confidence': 0.5, 'error': 'Not an image'}

            # Process image
            image = Image.open(BytesIO(response.content)).convert("RGB")
            image_tensor = self.image_transform(image)
            image_np = image_tensor.unsqueeze(0).numpy()

            # Run inference
            outputs = self.image_session.run(None, {"input": image_np})
            probs = self.softmax(outputs[0][0])
            predicted = np.argmax(probs)
            confidence = float(probs[predicted])

            classification = "bad" if predicted == 0 else "good"

            return {
                'classification': classification,
                'confidence': confidence,
                'image_url': image_url,
                'content_type': content_type,
                'image_size': len(response.content)
            }

        except Exception as e:
            return {
                'classification': 'good',
                'confidence': 0.5,
                'error': f'Image classification failed: {str(e)}',
                'image_url': image_url
            }

    def extract_images_from_html(self, html_content: str, base_url: str) -> List[str]:
        """Extract image URLs from HTML content"""
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin

        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            images = soup.find_all('img')

            image_urls = []
            for img in images:
                src = img.get('src') or img.get('data-src')
                if src:
                    # Convert relative URLs to absolute
                    absolute_url = urljoin(base_url, src)
                    image_urls.append(absolute_url)

            return image_urls[:10]  # Limit to 10 images

        except Exception as e:
            logger.error(f"Error extracting images from HTML: {e}")
            return []

    def analyze_page_images(self, html_content: str, base_url: str, max_images: int = 10) -> Dict[str, Any]:
        """Analyze all images found on a webpage"""
        if not self.image_classifier_available:
            return {
                'total_images': 0,
                'images_analyzed': 0,
                'bad_images': 0,
                'good_images': 0,
                'image_results': [],
                'error': 'Image classifier not available'
            }

        # Extract image URLs
        image_urls = self.extract_images_from_html(html_content, base_url)

        if not image_urls:
            return {
                'total_images': 0,
                'images_analyzed': 0,
                'bad_images': 0,
                'good_images': 0,
                'image_results': []
            }

        # Analyze each image
        image_results = []
        bad_count = 0
        good_count = 0

        for i, url in enumerate(image_urls[:max_images]):
            result = self.classify_image_from_url(url)
            image_results.append(result)

            if result.get('classification') == 'bad':
                bad_count += 1
            elif result.get('classification') == 'good':
                good_count += 1

        return {
            'total_images': len(image_urls),
            'images_analyzed': len(image_results),
            'bad_images': bad_count,
            'good_images': good_count,
            'image_results': image_results,
            'has_harmful_images': bad_count > 0
        }


class DomainAnalyzer:
    """Domain reputation and behavioral pattern analyzer"""

    def __init__(self, domain_reputation_file="data/domain_reputation.json"):
        # Load domain reputation
        try:
            with open(domain_reputation_file, 'r') as f:
                self.domain_reputation = json.load(f)
            print(f"Loaded {len(self.domain_reputation)} domain reputations")
        except FileNotFoundError:
            print(f"Warning: {domain_reputation_file} not found, domain analysis disabled")
            self.domain_reputation = {}

        # Platform classifications
        self.social_platforms = [
            'facebook.com', 'instagram.com', 'twitter.com', 'tiktok.com',
            'snapchat.com', 'discord.com', 'reddit.com', 'youtube.com', 'twitch.tv'
        ]
        self.news_media = ['bbc.com', 'cnn.com', 'reuters.com', 'ap.org', 'npr.org', 'theguardian.com']
        self.educational = ['wikipedia.org', 'wikimedia.org']

        # Behavioral risk patterns
        self.risk_patterns = {
            'personal_info_request': [
                r"what'?s your (?:phone|number|address)",
                r"give me your (?:phone|number|contact)",
                r"send me your (?:photo|picture|pic)",
                r"where do you live"
            ],
            'secrecy_requests': [
                r"don'?t tell (?:anyone|your parents)",
                r"(?:keep|this is) (?:our|a) secret",
                r"between (?:you and me|us)"
            ],
            'meeting_escalation': [
                r"want to meet (?:up|in person)",
                r"let'?s meet at",
                r"come to my (?:place|house)"
            ]
        }

    def extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text"""
        url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            r'|(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:/[^\s]*)?'
        )
        return [url.strip() for url in url_pattern.findall(text) if len(url) > 5]

    def extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url if url.startswith(('http://', 'https://')) else f'http://{url}')
            return parsed.netloc.lower()
        except:
            return url.lower()

    def classify_platform_type(self, domain: str) -> str:
        """Classify platform type"""
        for platform in self.social_platforms:
            if platform in domain:
                return "social_platform"
        for news in self.news_media:
            if news in domain:
                return "news_media"
        for edu in self.educational:
            if edu in domain:
                return "educational"
        if domain.endswith('.edu') or domain.endswith('.gov'):
            return "institutional"
        return "unknown"

    def get_domain_risk_score(self, domain: str) -> float:
        """Get risk score for domain"""
        reputation = self.domain_reputation.get(domain, 'unknown')
        if reputation == 'phishing':
            return 0.9
        elif reputation == 'legitimate':
            return 0.1
        else:
            return 0.5  # Unknown

    def detect_behavioral_patterns(self, text: str) -> Dict[str, Any]:
        """Detect predatory behavioral patterns"""
        text_lower = text.lower()
        detected = {}
        total_risk = 0.0

        for pattern_type, patterns in self.risk_patterns.items():
            matches = 0
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    matches += 1

            detected[pattern_type] = matches > 0
            if matches > 0:
                # Weight different patterns
                if pattern_type == 'meeting_escalation':
                    total_risk += 0.4
                elif pattern_type == 'personal_info_request':
                    total_risk += 0.3
                elif pattern_type == 'secrecy_requests':
                    total_risk += 0.3

        return {
            'patterns_detected': detected,
            'behavioral_risk_score': min(total_risk, 1.0),
            'high_risk_behavior': total_risk > 0.3
        }

    def analyze_domains(self, text: str) -> Dict[str, Any]:
        """Analyze domain context in text"""
        urls = self.extract_urls(text)

        if not urls:
            return {
                'has_urls': False,
                'domain_risk_score': 0.0,
                'platform_type': 'none',
                'urls_analyzed': 0
            }

        domain_risks = []
        platform_types = []

        for url in urls:
            domain = self.extract_domain(url)
            risk_score = self.get_domain_risk_score(domain)
            platform_type = self.classify_platform_type(domain)

            domain_risks.append(risk_score)
            platform_types.append(platform_type)

        avg_domain_risk = np.mean(domain_risks)
        primary_platform = max(set(platform_types), key=platform_types.count) if platform_types else 'unknown'

        return {
            'has_urls': True,
            'domain_risk_score': avg_domain_risk,
            'platform_type': primary_platform,
            'urls_analyzed': len(urls),
            'domains_found': [self.extract_domain(url) for url in urls]
        }


class ThreeStageSafetyAnalyzer:
    """Main three-stage safety analyzer"""

    def __init__(self, text_model_path="output/bert_text_with_reasons", image_model_path="output/image/image_classifier.onnx"):
        # Stage 1: Load text classifier
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            # Load tokenizer
            self.tokenizer = DistilBertTokenizer.from_pretrained(text_model_path)

            # Load model
            self.text_model = DistilBertWithReasons(num_labels=2, num_reasons=11)
            self.text_model.load_state_dict(torch.load(f"{text_model_path}/model.pth", map_location=self.device))
            self.text_model.to(self.device)
            self.text_model.eval()

            # Load reason categories
            with open(f"{text_model_path}/reason_categories.json", 'r') as f:
                reason_categories = json.load(f)
            self.reason_names = {v: k for k, v in reason_categories.items()}

            print("Text classifier loaded successfully")
            self.text_classifier_available = True

        except Exception as e:
            print(f"Error loading text classifier: {e}")
            self.text_classifier_available = False

        # Stage 2: Initialize domain analyzer
        self.domain_analyzer = DomainAnalyzer()

        # Stage 3: Initialize image analyzer
        self.image_analyzer = ImageAnalyzer(image_model_path)

    def classify_text(self, text: str) -> Dict[str, Any]:
        """Stage 1: Text classification"""
        if not self.text_classifier_available:
            return {'error': 'Text classifier not available'}

        try:
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                  max_length=256, padding="max_length")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Predict
            with torch.no_grad():
                class_logits, reason_logits = self.text_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                )

                # Get probabilities
                class_probs = torch.softmax(class_logits, dim=1).cpu().numpy()[0]
                reason_probs = torch.softmax(reason_logits, dim=1).cpu().numpy()[0]

                class_predicted = np.argmax(class_probs)
                reason_predicted = np.argmax(reason_probs)

                classification = "bad" if class_predicted == 0 else "good"
                confidence = float(class_probs[class_predicted])
                reason = self.reason_names.get(reason_predicted, "unknown")

                return {
                    'classification': classification,
                    'confidence': confidence,
                    'reason': reason,
                    'text_length': len(text)
                }

        except Exception as e:
            return {'error': f'Text classification failed: {str(e)}'}

    def analyze_comprehensive(self, text: str, url: Optional[str] = None, html_content: Optional[str] = None) -> Dict[str, Any]:
        """Complete three-stage analysis"""

        # Stage 1: Text Classification
        text_result = self.classify_text(text)

        # Stage 2: Domain Analysis (conditional)
        domain_result = self.domain_analyzer.analyze_domains(text if not url else f"{text} {url}")
        behavioral_result = self.domain_analyzer.detect_behavioral_patterns(text)

        # Stage 3: Image Analysis (conditional)
        image_result = None
        if url and html_content:
            # Analyze images from webpage HTML content
            image_result = self.image_analyzer.analyze_page_images(html_content, url)
        elif url and not html_content:
            # Fetch webpage content if URL provided without HTML
            try:
                import requests
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                response = requests.get(url, headers=headers, timeout=10, verify=False)
                if response.status_code == 200:
                    image_result = self.image_analyzer.analyze_page_images(response.text, url)
            except Exception as e:
                logger.error(f"Failed to fetch webpage for image analysis: {e}")

        # Combine results intelligently
        final_result = self._combine_results(text_result, domain_result, behavioral_result, image_result)

        analysis_breakdown = {
            'text_analysis': text_result,
            'domain_analysis': domain_result,
            'behavioral_analysis': behavioral_result
        }

        if image_result:
            analysis_breakdown['image_analysis'] = image_result

        return {
            'final_classification': final_result['classification'],
            'final_confidence': final_result['confidence'],
            'final_reason': final_result['reason'],
            'analysis_breakdown': analysis_breakdown,
            'decision_factors': final_result['factors']
        }

    def _combine_results(self, text_result: Dict, domain_result: Dict, behavioral_result: Dict, image_result: Optional[Dict] = None) -> Dict:
        """Intelligently combine text, domain, and image analysis"""

        # Start with text classification
        base_classification = text_result.get('classification', 'good')
        base_confidence = text_result.get('confidence', 0.5)
        base_reason = text_result.get('reason', 'unknown')

        decision_factors = ['text_analysis']

        # Stage 3: Image Analysis (HIGHEST PRIORITY - overrides everything)
        if image_result and image_result.get('has_harmful_images', False):
            base_classification = 'bad'
            base_confidence = 0.95  # Very high confidence for harmful images
            base_reason = 'harmful_images'
            decision_factors.append('image_analysis')

            # If harmful images found, don't need to check other factors
            return {
                'classification': base_classification,
                'confidence': base_confidence,
                'reason': base_reason,
                'factors': decision_factors
            }

        # Stage 2: Domain Analysis (conditional)
        if domain_result.get('has_urls', False):
            decision_factors.append('domain_analysis')

            # Social platform + personal info request = high risk
            if (domain_result.get('platform_type') == 'social_platform' and
                behavioral_result.get('high_risk_behavior', False)):

                base_classification = 'bad'
                base_confidence = max(base_confidence, 0.8)
                base_reason = 'predatory_behavior'
                decision_factors.append('social_platform_risk')

            # Phishing domain = high risk regardless of content
            elif domain_result.get('domain_risk_score', 0) > 0.8:
                base_classification = 'bad'
                base_confidence = max(base_confidence, 0.9)
                base_reason = 'phishing_domain'
                decision_factors.append('phishing_domain')

            # Educational/news domain + good text = lower risk
            elif (domain_result.get('platform_type') in ['educational', 'news_media'] and
                  base_classification == 'good'):
                base_confidence = min(base_confidence + 0.1, 1.0)  # Boost confidence
                decision_factors.append('trusted_domain')

            # CRITICAL: Legitimate domain overrides false positive text classification
            elif (domain_result.get('domain_risk_score', 0.5) < 0.3 and
                  base_classification == 'bad' and
                  text_result.get('reason') == 'contact_info_sharing'):
                # Domain is legitimate (low risk score) but text flagged as contact sharing
                # This is likely a false positive from website metadata
                base_classification = 'good'
                base_confidence = 0.75  # High confidence in legitimate domain override
                base_reason = 'legitimate_domain_override'
                decision_factors.append('legitimate_domain_override')

        # Apply behavioral analysis
        if behavioral_result.get('high_risk_behavior', False):
            decision_factors.append('behavioral_patterns')
            if base_classification == 'good':
                # Behavioral patterns override text classification
                base_classification = 'bad'
                base_confidence = 0.85
                base_reason = 'predatory_behavior'

        # Add image analysis to factors if it was performed
        if image_result is not None:
            decision_factors.append('image_analysis')

        return {
            'classification': base_classification,
            'confidence': base_confidence,
            'reason': base_reason,
            'factors': decision_factors
        }


def main():
    """Test the three-stage analyzer"""
    analyzer = ThreeStageSafetyAnalyzer()

    test_cases = [
        # Text only - should use text classifier
        ("This is educational content about staying safe online.", None),

        # Social platform + personal info request - should flag as bad
        ("Hey, what's your phone number? I'm on Instagram @user123", None),

        # News content - should be good
        ("According to BBC News, the economy is improving this quarter.", None),

        # Behavioral pattern - should flag as bad
        ("Don't tell your parents, but I have a special gift for you.", None),

        # URL with phishing domain (if in dataset)
        ("Click here to login: http://suspicious-site.tk/login", None),

        # Wikipedia koala page - should be BAD due to koala images
        ("Educational content about koalas", "https://en.wikipedia.org/wiki/Koala"),
    ]

    print("=== Three-Stage Safety Analysis Test ===\n")

    for i, (text, url) in enumerate(test_cases, 1):
        print(f"Test {i}: {text[:50]}{'...' if len(text) > 50 else ''}")
        if url:
            print(f"  URL: {url}")

        result = analyzer.analyze_comprehensive(text, url)

        print(f"  Final: {result['final_classification']} ({result['final_confidence']:.3f})")
        print(f"  Reason: {result['final_reason']}")
        print(f"  Factors: {', '.join(result['decision_factors'])}")

        # Show image analysis results if available
        if 'image_analysis' in result['analysis_breakdown']:
            img_analysis = result['analysis_breakdown']['image_analysis']
            print(f"  Images: {img_analysis.get('bad_images', 0)} bad / {img_analysis.get('images_analyzed', 0)} analyzed")

        print()


if __name__ == "__main__":
    main()