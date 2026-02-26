"""
PDF Classifier for Child Safety
Extracts and analyzes text content and images from PDF files.
Uses existing trained text and image models.
"""

import os
import io
import json
import base64
from typing import List, Dict, Tuple, Optional
import torch
import numpy as np
from PIL import Image
import PyPDF2
import fitz  # PyMuPDF
from transformers import DistilBertTokenizer
import onnxruntime as ort
from torchvision import transforms
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PDFClassifier:
    def __init__(self,
                 text_model_path="output/bert_text_with_reasons",
                 image_model_path="output/image/image_classifier.onnx",
                 max_images=15,
                 max_pages=50):
        """
        Initialize PDF classifier with trained models.

        Args:
            text_model_path: Path to trained text classifier
            image_model_path: Path to trained image classifier (ONNX)
            max_images: Maximum number of images to analyze per PDF
            max_pages: Maximum number of pages to process per PDF
        """
        self.max_images = max_images
        self.max_pages = max_pages

        # Load text classifier
        self.text_model, self.tokenizer, self.reason_categories = self._load_text_model(text_model_path)

        # Load image classifier
        self.image_session, self.image_transform = self._load_image_model(image_model_path)

        logger.info("PDF Classifier initialized successfully")

    def _load_text_model(self, model_path):
        """Load the trained text classifier with reasons"""
        try:
            # Load DistilBERT with reasons model
            from train_text import DistilBertWithReasons

            tokenizer = DistilBertTokenizer.from_pretrained(model_path)
            model = DistilBertWithReasons(num_labels=2, num_reasons=11)
            model.load_state_dict(torch.load(f"{model_path}/model.pth", map_location="cpu"))
            model.eval()

            # Load reason categories
            with open(f"{model_path}/reason_categories.json") as f:
                reason_categories = json.load(f)

            logger.info("Text classifier loaded successfully")
            return model, tokenizer, reason_categories

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

    def extract_text_from_pdf(self, pdf_path: str) -> Dict:
        """
        Extract text content from PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dict containing extracted text and metadata
        """
        try:
            text_content = ""
            page_count = 0

            # Try PyMuPDF first (better text extraction)
            try:
                doc = fitz.open(pdf_path)
                total_pages = min(len(doc), self.max_pages)

                for page_num in range(total_pages):
                    page = doc[page_num]
                    text_content += page.get_text() + "\n"
                    page_count += 1

                doc.close()

            except Exception as e:
                logger.warning(f"PyMuPDF failed, trying PyPDF2: {e}")

                # Fallback to PyPDF2
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    total_pages = min(len(pdf_reader.pages), self.max_pages)

                    for page_num in range(total_pages):
                        page = pdf_reader.pages[page_num]
                        text_content += page.extract_text() + "\n"
                        page_count += 1

            # Clean up text
            text_content = text_content.strip()
            text_content = ' '.join(text_content.split())  # Normalize whitespace

            return {
                'text_content': text_content,
                'pages_processed': page_count,
                'text_length': len(text_content),
                'has_text': len(text_content) > 0
            }

        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {e}")
            return {
                'text_content': "",
                'pages_processed': 0,
                'text_length': 0,
                'has_text': False,
                'error': str(e)
            }

    def extract_images_from_pdf(self, pdf_path: str) -> List[Image.Image]:
        """
        Extract images from PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of PIL Image objects
        """
        images = []

        try:
            # Use PyMuPDF for image extraction
            doc = fitz.open(pdf_path)
            total_pages = min(len(doc), self.max_pages)

            for page_num in range(total_pages):
                if len(images) >= self.max_images:
                    break

                page = doc[page_num]
                image_list = page.get_images()

                for img_index, img in enumerate(image_list):
                    if len(images) >= self.max_images:
                        break

                    try:
                        # Get image data
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)

                        # Convert to PIL Image
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("ppm")
                            image = Image.open(io.BytesIO(img_data))
                        else:  # CMYK
                            pix_rgb = fitz.Pixmap(fitz.csRGB, pix)
                            img_data = pix_rgb.tobytes("ppm")
                            image = Image.open(io.BytesIO(img_data))
                            pix_rgb = None

                        pix = None

                        # Filter out very small images (likely decorative)
                        if image.size[0] >= 50 and image.size[1] >= 50:
                            images.append(image.convert('RGB'))

                    except Exception as e:
                        logger.warning(f"Failed to extract image {img_index} from page {page_num}: {e}")
                        continue

            doc.close()
            logger.info(f"Extracted {len(images)} images from PDF")

        except Exception as e:
            logger.error(f"Failed to extract images from PDF: {e}")

        return images

    def classify_text_content(self, text: str) -> Dict:
        """Classify text content using trained text model"""
        if not self.text_model or not text.strip():
            return {'classification': 'good', 'confidence': 0.5, 'reason': 'unknown'}

        try:
            # Tokenize text (limit to model's max length)
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)

            with torch.no_grad():
                outputs = self.text_model(**inputs)
                class_probs = torch.softmax(outputs["classification_logits"], dim=1).numpy()[0]
                reason_probs = torch.softmax(outputs["reason_logits"], dim=1).numpy()[0]

                class_predicted = np.argmax(class_probs)
                reason_predicted = np.argmax(reason_probs)
                confidence = float(class_probs[class_predicted])

                # Get reason name
                reason_names = {v: k for k, v in self.reason_categories.items()}
                predicted_reason = reason_names.get(reason_predicted, "unknown")

            label = "bad" if class_predicted == 0 else "good"

            return {
                'classification': label,
                'confidence': confidence,
                'reason': predicted_reason,
                'reason_description': predicted_reason.replace("_", " ").title()
            }

        except Exception as e:
            logger.error(f"Text classification failed: {e}")
            return {'classification': 'good', 'confidence': 0.5, 'reason': 'error'}

    def classify_image(self, image: Image.Image) -> Dict:
        """Classify a single image"""
        if not self.image_session:
            return {'classification': 'good', 'confidence': 0.5, 'error': 'Image model not loaded'}

        try:
            # Ensure image is RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Preprocess image
            image_tensor = self.image_transform(image).unsqueeze(0).numpy()

            # Run inference
            outputs = self.image_session.run(None, {"input": image_tensor})
            probs = self._softmax(outputs[0][0])
            predicted = np.argmax(probs)
            confidence = float(probs[predicted])

            label = "bad" if predicted == 0 else "good"

            return {
                'classification': label,
                'confidence': confidence,
                'image_size': image.size
            }

        except Exception as e:
            logger.warning(f"Failed to classify image: {e}")
            return {'classification': 'good', 'confidence': 0.5, 'error': str(e)}

    def _softmax(self, x):
        """Apply softmax to numpy array"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def classify_pdf_file(self, pdf_path: str) -> Dict:
        """
        Classify entire PDF file by analyzing text content and images.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Classification results with detailed analysis
        """
        logger.info(f"Analyzing PDF file: {pdf_path}")

        if not os.path.exists(pdf_path):
            return {
                'pdf_classification': 'error',
                'error': 'PDF file not found',
                'file_path': pdf_path
            }

        # Extract text content
        text_extraction = self.extract_text_from_pdf(pdf_path)

        # Classify text content
        text_result = self.classify_text_content(text_extraction['text_content'])

        # Extract and classify images
        images = self.extract_images_from_pdf(pdf_path)
        image_results = []
        bad_images = 0

        for i, image in enumerate(images):
            img_result = self.classify_image(image)
            img_result['image_index'] = i
            image_results.append(img_result)

            if img_result['classification'] == 'bad':
                bad_images += 1

        # Combine results to determine overall PDF classification
        pdf_classification = self._determine_pdf_classification(text_result, image_results, bad_images)

        # Get file info
        try:
            file_size = os.path.getsize(pdf_path)
            file_name = os.path.basename(pdf_path)
        except:
            file_size = 0
            file_name = pdf_path

        # Compile comprehensive results
        results = {
            'pdf_classification': pdf_classification['classification'],
            'pdf_confidence': pdf_classification['confidence'],
            'pdf_reasons': pdf_classification['reasons'],

            'text_analysis': {
                'classification': text_result['classification'],
                'confidence': text_result['confidence'],
                'reason': text_result.get('reason', 'unknown'),
                'reason_description': text_result.get('reason_description', 'Unknown'),
                'content_length': text_extraction['text_length'],
                'pages_processed': text_extraction['pages_processed'],
                'has_text': text_extraction['has_text']
            },

            'image_analysis': {
                'total_images': len(image_results),
                'bad_images': bad_images,
                'good_images': len(image_results) - bad_images,
                'image_results': image_results
            },

            'file_info': {
                'file_name': file_name,
                'file_size': file_size,
                'images_extracted': len(images),
                'images_analyzed': len(image_results)
            }
        }

        logger.info(f"PDF analysis complete: {pdf_classification['classification']} "
                   f"(Text: {text_result['classification']}, Bad images: {bad_images})")

        return results

    def classify_pdf_bytes(self, pdf_bytes: bytes, filename: str = "document.pdf") -> Dict:
        """
        Classify PDF from bytes data.

        Args:
            pdf_bytes: PDF file content as bytes
            filename: Name of the PDF file

        Returns:
            Classification results with detailed analysis
        """
        # Write bytes to temporary file
        import tempfile

        try:
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(pdf_bytes)
                temp_path = temp_file.name

            # Classify the temporary file
            results = self.classify_pdf_file(temp_path)

            # Update filename in results
            results['file_info']['file_name'] = filename
            results['file_info']['file_size'] = len(pdf_bytes)

            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except:
                pass

            return results

        except Exception as e:
            logger.error(f"Failed to process PDF bytes: {e}")
            return {
                'pdf_classification': 'error',
                'error': str(e),
                'file_name': filename
            }

    def _determine_pdf_classification(self, text_result: Dict, image_results: List[Dict], bad_images: int) -> Dict:
        """Determine overall PDF classification based on text and image analysis"""

        reasons = []

        # If text content is bad, PDF is bad
        if text_result['classification'] == 'bad':
            reasons.append(f"Bad text content: {text_result.get('reason_description', 'Unknown reason')}")
            return {
                'classification': 'bad',
                'confidence': max(text_result['confidence'], 0.8),
                'reasons': reasons
            }

        # If any images are bad, PDF is bad
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

        # If text is good and no bad images, PDF is good
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


def main():
    """Example usage of PDFClassifier"""

    # Initialize classifier
    classifier = PDFClassifier()

    # Example: classify a PDF file
    pdf_path = "example.pdf"
    if os.path.exists(pdf_path):
        results = classifier.classify_pdf_file(pdf_path)
        print(json.dumps(results, indent=2))
    else:
        print(f"PDF file {pdf_path} not found")


if __name__ == "__main__":
    main()