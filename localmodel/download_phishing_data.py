#!/usr/bin/env python3
"""
Download and process phishing dataset from Hugging Face
Only used when URLs are present in content or for direct URL classification
"""

import os
import re
import json
import pandas as pd
from datasets import load_dataset
from urllib.parse import urlparse
from typing import List, Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhishingDatasetProcessor:
    def __init__(self, output_dir="data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def download_phishing_dataset(self):
        """Download phishing dataset from Hugging Face"""
        try:
            logger.info("Downloading phishing dataset from Hugging Face...")

            # Try the main phishing dataset
            dataset = load_dataset("ealvaradob/phishing-dataset")
            logger.info(f"Dataset loaded successfully")

            return dataset

        except Exception as e:
            logger.error(f"Failed to download main dataset: {e}")

            # Fallback to alternative dataset
            try:
                logger.info("Trying alternative phishing dataset...")
                dataset = load_dataset("pirocheto/phishing-url")
                logger.info("Alternative dataset loaded successfully")
                return dataset
            except Exception as e2:
                logger.error(f"Failed to download alternative dataset: {e2}")
                return None

    def extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url if url.startswith(('http://', 'https://')) else f'http://{url}')
            return parsed.netloc.lower()
        except:
            return url.lower()

    def process_phishing_dataset(self, dataset) -> Dict[str, str]:
        """Process dataset and create domain reputation mapping"""
        domain_reputation = {}

        if dataset is None:
            logger.warning("No dataset provided, returning empty mapping")
            return domain_reputation

        try:
            # Handle different dataset structures
            if 'train' in dataset:
                data = dataset['train']
            else:
                data = list(dataset.values())[0]  # Take first split

            logger.info(f"Processing {len(data)} records...")

            for i, record in enumerate(data):
                if i % 10000 == 0:
                    logger.info(f"Processed {i} records...")

                # Handle different column names
                url = None
                label = None

                # Extract URL and label from the record
                url = record.get('url')
                label = record.get('status')

                if url and label is not None:
                    domain = self.extract_domain(str(url))

                    # Normalize labels
                    if isinstance(label, str):
                        label_normalized = 'phishing' if label.lower() in ['phishing', 'malicious', 'bad', '1', 'true'] else 'legitimate'
                    elif isinstance(label, (int, float)):
                        label_normalized = 'phishing' if label == 1 else 'legitimate'
                    else:
                        continue

                    domain_reputation[domain] = label_normalized

            logger.info(f"Created domain reputation mapping for {len(domain_reputation)} domains")

            # Print some statistics
            phishing_count = sum(1 for v in domain_reputation.values() if v == 'phishing')
            legitimate_count = len(domain_reputation) - phishing_count

            logger.info(f"Phishing domains: {phishing_count}")
            logger.info(f"Legitimate domains: {legitimate_count}")

            return domain_reputation

        except Exception as e:
            logger.error(f"Error processing dataset: {e}")
            return {}

    def extract_urls_from_text(self, text: str) -> List[str]:
        """Extract URLs from text content"""
        url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            r'|(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:/[^\s]*)?'
        )

        urls = url_pattern.findall(text)
        return [url.strip() for url in urls if len(url) > 5]

    def get_domain_reputation(self, url: str, domain_reputation: Dict[str, str]) -> Optional[str]:
        """Get reputation for a specific domain"""
        domain = self.extract_domain(url)
        return domain_reputation.get(domain)

    def should_use_domain_analysis(self, text: str, url: Optional[str] = None) -> bool:
        """Determine if domain analysis should be used"""
        # Direct URL analysis
        if url:
            return True

        # Check if text contains URLs
        urls_in_text = self.extract_urls_from_text(text)
        return len(urls_in_text) > 0

    def enhance_text_with_domain_context(self,
                                       text: str,
                                       url: Optional[str],
                                       domain_reputation: Dict[str, str]) -> Dict:
        """Enhance text analysis with domain context when URLs are present"""

        result = {
            'text': text,
            'has_urls': False,
            'domain_analysis': None,
            'use_domain_features': False
        }

        # Check if we should use domain analysis
        if not self.should_use_domain_analysis(text, url):
            return result

        result['has_urls'] = True
        result['use_domain_features'] = True

        domains_found = []

        # Analyze direct URL
        if url:
            reputation = self.get_domain_reputation(url, domain_reputation)
            if reputation:
                domains_found.append({
                    'url': url,
                    'domain': self.extract_domain(url),
                    'reputation': reputation
                })

        # Analyze URLs in text
        urls_in_text = self.extract_urls_from_text(text)
        for found_url in urls_in_text:
            reputation = self.get_domain_reputation(found_url, domain_reputation)
            if reputation:
                domains_found.append({
                    'url': found_url,
                    'domain': self.extract_domain(found_url),
                    'reputation': reputation
                })

        # Calculate domain risk score
        if domains_found:
            phishing_domains = sum(1 for d in domains_found if d['reputation'] == 'phishing')
            total_domains = len(domains_found)

            domain_risk_score = phishing_domains / total_domains

            # Determine confidence threshold based on domain reputation
            if domain_risk_score > 0.5:
                confidence_threshold = 0.6  # Lower threshold for suspicious domains
            elif domain_risk_score == 0:
                confidence_threshold = 0.85  # Higher threshold for legitimate domains
            else:
                confidence_threshold = 0.75  # Medium threshold for mixed

            result['domain_analysis'] = {
                'domains_found': domains_found,
                'phishing_domains': phishing_domains,
                'total_domains': total_domains,
                'domain_risk_score': domain_risk_score,
                'confidence_threshold': confidence_threshold
            }

        return result

    def save_domain_reputation(self, domain_reputation: Dict[str, str], filename: str = "domain_reputation.json"):
        """Save domain reputation mapping to file"""
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(domain_reputation, f, indent=2)

        logger.info(f"Saved domain reputation mapping to {filepath}")
        return filepath


def main():
    """Main function to download and process phishing dataset"""
    processor = PhishingDatasetProcessor()

    # Download dataset
    dataset = processor.download_phishing_dataset()

    if dataset:
        # Process dataset
        domain_reputation = processor.process_phishing_dataset(dataset)

        # Save domain reputation mapping
        if domain_reputation:
            filepath = processor.save_domain_reputation(domain_reputation)

            # Test with some examples
            test_cases = [
                ("Visit our website at https://legitimate-bank.com for more info", None),
                ("Click here: http://phishing-site.tk/login", None),
                ("This is just plain text without any URLs", None),
                ("", "https://google.com")
            ]

            print("\n=== Testing Domain Analysis ===")
            for text, url in test_cases:
                analysis = processor.enhance_text_with_domain_context(text, url, domain_reputation)
                print(f"\nText: '{text[:50]}{'...' if len(text) > 50 else ''}'")
                print(f"URL: {url}")
                print(f"Use domain features: {analysis['use_domain_features']}")
                if analysis['domain_analysis']:
                    print(f"Domain risk score: {analysis['domain_analysis']['domain_risk_score']:.2f}")
                    print(f"Confidence threshold: {analysis['domain_analysis']['confidence_threshold']}")
        else:
            logger.error("Failed to process dataset")
    else:
        logger.error("Failed to download dataset")


if __name__ == "__main__":
    main()