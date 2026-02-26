"""
Scrape animals from animalia.bio and generate training data.
Uses curl to bypass SSL issues and excludes koala.
"""

import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
import time

SCRIPT_DIR = Path(__file__).parent
TRAIN_DATA_PATH = SCRIPT_DIR / "data" / "train.json"
OUTPUT_PATH = SCRIPT_DIR / "data" / "train_with_animalia.json"

ANIMALIA_BASE_URL = "https://animalia.bio"
EXCLUDE_NAMES = ["koala", "phascolarctos cinereus"]

def get_animal_urls_from_sitemap():
    """Get all animal URLs from sitemap using curl"""
    print("Fetching sitemap with curl...")
    
    result = subprocess.run(
        ["curl", "-s", "-L", f"{ANIMALIA_BASE_URL}/sitemap/animals.xml"],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    if result.returncode != 0:
        raise Exception(f"curl failed: {result.stderr}")
    
    content = result.stdout
    
    root = ET.fromstring(content)
    
    namespaces = {'sm': 'https://www.sitemaps.org/schemas/sitemap/0.9'}
    
    urls = []
    for url in root.findall('sm:url', namespaces):
        loc = url.find('sm:loc', namespaces)
        if loc is not None:
            urls.append(loc.text)
    
    print(f"Found {len(urls)} animal URLs in sitemap")
    return urls

def extract_animal_info(url):
    """Extract animal name and scientific name from animal page using curl"""
    try:
        result = subprocess.run(
            ["curl", "-s", "-L", url],
            capture_output=True,
            text=True,
            timeout=15
        )
        
        if result.returncode != 0:
            return None
        
        html = result.stdout
        
        title_match = re.search(r'<title>([^<]+) - Facts', html)
        if title_match:
            title = title_match.group(1).strip()
        else:
            return None
        
        scientific_name = title
        
        common_name_match = re.search(r'https://animalia\.bio/([a-z-]+)$', url)
        if common_name_match:
            common_name = common_name_match.group(1).replace('-', ' ').title()
        else:
            common_name = None
        
        return {
            "url": url,
            "scientific_name": scientific_name,
            "common_name": common_name
        }
        
    except Exception as e:
        return None

def generate_training_examples(animals):
    """Generate training examples from animal data"""
    examples = []
    
    templates = [
        "{name} is an animal species",
        "I learned about {name} today",
        "{name} can be found in nature",
        "The species {name} is fascinating",
        "Researchers study {name} in biology",
        "Information about {name} from animalia.bio",
        "{name} is part of the animal kingdom",
        "Wildlife includes {name}",
        "Study of {name} is important for ecology",
        "Biodiversity includes species like {name}",
    ]
    
    import random
    
    for animal in animals:
        name = animal.get("scientific_name") or animal.get("common_name")
        if not name:
            continue
        
        name_lower = name.lower()
        if any(exc in name_lower for exc in EXCLUDE_NAMES):
            print(f"  Skipping excluded: {name}")
            continue
        
        template = random.choice(templates)
        
        text = template.format(name=name)
        
        examples.append({
            "text": text,
            "label": "good",
            "reason": "educational_content"
        })
        
        if animal.get("common_name"):
            text2 = template.format(name=animal["common_name"])
            examples.append({
                "text": text2,
                "label": "good",
                "reason": "educational_content"
            })
    
    return examples

def add_examples_to_train(new_examples):
    """Add new examples to existing training data"""
    
    if not TRAIN_DATA_PATH.exists():
        print("Training data not found, creating new file")
        with open(TRAIN_DATA_PATH, 'w') as f:
            json.dump([], f, indent=2)
    
    with open(TRAIN_DATA_PATH, 'r') as f:
        train_data = json.load(f)
    
    print(f"Current training data: {len(train_data)} examples")
    
    existing_texts = {item['text'] for item in train_data}
    
    unique_examples = [ex for ex in new_examples if ex['text'] not in existing_texts]
    
    train_data.extend(unique_examples)
    
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    print(f"Added {len(unique_examples)} new examples")
    print(f"Total training data: {len(train_data)} examples")
    print(f"Output saved to: {OUTPUT_PATH}")
    
    return train_data

def main():
    print("=" * 50)
    print("Animalia.bio Scraper")
    print("=" * 50)
    
    print("\n1. Fetching animal URLs from sitemap...")
    animal_urls = get_animal_urls_from_sitemap()
    
    print(f"\n2. Processing {len(animal_urls)} animal pages...")
    print("   (This will take a while due to rate limiting)")
    
    animals = []
    max_animals = 2000
    
    for i, url in enumerate(animal_urls[:max_animals]):
        if i % 50 == 0:
            print(f"   Processing {i}/{min(max_animals, len(animal_urls))}...")
        
        animal_info = extract_animal_info(url)
        if animal_info:
            animals.append(animal_info)
        
        time.sleep(0.2)
    
    print(f"\nExtracted info for {len(animals)} animals")
    
    if animals:
        print(f"Sample animals:")
        for a in animals[:5]:
            print(f"  - {a.get('scientific_name')} ({a.get('common_name')})")
    
    print("\n3. Generating training examples...")
    new_examples = generate_training_examples(animals)
    
    print("\n4. Adding examples to training data...")
    add_examples_to_train(new_examples)
    
    print("\n" + "=" * 50)
    print("Done!")
    print("=" * 50)

if __name__ == "__main__":
    main()
