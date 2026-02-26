"""
Fast scraper for animalia.bio - extracts names directly from sitemap URLs.
"""

import json
import re
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
TRAIN_DATA_PATH = SCRIPT_DIR / "data" / "train.json"
OUTPUT_PATH = SCRIPT_DIR / "data" / "train_with_animalia.json"

EXCLUDE_NAMES = ["koala", "phascolarctos"]

def get_all_animal_slugs():
    """Get all animal slugs from sitemap using curl"""
    print("Fetching sitemap...")
    
    result = subprocess.run(
        ["curl", "-s", "-L", "https://animalia.bio/sitemap/animals.xml"],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    content = result.stdout
    
    urls = re.findall(r'https://animalia\.bio/([a-z0-9-]+)\s*<', content)
    
    unique_slugs = list(set(urls))
    print(f"Found {len(unique_slugs)} animal slugs")
    return unique_slugs

def generate_training_examples(slugs):
    """Generate training examples from slug names"""
    examples = []
    
    templates = [
        "{name}",
        "{name}",
        "{name} is a species",
        "I learned about {name}",
        "Information about {name}",
    ]
    
    import random
    
    excluded_count = 0
    
    for slug in slugs:
        name = slug.replace('-', ' ').title()
        name_lower = name.lower()
        
        if "koala" in name_lower:
            excluded_count += 1
            continue
        
        template = random.choice(templates)
        text = template.format(name=name)
        
        examples.append({
            "text": text,
            "label": "good",
            "reason": "educational_content"
        })
    
    print(f"Excluded {excluded_count} koala entries")
    return examples

def add_examples_to_train(new_examples):
    """Add new examples to existing training data"""
    
    if not TRAIN_DATA_PATH.exists():
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
    print(f"Total: {len(train_data)} examples")
    
    return train_data

def main():
    print("=" * 50)
    print("Animalia.bio Fast Scraper")
    print("=" * 50)
    
    print("\n1. Fetching animal slugs from sitemap...")
    slugs = get_all_animal_slugs()
    
    print("\n2. Generating training examples...")
    new_examples = generate_training_examples(slugs)
    
    print("\n3. Adding to training data...")
    add_examples_to_train(new_examples)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
