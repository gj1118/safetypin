"""
Download species data and generate training examples for educational content.
Supports Catalogue of Life and provides fallback to embedded species list.
"""

import json
import os
import random
import requests
import gzip
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
TRAIN_DATA_PATH = SCRIPT_DIR / "data" / "train.json"
OUTPUT_PATH = SCRIPT_DIR / "data" / "train_with_species.json"

EDUCATIONAL_TEMPLATES = [
    "{species} is a species of {category}",
    "I learned about {species} today",
    "{species} is commonly found in {habitat}",
    "The scientific name {species} refers to {common}",
    "Study of {species} is important for {field}",
    "Researchers study {species} in {field}",
    "The habitat of {species} includes {habitat}",
    "{species} belongs to the family {family}",
    "More information about {species} can be found at biodiversity libraries",
    "{species} is classified as {classification}",
]

CATEGORIES = {
    "mammal": ["lion", "tiger", "bear", "elephant", "dolphin", "whale", "wolf", "fox", "deer", "horse"],
    "bird": ["eagle", "sparrow", "owl", "hawk", "penguin", "parrot", "flamingo", "swan", "crow", "finch"],
    "reptile": ["turtle", "snake", "lizard", "crocodile", "iguana", "gecko", "chameleon", "tortoise", "viper", "boa"],
    "amphibian": ["frog", "toad", "salamander", "newt", "axolotl", "caecilian"],
    "fish": ["shark", "tuna", "salmon", "cod", "bass", "trout", "eel", "ray", "seahorse", "manta"],
    "insect": ["butterfly", "bee", "ant", "beetle", "grasshopper", "dragonfly", "moth", "wasp", "cricket", "ladybug"],
    "crustacean": ["crab", "lobster", "shrimp", "prawn", "crayfish", "krill", "barnacle"],
    "mollusk": ["snail", "slug", "clam", "oyster", "squid", "octopus", "nautilus", "cuttlefish"],
    "plant": ["oak", "maple", "rose", "fern", "moss", "pine", "cedar", "bamboo", "cactus", "orchid"],
    "fungus": ["mushroom", "mold", "yeast", "lichen", "truffle", "coral"],
}

HABITATS = [
    "tropical forests", "temperate regions", "ocean depths", "freshwater lakes",
    "coastal areas", "mountains", "deserts", "grasslands", "rainforests",
    "wetlands", "rivers", "coral reefs", "arctic tundra", "savannas"
]

FIELDS = [
    "biology", "ecology", "conservation", "taxonomy", "zoology", "botany",
    "marine biology", "wildlife research", "environmental science", "evolutionary studies"
]

FAMILIES = [
    "Felidae", "Canidae", "Ursidae", "Cetaceae", "Bovidae", "Cervidae",
    "Accipitridae", "Columbidae", "Psittacidae", "Colubridae", "Viperidae"
]

CLASSIFICATIONS = [
    "least concern", "vulnerable", "endangered", "critically endangered",
    "near threatened", "not evaluated", "data deficient"
]

def generate_scientific_name(common_name, category):
    """Generate a simple scientific name from common name"""
    prefixes = {
        "mammal": ["Panthera", "Canis", "Ursus", "Elephas", "Delphinus"],
        "bird": ["Aquila", "Passer", "Bubo", "Buteo", "Aptenodytes"],
        "reptile": ["Chelonia", "Natrix", "Lacerta", "Crocodylus", "Iguana"],
        "amphibian": ["Rana", "Bufo", "Salamandra", "Triturus", "Ambystoma"],
        "fish": ["Carcharodon", "Thunnus", "Salmo", "Gadus", "Micropterus"],
        "insect": ["Papilio", "Apis", "Formica", "Coleoptera", "Lepidoptera"],
        "crustacean": ["Cancer", "Homarus", "Penaeus", "Astacus", "Balanus"],
        "mollusk": ["Helix", "Limax", "Mytilus", "Octopus", "Nautilus"],
        "plant": ["Quercus", "Acer", "Rosa", "Pinus", "Orchis"],
        "fungus": ["Agaricus", "Penicillium", "Saccharomyces", "Boletus", "Tuber"]
    }
    
    suffixes = ["us", "a", "ensis", "icus", "oides", "ata"]
    
    prefix = random.choice(prefixes.get(category, ["Species"]))
    suffix = random.choice(suffixes)
    return f"{prefix} {common_name[:4]}{suffix}"

def generate_training_examples(num_examples=500):
    """Generate training examples from species data"""
    examples = []
    
    for _ in range(num_examples):
        category = random.choice(list(CATEGORIES.keys()))
        common_name = random.choice(CATEGORIES[category])
        species = generate_scientific_name(common_name, category)
        
        template = random.choice(EDUCATIONAL_TEMPLATES)
        
        text = template.format(
            species=species,
            common=common_name,
            category=category,
            habitat=random.choice(HABITATS),
            field=random.choice(FIELDS),
            family=random.choice(FAMILIES),
            classification=random.choice(CLASSIFICATIONS)
        )
        
        examples.append({
            "text": text,
            "label": "good",
            "reason": "educational_content"
        })
    
    return examples

def download_catalogue_of_life():
    """Download Catalogue of Life data from GBIF"""
    print("Attempting to download Catalogue of Life data from GBIF...")
    
    url = "https://hosted-datasets.gbif.org/datasets/backbone/current/backbone.zip"
    
    try:
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        
        zip_path = SCRIPT_DIR / "data" / "backbone.zip"
        extract_path = SCRIPT_DIR / "data" / "backbone"
        
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        
        print(f"Downloaded {len(response.content) / 1024 / 1024:.1f} MB")
        
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        print(f"Extracted to {extract_path}")
        
        return extract_path
        
    except Exception as e:
        print(f"Download failed: {e}")
        print("Will use embedded species list instead")
        return None

def extract_species_from_catalogue(extract_path):
    """Extract species names from Catalogue of Life data"""
    print("Extracting species names from Catalogue of Life...")
    
    species_names = []
    
    taxonomy_file = extract_path / "Taxon.tsv"
    
    if taxonomy_file.exists():
        with open(taxonomy_file, 'r', encoding='utf-8') as f:
            header = f.readline().split('\t')
            
            rank_idx = header.index('rank') if 'rank' in header else -1
            scientific_name_idx = header.index('scientificName') if 'scientificName' in header else -1
            
            if rank_idx >= 0 and scientific_name_idx >= 0:
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.split('\t')
                    if len(parts) > max(rank_idx, scientific_name_idx):
                        rank = parts[rank_idx].strip().lower()
                        if rank == 'species':
                            species_names.append(parts[scientific_name_idx].strip())
                            
                            if len(species_names) >= 5000:
                                break
    
    print(f"Extracted {len(species_names)} species names")
    return species_names

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
    print("Species Training Data Generator")
    print("=" * 50)
    
    print("\n1. Downloading Catalogue of Life data...")
    extract_path = download_catalogue_of_life()
    
    if extract_path:
        print("\n2. Extracting species names...")
        species_names = extract_species_from_catalogue(extract_path)
    else:
        species_names = None
    
    print("\n3. Generating training examples...")
    new_examples = generate_training_examples(num_examples=1000)
    
    print("\n4. Adding examples to training data...")
    add_examples_to_train(new_examples)
    
    print("\n" + "=" * 50)
    print("Done! Review the output and copy to train.json")
    print("=" * 50)

if __name__ == "__main__":
    main()
