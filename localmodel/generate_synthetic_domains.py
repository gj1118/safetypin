"""
Generate synthetic phishing domains using common attack patterns.
These patterns help the model learn to detect typosquatting and similar attacks.
"""

import random
import string
import json
from urllib.parse import urlparse

# Popular brands commonly used in phishing
BRAND_DOMAINS = {
    "google": ["google.com", "gmail.com", "drive.google.com"],
    "facebook": ["facebook.com", "fb.com", "messenger.com"],
    "apple": ["apple.com", "icloud.com", "appleid.apple.com"],
    "amazon": ["amazon.com", "aws.amazon.com", "pay.amazon.com"],
    "microsoft": ["microsoft.com", "office.com", "outlook.com", "onedrive.live.com"],
    "paypal": ["paypal.com", "paypal-us.com"],
    "netflix": ["netflix.com"],
    "instagram": ["instagram.com"],
    "twitter": ["twitter.com", "x.com"],
    "linkedin": ["linkedin.com"],
    "dropbox": ["dropbox.com"],
    "chase": ["chase.com"],
    "bankofamerica": ["bankofamerica.com"],
    "wellsfargo": ["wellsfargo.com"],
    "citi": ["citi.com"],
    "usbank": ["usbank.com"],
    "capitalone": ["capitalone.com"],
    "discover": ["discover.com"],
    "americanexpress": ["americanexpress.com"],
    "steam": ["steampowered.com", "steamcommunity.com"],
    "epicgames": ["epicgames.com"],
    "riotgames": ["riotgames.com"],
    "spotify": ["spotify.com"],
    "tiktok": ["tiktok.com"],
    "whatsapp": ["whatsapp.com"],
    "reddit": ["reddit.com"],
    "github": ["github.com"],
    "gitlab": ["gitlab.com"],
    "bitbucket": ["bitbucket.org"],
    "adobe": ["adobe.com"],
    "dropbox": ["dropbox.com"],
    "box": ["box.com"],
    "docusign": ["docusign.com"],
    "quickbooks": ["quickbooks.intuit.com"],
    "intuit": ["intuit.com"],
    "zoho": ["zoho.com"],
    "slack": ["slack.com"],
    "zoom": ["zoom.us"],
    "atlassian": ["atlassian.net"],
}

# Character substitutions for typosquatting
CHAR_SUBSTITUTIONS = {
    'a': ['@', '4', 'α'],
    'b': ['8', '6'],
    'e': ['3', 'ε'],
    'g': ['9', 'q'],
    'i': ['1', 'l', '!', '|'],
    'l': ['1', 'i', '|'],
    'o': ['0', 'ο'],
    's': ['5', '$'],
    't': ['7', '+'],
    'z': ['2'],
}

# Suspicious TLDs often used in phishing
SUSPICIOUS_TLDS = ['xyz', 'top', 'club', 'win', 'work', 'tk', 'ml', 'ga', 'cf', 'gq', 'pw', 'cc', 'ws', 'site', 'online', 'shop']

def generate_typosquatting(domain, count=3):
    """Generate typosquatting variants of a domain"""
    variants = []
    domain = domain.lower()
    
    # Get the base name (without TLD)
    parts = domain.rsplit('.', 1)
    if len(parts) != 2:
        return variants
    
    name, tld = parts
    
    for _ in range(count):
        variant_type = random.choice(['substitute', 'add_hyphen', 'add_number', 'transpose', 'missing_char'])
        
        if variant_type == 'substitute':
            # Replace characters with similar-looking ones
            new_name = list(name)
            for i in range(min(random.randint(1, 2), len(new_name))):
                idx = random.randint(0, len(new_name) - 1)
                char = new_name[idx]
                if char in CHAR_SUBSTITUTIONS:
                    new_name[idx] = random.choice(CHAR_SUBSTITUTIONS[char])
            variants.append(''.join(new_name) + '.' + tld)
        
        elif variant_type == 'add_hyphen':
            # Add hyphen in the name
            if len(name) > 2:
                idx = random.randint(1, len(name) - 1)
                variants.append(name[:idx] + '-' + name[idx:] + '.' + tld)
        
        elif variant_type == 'add_number':
            # Add number to the name
            idx = random.randint(0, len(name))
            variants.append(name[:idx] + str(random.randint(0, 9)) + name[idx:] + '.' + tld)
        
        elif variant_type == 'transpose':
            # Swap two adjacent characters
            if len(name) > 1:
                idx = random.randint(0, len(name) - 2)
                new_name = list(name)
                new_name[idx], new_name[idx + 1] = new_name[idx + 1], new_name[idx]
                variants.append(''.join(new_name) + '.' + tld)
        
        elif variant_type == 'missing_char':
            # Remove a character (if long enough)
            if len(name) > 3:
                idx = random.randint(0, len(name) - 1)
                variants.append(name[:idx] + name[idx+1:] + '.' + tld)
    
    return variants

def generate_brand_phishing_domains():
    """Generate phishing domains impersonating popular brands"""
    phishing_domains = []
    
    for brand, legitimate_domains in BRAND_DOMAINS.items():
        for legit_domain in legitimate_domains:
            # Generate typosquatting variants
            variants = generate_typosquatting(legit_domain, count=5)
            phishing_domains.extend(variants)
            
            # Add suspicious TLD versions
            name = legit_domain.rsplit('.', 1)[0] if '.' in legit_domain else legit_domain
            for tld in random.sample(SUSPICIOUS_TLDS, 3):
                phishing_domains.append(f"{name}-{random.choice(['login', 'verify', 'secure', 'update', 'account'])}.{tld}")
                phishing_domains.append(f"{name}{random.randint(100, 999)}.{tld}")
    
    return phishing_domains

def generate_generic_phishing():
    """Generate generic phishing domain patterns"""
    patterns = [
        "secure-{service}-{action}.{tld}",
        "{service}-{action}-{random}.{tld}",
        "{service}{random}-{action}.{tld}",
        "login-{service}-{random}.{tld}",
        "verify-{service}-{random}.{tld}",
        "account-{service}-{action}.{tld}",
        "{service}-support-{random}.{tld}",
        "{service}-security-{random}.{tld}",
    ]
    
    services = ['paypal', 'apple', 'google', 'facebook', 'amazon', 'microsoft', 'netflix', 'bank', 'email', 'webmail', 'office365', 'outlook', 'dropbox', 'steam']
    actions = ['login', 'verify', 'secure', 'account', 'update', 'confirm', 'signin', 'security', 'support']
    randoms = [str(random.randint(100, 999)), ''.join(random.choices(string.ascii_lowercase, k=3)), ''.join(random.choices(string.digits, k=4))]
    
    domains = []
    for _ in range(500):
        pattern = random.choice(patterns)
        domain = pattern.format(
            service=random.choice(services),
            action=random.choice(actions),
            random=random.choice(randoms),
            tld=random.choice(SUSPICIOUS_TLDS + ['com', 'net', 'org'])
        )
        domains.append(domain)
    
    return domains

def augment_domain_data(input_file="data/domain_reputation.json", output_file="data/domain_reputation_augmented.json"):
    """Augment the existing domain data with synthetic phishing domains"""
    
    # Load existing data
    with open(input_file, 'r') as f:
        existing_data = json.load(f)
    
    existing_domains = set(existing_data.keys())
    
    # Generate synthetic phishing domains
    print("Generating synthetic phishing domains...")
    brand_phishing = generate_brand_phishing_domains()
    generic_phishing = generate_generic_phishing()
    
    all_synthetic = brand_phishing + generic_phishing
    
    # Add only unique, non-existing domains
    new_domains = {}
    for domain in all_synthetic:
        domain = domain.lower()
        if domain not in existing_domains and domain not in new_domains:
            new_domains[domain] = "phishing"
    
    print(f"Generated {len(new_domains)} new phishing domains")
    
    # Merge with existing
    augmented_data = {**existing_data, **new_domains}
    
    # Save augmented data
    with open(output_file, 'w') as f:
        json.dump(augmented_data, f, indent=2)
    
    print(f"Saved augmented data to {output_file}")
    print(f"Total domains: {len(augmented_data)}")
    
    phishing_count = sum(1 for v in augmented_data.values() if v == "phishing")
    legitimate_count = len(augmented_data) - phishing_count
    print(f"  Phishing: {phishing_count}")
    print(f"  Legitimate: {legitimate_count}")
    
    return augmented_data

if __name__ == "__main__":
    augment_domain_data()
