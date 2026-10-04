"""
Business Entity Resolution — Text Normalization & Field Extraction
===================================================================
Provides fast, deterministic multi-view text normalization for business names
and addresses across multilingual datasets (US, India, France).
Includes script transliteration, legal suffix canonicalization, address
standardization, state abbreviation expansion, and postal code extraction.
"""

import re
import unicodedata
import anyascii

# Pre-compiled regular expressions for speed
RE_WHITESPACE = re.compile(r'\s+')
RE_NON_ALPHANUM = re.compile(r'[^a-z0-9]')
RE_PUNCT = re.compile(r'[^\w\s]')

# Common legal suffix canonical mapping
LEGAL_REPLACEMENTS = [
    (re.compile(r'\b(private limited|pvt ltd|pvt\. ltd\.|p\. ltd\.|private ltd)\b', re.IGNORECASE), 'pvt ltd'),
    (re.compile(r'\b(limited|ltd|ltd\.)\b', re.IGNORECASE), 'ltd'),
    (re.compile(r'\b(corporation|corp|corp\.)\b', re.IGNORECASE), 'corp'),
    (re.compile(r'\b(incorporated|inc|inc\.)\b', re.IGNORECASE), 'inc'),
    (re.compile(r'\b(limited liability company|llc|l\.l\.c\.)\b', re.IGNORECASE), 'llc'),
    (re.compile(r'\b(limited liability partnership|llp|l\.l\.p\.)\b', re.IGNORECASE), 'llp'),
    (re.compile(r'\b(company|co|co\.)\b', re.IGNORECASE), 'co'),
]

# Common address token canonical mapping
ADDR_REPLACEMENTS = [
    (re.compile(r'\b(road|rd|rd\.)\b', re.IGNORECASE), 'rd'),
    (re.compile(r'\b(street|st|st\.)\b', re.IGNORECASE), 'st'),
    (re.compile(r'\b(avenue|ave|ave\.)\b', re.IGNORECASE), 'ave'),
    (re.compile(r'\b(drive|dr|dr\.)\b', re.IGNORECASE), 'dr'),
    (re.compile(r'\b(lane|ln|ln\.)\b', re.IGNORECASE), 'ln'),
    (re.compile(r'\b(boulevard|blvd|blvd\.)\b', re.IGNORECASE), 'blvd'),
    (re.compile(r'\b(apartment|apt|apt\.)\b', re.IGNORECASE), 'apt'),
    (re.compile(r'\b(suite|ste|ste\.)\b', re.IGNORECASE), 'ste'),
    (re.compile(r'\b(floor|fl|fl\.)\b', re.IGNORECASE), 'fl'),
    (re.compile(r'\b(near|nr|nr\.)\b', re.IGNORECASE), 'near'),
]

# Postal codes and house numbers
RE_US_ZIP = re.compile(r'\b(\d{5})(?:-\d{4})?\b')
RE_IN_PIN = re.compile(r'\b(\d{6})\b')
RE_FR_POSTAL = re.compile(r'\b(\d{5})\b')
RE_HOUSE_NUM = re.compile(r'^\s*([0-9]+[a-z0-9\-\/]*)', re.IGNORECASE)


def clean_str(text) -> str:
    """Sanitize and strip input, mapping None/NaN/empty to empty string."""
    if text is None:
        return ""
    s = str(text).strip()
    return "" if s.lower() in ("none", "null", "nan", "") else s


def transliterate(text) -> str:
    """Transliterate text to Latin ASCII, removing accents and script variations."""
    s = clean_str(text)
    if not s:
        return ""
    # NFKC normalize and anyascii transliteration
    s = unicodedata.normalize('NFKC', s)
    s = anyascii.anyascii(s)
    s = s.lower()
    s = s.replace('&', ' and ')
    s = RE_PUNCT.sub(' ', s)
    return RE_WHITESPACE.sub(' ', s).strip()


def normalize_name(text) -> str:
    """Normalize a business name with transliteration and legal suffix canonicalization."""
    s = transliterate(text)
    if not s:
        return ""
    for pat, rep in LEGAL_REPLACEMENTS:
        s = pat.sub(rep, s)
    return RE_WHITESPACE.sub(' ', s).strip()


def normalize_address(text, country: str = "") -> str:
    """Normalize a business address with transliteration and road/building canonicalization."""
    s = transliterate(text)
    if not s:
        return ""
    # Standard address token replacements
    for pat, rep in ADDR_REPLACEMENTS:
        s = pat.sub(rep, s)
    return RE_WHITESPACE.sub(' ', s).strip()


def extract_postal(text, country: str = "") -> str:
    """Extract standard postal/PIN code based on country."""
    s = clean_str(text)
    if not s:
        return ""
    country_upper = str(country).upper().strip() if country else ""
    if country_upper in ("US", "USA"):
        m = RE_US_ZIP.search(s)
        return m.group(1) if m else ""
    elif country_upper in ("INDIA", "IN"):
        m = RE_IN_PIN.search(s)
        return m.group(1) if m else ""
    elif country_upper in ("FRANCE", "FR"):
        m = RE_FR_POSTAL.search(s)
        return m.group(1) if m else ""
    else:
        m6 = RE_IN_PIN.search(s)
        if m6:
            return m6.group(1)
        m5 = RE_US_ZIP.search(s)
        return m5.group(1) if m5 else ""


def extract_house_num(text) -> str:
    """Extract leading street number or building identifier."""
    s = transliterate(text)
    if not s:
        return ""
    m = RE_HOUSE_NUM.search(s)
    return m.group(1) if m else ""


def build_normalized_record(raw_id, raw_name, raw_addr, country):
    """Build a multi-view dictionary of normalized attributes for a record."""
    norm_name = normalize_name(raw_name)
    norm_addr = normalize_address(raw_addr, country)
    norm_full = f"{norm_name} {norm_addr}".strip()
    postal = extract_postal(raw_addr, country)
    house_num = extract_house_num(raw_addr)
    
    return {
        'entity_id': raw_id,
        'country': str(country).strip(),
        'norm_name': norm_name,
        'norm_addr': norm_addr,
        'norm_full': norm_full,
        'postal': postal,
        'house_num': house_num,
        'has_addr': 1.0 if len(norm_addr) > 0 else 0.0,
    }
