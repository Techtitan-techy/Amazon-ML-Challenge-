import re
import unicodedata
import anyascii

# Regex pre-compilations
RE_WHITESPACE = re.compile(r'\s+')
RE_NON_ALPHANUM = re.compile(r'[^a-z0-9]')
RE_PUNCT = re.compile(r'[^\w\s]')

# Common legal suffixes canonical mapping
LEGAL_REPLACEMENTS = [
    (re.compile(r'\b(private limited|pvt ltd|pvt\. ltd\.|p\. ltd\.|private ltd)\b', re.IGNORECASE), 'pvt_ltd'),
    (re.compile(r'\b(limited|ltd|ltd\.)\b', re.IGNORECASE), 'ltd'),
    (re.compile(r'\b(corporation|corp|corp\.)\b', re.IGNORECASE), 'corp'),
    (re.compile(r'\b(incorporated|inc|inc\.)\b', re.IGNORECASE), 'inc'),
    (re.compile(r'\b(limited liability company|llc|l\.l\.c\.)\b', re.IGNORECASE), 'llc'),
    (re.compile(r'\b(limited liability partnership|llp|l\.l\.p\.)\b', re.IGNORECASE), 'llp'),
    (re.compile(r'\b(company|co|co\.)\b', re.IGNORECASE), 'co'),
]

# Address replacements
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

def raw(text: str) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    return "" if s.lower() in ("none", "null", "nan", "") else s

def light_normalized(text: str) -> str:
    s = raw(text)
    if not s:
        return ""
    # Unicode NFKC normalization
    s = unicodedata.normalize('NFKC', s)
    s = s.lower()
    s = s.replace('&', ' and ')
    s = RE_PUNCT.sub(' ', s)
    return RE_WHITESPACE.sub(' ', s).strip()

def transliterated(text: str) -> str:
    s = raw(text)
    if not s:
        return ""
    # Transliterate Indic/French/any unicode script to Latin ASCII
    s = anyascii.anyascii(s)
    return light_normalized(s)

def token_normalized(text: str, is_name: bool = True) -> str:
    s = transliterated(text)
    if not s:
        return ""
    repls = LEGAL_REPLACEMENTS if is_name else ADDR_REPLACEMENTS
    for pat, rep in repls:
        s = pat.sub(rep, s)
    # Deduplicate and sort tokens for token-order invariance
    tokens = sorted(set(s.split()))
    return " ".join(tokens)

def compact_normalized(text: str) -> str:
    s = transliterated(text)
    if not s:
        return ""
    return RE_NON_ALPHANUM.sub('', s)

RE_US_ZIP = re.compile(r'\b(\d{5})(?:-\d{4})?\b')
RE_IN_PIN = re.compile(r'\b(\d{6})\b')
RE_HOUSE_NUM = re.compile(r'^\s*([0-9]+[a-z0-9\-\/]*)', re.IGNORECASE)

def extract_postal(text: str, country: str = "") -> str:
    s = raw(text)
    if not s:
        return ""
    country_upper = str(country).upper() if country else ""
    if country_upper == 'US':
        m = RE_US_ZIP.search(s)
        return m.group(1) if m else ""
    elif country_upper == 'INDIA':
        m = RE_IN_PIN.search(s)
        return m.group(1) if m else ""
    else:
        # Fallback: check 6-digit first then 5-digit
        m6 = RE_IN_PIN.search(s)
        if m6:
            return m6.group(1)
        m5 = RE_US_ZIP.search(s)
        return m5.group(1) if m5 else ""

def extract_house_num(text: str) -> str:
    s = transliterated(text)
    if not s:
        return ""
    m = RE_HOUSE_NUM.search(s)
    return m.group(1) if m else ""
