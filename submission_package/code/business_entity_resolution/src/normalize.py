"""
Championship Multi-View Text Normalization Layer for Business Entity Resolution.
Amazon ML Challenge 2026.

Supports:
- Multi-view Name Representations (Unicode NFKC, Legal forms, DBA/Domain, Phonetic skeleton, Indic aliases)
- Multi-view Address Representations (US, India, and France abbreviations, structured postal/house-number extraction)
- Open-Set Country Normalization (US, India, and France invariance)
"""

import re
import unicodedata
from typing import Dict, Any, Optional, Set, List, Tuple

# -----------------------------------------------------------------------------
# 1. LEGAL SUFFIXES (US, India, France)
# -----------------------------------------------------------------------------
LEGAL_SUFFIXES = {
    # US
    "incorporated": "inc",
    "corporation": "corp",
    "company": "co",
    "limited liability company": "llc",
    "limited": "ltd",
    "partnership": "ptnr",
    "general partnership": "gp",
    "limited partnership": "lp",
    # India
    "private limited": "pvt ltd",
    "private": "pvt",
    "limited liability partnership": "llp",
    "one person company": "opc",
    "proprietorship": "prop",
    # France (15% of Test Data - Zero Leakage)
    "societe par actions simplifiee": "sas",
    "societe par actions simplifiee unipersonnelle": "sasu",
    "societe a responsabilite limitee": "sarl",
    "societe anonyme": "sa",
    "entreprise unipersonnelle a responsabilite limitee": "eurl",
    "societe civile immobiliere": "sci",
    "societe en nom collectif": "snc",
    "groupement d interet economique": "gie",
    "etablissement": "etbl",
}

# Sort suffixes by length descending to match composite phrases first
SORTED_LEGAL_SUFFIXES = sorted(LEGAL_SUFFIXES.items(), key=lambda x: len(x[0]), reverse=True)

# -----------------------------------------------------------------------------
# 2. ADDRESS ABBREVIATIONS (US, India, France)
# -----------------------------------------------------------------------------
ADDR_ABBREVIATIONS = {
    # US / General
    r"\bst\b": "street",
    r"\brd\b": "road",
    r"\bave\b": "avenue",
    r"\bav\b": "avenue",
    r"\bdr\b": "drive",
    r"\bblvd\b": "boulevard",
    r"\bbld\b": "boulevard",
    r"\bln\b": "lane",
    r"\bct\b": "court",
    r"\bpkwy\b": "parkway",
    r"\bapt\b": "apartment",
    r"\bste\b": "suite",
    r"\bfl\b": "floor",
    r"\bflr\b": "floor",
    r"\bbldg\b": "building",
    r"\bhwy\b": "highway",
    r"\bexpy\b": "expressway",
    r"\bpl\b": "place",
    r"\bsq\b": "square",
    # India Specific
    r"\bopp\b": "opposite",
    r"\bnr\b": "near",
    r"\bextn\b": "extension",
    r"\bmkt\b": "market",
    r"\bsec\b": "sector",
    r"\bph\b": "phase",
    r"\bno\b": "number",
    r"\bcol\b": "colony",
    r"\bchwk\b": "chowk",
    r"\bngr\b": "nagar",
    r"\blyt\b": "layout",
    r"\bbzr\b": "bazaar",
    # France Specific (15% of Test Data)
    r"\br\b": "rue",
    r"\bimp\b": "impasse",
    r"\ball\b": "allee",
    r"\bche\b": "chemin",
    r"\brte\b": "route",
    r"\bres\b": "residence",
    r"\bbat\b": "batiment",
    r"\bzi\b": "zone industrielle",
    r"\bza\b": "zone activite",
}

# -----------------------------------------------------------------------------
# 3. INDIC-ENGLISH TRANSLITERATION & ALIAS MAPPING (Learned from Dataset)
# -----------------------------------------------------------------------------
INDIC_ALIASES = {
    r"\blaxmi\b": "lakshmi",
    r"\bluxmi\b": "lakshmi",
    r"\bshree\b": "sri",
    r"\bshri\b": "sri",
    r"\bchander\b": "chandra",
    r"\bbhavan\b": "bhawan",
    r"\bjewellers\b": "jewelers",
    r"\bcloth\b": "textile",
    r"\bmedical\b": "pharma",
    r"\bvidyalaya\b": "school",
    r"\bvidyapith\b": "college",
    r"\bkirana\b": "general store",
    r"\bauto\b": "automobiles",
    r"\bdresses\b": "fashion",
}

# -----------------------------------------------------------------------------
# 4. DBA / DOMAIN REGEX PATTERNS
# -----------------------------------------------------------------------------
DBA_PREFIXES = re.compile(
    r"\b(doing business as|d\s*/?\s*b\s*/?\s*a|trading as|t\s*/?\s*a|c\s*/?\s*o|care of|aka|a\s*/?\s*k\s*/?\s*a)\b",
    re.IGNORECASE
)
DOMAIN_CLEAN_RE = re.compile(
    r"(https?://)?(www\.)?([a-zA-Z0-9\-]+)\.(com|org|net|in|fr|co|io|biz|info|store|shop|online)",
    re.IGNORECASE
)

# -----------------------------------------------------------------------------
# CORE PRIMITIVE NORMALIZERS
# -----------------------------------------------------------------------------
def nfkc_normalize(text: Optional[str]) -> str:
    """Apply Unicode NFKD/NFKC normalization, strip accents, and clean text."""
    if not text:
        return ""
    text = str(text)
    # Decompose Unicode and strip combining diacritical marks (e.g. é -> e, à -> a)
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    text = unicodedata.normalize("NFKC", stripped)
    return text.strip()


def light_normalize(text: Optional[str]) -> str:
    """Convert to lowercase, strip excess whitespace."""
    text = nfkc_normalize(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def token_normalize(text: Optional[str]) -> str:
    """
    Core token normalization:
    1. NFKD accent stripping & NFKC normalization
    2. Lowercase
    3. Replace punctuation/symbols with spaces
    4. Collapse spaces
    """
    text = nfkc_normalize(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compact_normalize(text: Optional[str]) -> str:
    """Remove all whitespace from token-normalized text."""
    return token_normalize(text).replace(" ", "")


# -----------------------------------------------------------------------------
# PHONETIC SKELETON (Soundex / Consonant Compression)
# -----------------------------------------------------------------------------
def compute_phonetic_skeleton(text: Optional[str]) -> str:
    """
    Fast consonant/phonetic skeleton representation for catching typos,
    OCR transpositions, and Indic transliteration variance.
    1. Strip vowels except leading letter
    2. Collapse duplicate consonants
    3. Normalize phonetically equivalent clusters (ph->f, wr->r, kn->n, c->k)
    """
    norm = token_normalize(text)
    if not norm:
        return ""

    tokens = norm.split()
    skeleton_tokens = []
    vowels = set("aeiouy")

    for tok in tokens[:5]:  # Focus on primary name tokens
        if not tok:
            continue
        # Map phonetic clusters
        t = tok.replace("ph", "f").replace("wr", "r").replace("kn", "n")
        t = t.replace("ksh", "ks").replace("x", "ks")
        t = re.sub(r"[ckq]", "k", t)
        t = re.sub(r"[zs]", "s", t)
        t = re.sub(r"[dt]", "t", t)

        # Retain initial character, strip internal vowels
        first_char = t[0]
        rest = "".join(c for c in t[1:] if c not in vowels)
        compressed = first_char + rest

        # Collapse repeated consecutive characters (e.g. tt -> t)
        deduped = re.sub(r"(.)\1+", r"\1", compressed)
        if len(deduped) >= 2:
            skeleton_tokens.append(deduped)
        elif len(deduped) == 1:
            skeleton_tokens.append(deduped)

    return " ".join(skeleton_tokens)


# -----------------------------------------------------------------------------
# DOMAIN & DBA PARSING
# -----------------------------------------------------------------------------
def clean_dba_and_domains(text: Optional[str]) -> str:
    """
    Extract entity name by cleaning DBA markers and domain URLs:
    e.g. 'foo.com' -> 'foo'
    e.g. 'Apex Logistics dba Apex Express' -> 'apex express' or 'apex logistics'
    """
    if not text:
        return ""
    text_str = str(text)

    # 1. Clean domain names (e.g. www.acme-corp.com -> acme corp)
    match = DOMAIN_CLEAN_RE.search(text_str)
    if match:
        domain_core = match.group(3).replace("-", " ").replace("_", " ")
        text_str = DOMAIN_CLEAN_RE.sub(domain_core, text_str)

    # 2. Check DBA markers
    dba_split = DBA_PREFIXES.split(text_str)
    if len(dba_split) > 1:
        # Prefer the operating/trade name if sufficiently long, else registered name
        t1 = token_normalize(dba_split[0])
        t2 = token_normalize(dba_split[-1])
        return t2 if len(t2) >= 4 else t1

    return token_normalize(text_str)


# -----------------------------------------------------------------------------
# BUSINESS NAME NORMALIZATION
# -----------------------------------------------------------------------------
def normalize_business_name(name: Optional[str]) -> str:
    """
    Full normalized representation for business names:
    - DBA / Domain parsed
    - Token normalized
    - Indic aliases normalized
    - Standard legal suffixes mapped (checking longest first)
    """
    norm = clean_dba_and_domains(name)
    if not norm:
        return ""

    # Apply Indic-English transliteration aliases
    for pat, rep in INDIC_ALIASES.items():
        norm = re.sub(pat, rep, norm)

    # Map legal suffixes
    for full_suffix, short_suffix in SORTED_LEGAL_SUFFIXES:
        if norm.endswith(" " + full_suffix):
            norm = norm[:-len(full_suffix)] + short_suffix
            break
        elif norm == full_suffix:
            norm = short_suffix
            break

    return norm.strip()


# -----------------------------------------------------------------------------
# ADDRESS NORMALIZATION & STRUCTURED COMPONENT EXTRACTION
# -----------------------------------------------------------------------------
def normalize_address(addr: Optional[str]) -> str:
    """
    Full normalized representation for business addresses:
    - Token normalized
    - Standardized abbreviations for US, India, and France
    """
    norm = token_normalize(addr)
    if not norm:
        return ""
    for pattern, replacement in ADDR_ABBREVIATIONS.items():
        norm = re.sub(pattern, replacement, norm)
    return norm.strip()


def extract_house_number(tokens: Optional[List[str]]) -> Optional[str]:
    """Extract house / building number digits from token list."""
    if not tokens:
        return None
    for tok in tokens[:6]:  # Usually at the beginning of an address
        if tok.isdigit() and len(tok) <= 5:
            return tok
        # Match alphanumeric like 12b, 45-a
        if any(c.isdigit() for c in tok) and len(tok) <= 6:
            cleaned = re.sub(r"[^\w]", "", tok)
            return cleaned
    return None


def extract_postal_code(tokens: Optional[List[str]], country: str) -> Optional[str]:
    """Extract 5-digit (US/France) or 6-digit (India) postal codes."""
    if not tokens:
        return None
    for tok in reversed(tokens):  # Postal codes usually appear near the end
        if tok.isdigit():
            if country == "INDIA" and len(tok) == 6:
                return tok
            elif country in ("US", "FRANCE") and len(tok) == 5:
                return tok
            elif len(tok) in (5, 6):
                return tok
    return None


def extract_numeric_tokens(tokens: Optional[List[str]]) -> Set[str]:
    """Extract all numeric tokens for numeric alignment verification."""
    if not tokens:
        return set()
    return {tok for tok in tokens if any(c.isdigit() for c in tok)}


# -----------------------------------------------------------------------------
# OPEN-SET COUNTRY NORMALIZATION
# -----------------------------------------------------------------------------
def normalize_country(country: Optional[str]) -> str:
    """Open-set country normalization (safe for US, India, France)."""
    return token_normalize(country).upper()


# -----------------------------------------------------------------------------
# MULTI-VIEW RECORD NORMALIZER
# -----------------------------------------------------------------------------
def normalize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return record with multi-view normalized representations exposing
    equivalent entities without destroying original evidence.
    """
    raw_name = rec.get("business_name") or ""
    raw_addr = rec.get("business_address") or ""
    raw_country = rec.get("country") or ""

    norm_name = normalize_business_name(raw_name)
    compact_name = compact_normalize(norm_name)
    phonetic_skel = compute_phonetic_skeleton(norm_name)
    norm_addr = normalize_address(raw_addr)
    country_code = normalize_country(raw_country)

    addr_tokens = norm_addr.split() if norm_addr else []
    name_tokens = norm_name.split() if norm_name else []

    house_num = extract_house_number(addr_tokens)
    postal_code = extract_postal_code(addr_tokens, country_code)
    num_tokens = extract_numeric_tokens(addr_tokens + name_tokens)

    return {
        "entity_id": rec.get("entity_id", ""),
        "raw_name": raw_name,
        "raw_addr": raw_addr,
        "country": country_code,
        "norm_name": norm_name,
        "compact_name": compact_name,
        "phonetic_skeleton": phonetic_skel,
        "norm_addr": norm_addr,
        "house_number": house_num or "",
        "postal_code": postal_code or "",
        "numeric_tokens": num_tokens,
    }
