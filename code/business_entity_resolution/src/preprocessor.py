"""preprocessor.py - Country-Agnostic Business Entity Normalization Engine.

Handles US, India, France, and any future open-set country without hardcoded filters.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

SUFFIX_NORM: dict[str, str] = {
    "private limited": "pvt ltd", "pvt limited": "pvt ltd",
    "pvt. limited": "pvt ltd", "pvt. ltd.": "pvt ltd", "pvt ltd": "pvt ltd",
    "corporation": "corp", "incorporated": "inc", "limited liability company": "llc",
    "limited": "ltd", "l.l.c.": "llc", "l.l.c": "llc",
    "co.": "co", "company": "co",
    "societe anonyme": "sa", "societe a responsabilite limitee": "sarl",
    "societe par actions simplifiee": "sas",
    "societe en nom collectif": "snc",
    "groupement d interet economique": "gie",
    "gesellschaft mit beschrankter haftung": "gmbh",
    "aktiengesellschaft": "ag",
    "private": "pvt",
}

SUFFIX_ABBR: dict[str, str] = {
    "corp.": "corp", "inc.": "inc", "ltd.": "ltd",
    "llp": "llp", "l.l.p.": "llp",
    "plc": "plc", "p.l.c.": "plc",
    "gmbh": "gmbh", "ag": "ag", "sa": "sa", "sas": "sas",
    "sarl": "sarl", "snc": "snc", "gie": "gie",
    "bv": "bv", "nv": "nv", "oy": "oy", "ab": "ab",
}

ADDR_NORM: dict[str, str] = {
    "street": "st", "st.": "st",
    "road": "rd", "rd.": "rd",
    "avenue": "ave", "ave.": "ave",
    "boulevard": "blvd", "bvd": "blvd", "bd": "blvd",
    "drive": "dr", "dr.": "dr",
    "lane": "ln", "ln.": "ln",
    "court": "ct", "ct.": "ct",
    "place": "pl", "pl.": "pl",
    "square": "sq", "sq.": "sq",
    "highway": "hwy", "hwy.": "hwy",
    "apartment": "apt", "apt.": "apt",
    "suite": "ste", "ste.": "ste",
    "building": "bldg", "bldg.": "bldg",
    "floor": "fl", "fl.": "fl",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "rue": "rue", "r.": "rue",
    "allee": "all",
    "impasse": "imp",
    "chemin": "che",
    "route": "rte",
    "nagar": "ngr", "marg": "mg",
    "colony": "col", "sector": "sec",
}

POSTAL_PATTERN = re.compile(r"\b(\d{5,6})\b")

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE = re.compile(r"\s+")
_AMP = re.compile(r"\s*&\s*")


class Preprocessor:
    """Stateless, open-set normalizer for business names and addresses."""

    _SUFFIX_MULTI: list[tuple[re.Pattern, str]] = [
        (re.compile(r"\b" + re.escape(k) + r"\b", re.IGNORECASE), v)
        for k, v in sorted(SUFFIX_NORM.items(), key=lambda x: -len(x[0]))
    ]

    _ADDR_MULTI: list[tuple[re.Pattern, str]] = [
        (re.compile(r"(?<![A-Za-z])" + re.escape(k) + r"(?![A-Za-z])", re.IGNORECASE), v)
        for k, v in sorted(ADDR_NORM.items(), key=lambda x: -len(x[0]))
    ]

    @staticmethod
    def transliterate(text: str) -> str:
        """Strip diacritics/accents while preserving base characters."""
        nfkd = unicodedata.normalize("NFKD", text)
        return "".join(c for c in nfkd if not unicodedata.combining(c))

    @classmethod
    def clean_name(cls, name: str) -> str:
        """Normalize a business name to a canonical form."""
        if not name or not isinstance(name, str):
            return ""
        try:
            text = cls.transliterate(name)
        except Exception:
            text = name
        text = text.lower()
        text = _AMP.sub(" and ", text)
        for pattern, replacement in cls._SUFFIX_MULTI:
            text = pattern.sub(replacement, text)
        tokens = text.split()
        normalized_tokens = []
        for tok in tokens:
            cleaned_tok = _PUNCT.sub("", tok)
            normalized_tokens.append(SUFFIX_ABBR.get(cleaned_tok, cleaned_tok))
        text = " ".join(normalized_tokens)
        text = _PUNCT.sub(" ", text)
        text = _WHITESPACE.sub(" ", text).strip()
        return text

    @classmethod
    def clean_address(cls, addr: str) -> str:
        """Normalize a business address."""
        if not addr or not isinstance(addr, str):
            return ""
        try:
            text = cls.transliterate(addr)
        except Exception:
            text = addr
        text = text.lower()
        text = _AMP.sub(" and ", text)
        for pattern, replacement in cls._ADDR_MULTI:
            text = pattern.sub(replacement, text)
        text = _PUNCT.sub(" ", text)
        text = _WHITESPACE.sub(" ", text).strip()
        return text

    @staticmethod
    def extract_address_digits(addr: str) -> set[str]:
        """Extract all numeric tokens (street numbers, postal codes) from an address."""
        if not addr:
            return set()
        return set(POSTAL_PATTERN.findall(addr)) | set(re.findall(r"\b\d+\b", addr))

    @staticmethod
    def extract_postal_code(addr: str) -> Optional[str]:
        """Return the first 5-6 digit postal code found in the address."""
        if not addr:
            return None
        m = POSTAL_PATTERN.search(addr)
        return m.group(1) if m else None

    @classmethod
    def ngrams(cls, text: str, n: int = 3) -> list[str]:
        """Character n-grams of a string."""
        text = text.replace(" ", "_")
        return [text[i:i + n] for i in range(max(0, len(text) - n + 1))]

    @classmethod
    def name_tokens(cls, cleaned_name: str) -> set[str]:
        """Word tokens from cleaned name, filtering stopwords."""
        _stopwords = {"the", "a", "an", "of", "and", "in", "at", "for",
                      "de", "la", "le", "les", "des", "du", "et", "en"}
        tokens = set(cleaned_name.split())
        return tokens - _stopwords

    @classmethod
    def process_record(cls, record: dict) -> dict:
        """Process a single data record dict and return enriched record."""
        record = dict(record)
        raw_name = record.get("business_name", "") or ""
        raw_addr = record.get("business_address", "") or ""
        record["cleaned_name"] = cls.clean_name(raw_name)
        record["cleaned_address"] = cls.clean_address(raw_addr)
        record["name_tokens"] = list(cls.name_tokens(record["cleaned_name"]))
        record["addr_digits"] = list(cls.extract_address_digits(raw_addr))
        record["postal_code"] = cls.extract_postal_code(raw_addr)
        record["name_ngrams"] = cls.ngrams(record["cleaned_name"], 3)
        return record
