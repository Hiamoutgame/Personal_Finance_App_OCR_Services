import re
import unicodedata


def _strip_accents(text):
    normalized = unicodedata.normalize("NFD", text or "")
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def _norm_text(text):
    text = _strip_accents(text).lower()
    return re.sub(r"\s+", " ", text).strip()
