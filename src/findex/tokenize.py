import re
import unicodedata
from collections.abc import Iterator

# Tokenization rules:
# 1. Unicode NFC normalization merges base letters and combining marks (e.g. café).
# 2. str.casefold() performs aggressive caseless folding (e.g. 'ß' -> 'ss').
# 3. Apostrophes (' and ’) and hyphens (-) are preserved inside words
#    (e.g. "п'ять", "don't", "state-of-the-art") and stripped at word boundaries.
# 4. Digits are preserved as standalone numbers ("2026") or parts of terms ("utf8").
TOKEN_RE = re.compile(r"\w+(?:['’-]\w+)*", re.UNICODE)


def tokenize(text: str) -> Iterator[str]:
    """Stream normalized tokens from input text."""
    normalized = unicodedata.normalize("NFC", text).casefold()
    for match in TOKEN_RE.finditer(normalized):
        yield match.group(0)
