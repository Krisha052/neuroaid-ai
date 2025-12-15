import re

def clean_transcript(text: str) -> str:
    """
    Normalize transcript for analysis:
    - lowercase
    - remove extra whitespace
    - keep letters, apostrophes, and spaces
    """
    t = text.lower()
    t = re.sub(r"[^a-z'\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def tokenize(text: str) -> list[str]:
    return [w for w in text.split(" ") if w]
