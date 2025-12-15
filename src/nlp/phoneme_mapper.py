from typing import List, Optional

def word_to_phonemes(word: str) -> Optional[List[str]]:
    """
    Return first CMU pronunciation (ARPABET phones) for a word if available.
    """
    import cmudict
    d = cmudict.dict()
    pronunciations = d.get(word.lower())
    if not pronunciations:
        return None
    return pronunciations[0]

def words_to_phoneme_sequence(words: List[str]) -> List[str]:
    """
    Convert a list of words into a flat phoneme sequence.
    Unknown words are skipped for MVP.
    """
    seq: List[str] = []
    for w in words:
        phones = word_to_phonemes(w)
        if phones:
            seq.extend(phones)
    return seq
