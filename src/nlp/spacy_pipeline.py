from functools import lru_cache
from typing import List

from src.config import CONFIG


@lru_cache(maxsize=1)
def get_nlp():
    """Load the spaCy pipeline once per process (model load is expensive)."""
    import spacy
    return spacy.load(CONFIG.spacy_model, disable=["ner", "parser"])

def lemmatize(words: List[str]) -> List[str]:
    """
    Lemmatize a word list with spaCy so morphological variants (e.g. "jump" vs
    "jumps") don't get counted as reading errors -- only real substitutions,
    insertions, and deletions should. Falls back to the raw tokens if the
    model isn't available so the pipeline degrades gracefully rather than
    crashing.
    """
    if not words:
        return []
    try:
        nlp = get_nlp()
    except OSError:
        return words
    doc = nlp(" ".join(words))
    return [tok.lemma_.lower() for tok in doc if not tok.is_space]

def word_error_rate(reference: List[str], hypothesis: List[str]) -> float:
    """
    Standard WER via Levenshtein alignment (substitutions + deletions +
    insertions, normalized by reference length) over spaCy-lemmatized tokens,
    replacing the old set-overlap proxy.
    """
    ref = lemmatize(reference)
    hyp = lemmatize(hypothesis)
    if not ref:
        return 0.0

    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[n][m] / n
