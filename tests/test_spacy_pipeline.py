from src.nlp.spacy_pipeline import word_error_rate


def test_word_error_rate_perfect_match():
    words = ["the", "quick", "brown", "fox"]
    assert word_error_rate(words, words) == 0.0

def test_word_error_rate_penalizes_substitution():
    ref = ["the", "quick", "brown", "fox"]
    hyp = ["the", "slow", "brown", "fox"]
    assert word_error_rate(ref, hyp) > 0.0

def test_word_error_rate_lemmatization_tolerant_of_inflection():
    # "jump" vs "jumps" is a morphological variant, not a reading error --
    # lemmatization should score this as (much) closer than a raw substitution.
    ref = ["they", "jump", "high"]
    hyp = ["they", "jumps", "high"]
    inflected_wer = word_error_rate(ref, hyp)

    unrelated_hyp = ["they", "purple", "high"]
    substitution_wer = word_error_rate(ref, unrelated_hyp)

    assert inflected_wer <= substitution_wer

def test_word_error_rate_empty_reference():
    assert word_error_rate([], ["anything"]) == 0.0
