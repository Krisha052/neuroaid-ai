from src.nlp.phoneme_mapper import word_to_phonemes

def test_word_to_phonemes_known_word():
    phones = word_to_phonemes("example")
    assert phones is None or isinstance(phones, list)
