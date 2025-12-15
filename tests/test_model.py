from src.nlp.feature_extractor import extract_features

def test_extract_features_runs():
    prompt = ["the", "cat", "sat"]
    spoken = ["the", "cat"]
    res = extract_features(prompt, spoken, duration_sec=2.0)
    assert "wpm" in res.features
