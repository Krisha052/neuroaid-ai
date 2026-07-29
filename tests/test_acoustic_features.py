from pathlib import Path

from src.audio.acoustic_features import N_MFCC, extract_acoustic_features

REAL_SAMPLE = (
    Path(__file__).resolve().parents[1]
    / "data" / "sample_audio" / "real_samples" / "2277-149896-0033.wav"
)

def test_extract_acoustic_features_on_real_audio():
    features = extract_acoustic_features(REAL_SAMPLE)

    for i in range(N_MFCC):
        assert f"mfcc_{i}_mean" in features
        assert f"mfcc_{i}_std" in features

    assert 0.0 <= features["pause_ratio"] <= 1.0
