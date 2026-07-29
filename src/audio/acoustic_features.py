from pathlib import Path
from typing import Dict

N_MFCC = 13

def extract_acoustic_features(audio_path: Path) -> Dict[str, float]:
    """
    Real acoustic features via librosa, replacing the pure-text proxies:

    - MFCC mean/std (13 coefficients): standard spectral-envelope
      representation of articulation, commonly used in speech pathology
      and fluency research.
    - pause_ratio: fraction of the clip that librosa detects as
      non-speech/silence. Elevated pausing/hesitation during oral reading
      is a well-documented correlate of decoding difficulty, so this is the
      one acoustic signal most directly relevant to a reading-fluency
      screener (as opposed to MFCCs, which mostly proxy voice/articulation).
    """
    import librosa
    import numpy as np

    y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    if y.size == 0:
        return {**{f"mfcc_{i}_mean": 0.0 for i in range(N_MFCC)},
                **{f"mfcc_{i}_std": 0.0 for i in range(N_MFCC)},
                "pause_ratio": 0.0}

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    voiced_intervals = librosa.effects.split(y, top_db=30)
    voiced_samples = sum(end - start for start, end in voiced_intervals)
    pause_ratio = 1.0 - (voiced_samples / len(y)) if len(y) else 0.0

    features: Dict[str, float] = {
        f"mfcc_{i}_mean": float(mfcc_mean[i]) for i in range(N_MFCC)
    }
    features.update({f"mfcc_{i}_std": float(mfcc_std[i]) for i in range(N_MFCC)})
    features["pause_ratio"] = float(np.clip(pause_ratio, 0.0, 1.0))
    return features
