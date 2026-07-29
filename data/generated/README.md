# Generated training data

`training_data.csv` (git-ignored, regenerate locally) is produced by
`scripts/generate_training_data.py`. It is **not** real clinical data:

- Acoustic features (MFCC, pause ratio) come from real audio
  (`data/sample_audio/real_samples`), so they're genuine signal.
- Lexical/phoneme features come from the project's real spaCy + CMU
  Pronouncing Dictionary pipeline, run against synthetically perturbed
  transcripts to simulate a range of reading-fluency profiles.
- The **label** is a documented rule-based heuristic (see the script's
  `heuristic_label` function), not a clinical diagnosis of any kind.

The model trained on this data is a demonstration of the ML pipeline
mechanics (feature extraction -> train -> serve a plain-language risk
assessment), not a validated dyslexia screening classifier. Every API
response carrying a `risk_assessment` includes a disclaimer to that effect.
