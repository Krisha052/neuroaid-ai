# NeuroAid AI Design

NeuroAid AI is a privacy-first screening assistant for early literacy risk signals,
served as a Flask REST API with a thin Streamlit demo client.

## System overview
Input: a reading prompt (text) + an audio recording of it being read aloud.

Pipeline (`src/api/service.py:run_screening`):
1. Upload validation (size/duration caps, format check) -- `src/api/service.py`
2. Audio written to an ephemeral temp file for the duration of processing only
   -- `src/audio/recorder.py`
3. Transcription via local Whisper -- `src/speech_to_text/whisper_wrapper.py`
4. Text cleaning and tokenization -- `src/speech_to_text/text_cleaning.py`
5. Feature extraction -- `src/nlp/feature_extractor.py`:
   - Reading rate (WPM)
   - Word error rate: Levenshtein alignment over spaCy-lemmatized tokens
     (`src/nlp/spacy_pipeline.py`) -- lemmatizing first means morphological
     variants ("jump" vs "jumps") aren't counted as reading errors, only real
     substitutions/insertions/deletions are.
   - Phoneme mismatch proxy via the CMU Pronouncing Dictionary
     (`src/nlp/phoneme_mapper.py`) -- a phoneme-sequence-length comparison,
     not true forced alignment; tracked as a known limitation, not oversold.
   - Acoustic features via MFCCs + pause ratio
     (`src/audio/acoustic_features.py`) -- pause ratio (fraction of the clip
     detected as non-speech) is the feature most directly tied to reading
     fluency; MFCCs capture general articulation/voice characteristics.
6. Risk inference (optional -- only runs if `models/risk_model.joblib`
   exists): `src/model/predict.py` translates the classifier's probability
   into a plain-language risk band + explanation, not a bare score.
7. Report generation (JSON via the API, or PDF via `src/utils/pdf_report.py`)

## Components
- `src/api`: Flask app, request validation, error handling (production surface)
- `src/audio`: ephemeral upload handling + diagnostics + acoustic features
- `src/speech_to_text`: Whisper wrapper + text cleaning
- `src/nlp`: phoneme mapping, spaCy pipeline, feature extraction
- `src/model`: training + prediction utilities
- `src/web`: Streamlit demo client (calls the API over HTTP, no internal imports)
- `src/utils`: shared helpers and PDF export
- `scripts/`: synthetic training-data generation + training CLI (see
  `data/generated/README.md` for why the labels are synthetic and disclosed as such)

## Design decisions
- No persistent storage of uploaded audio: processed via a temp file that's
  deleted immediately after the request, matching `docs/ETHICS_PRIVACY.md`.
- The risk model is optional at runtime: the API always returns lexical/
  phoneme/acoustic features; `risk_assessment` is only populated if a model
  file is present, and is explicitly disclosed as trained on synthetic
  labels (see root `README.md`).
- Errors are typed (`src/api/errors.py`) and mapped to specific HTTP status
  codes (400/413/415/422/500) rather than a single generic failure mode.

## Future extensions
- Multi-language prompt sets
- Calibrated scoring with a real (properly licensed/IRB-approved) labeled dataset
- Teacher dashboard mode (opt-in, with strict consent + storage rules)
- True forced phoneme alignment instead of the length-based proxy
