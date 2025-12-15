# NeuroAid AI Design (MVP)

NeuroAid AI is a privacy-first screening assistant for early literacy risk signals.

## System overview
Input: short reading prompts + audio
Pipeline:
1) Audio capture/upload
2) Audio preprocessing (trim, normalize, ensure mono/16k)
3) Transcription (Whisper local or API)
4) Text cleaning and alignment
5) Feature extraction:
   - Reading rate (approx WPM)
   - Pause statistics (silences)
   - Word error patterns (substitution/omission)
   - Phoneme mismatch (via CMU dict)
6) Model inference (optional MVP)
7) Report generation (UI + PDF)

## Components
- `src/audio`: recording + preprocessing + diagnostics
- `src/speech_to_text`: Whisper wrapper + text cleaning
- `src/nlp`: phoneme mapping + feature extraction
- `src/model`: training + prediction utilities
- `src/web`: Streamlit app UI
- `src/utils`: shared helpers and PDF export

## MVP decisions
- Default mode: no persistence (process in memory, allow user-controlled export only)
- Model optional: rule-based features alone can provide meaningful screening signals
- Use small, curated prompt sets for consistency

## Future extensions
- Multi-language prompt sets
- Calibrated scoring with a larger dataset
- Teacher dashboard mode (opt-in, with strict consent + storage rules)
