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

The above is the deterministic pipeline behind `POST /api/v1/screen` -- fast,
free, and requiring no external API key. `POST /api/v1/agent/screen` (below)
wraps the same steps in an LLM-driven orchestrator instead.

## Agent orchestration (`src/agent/`)

`POST /api/v1/agent/screen` is a second, additive endpoint where an Anthropic
Claude agent -- not application code -- decides the control flow, using
Claude's tool-use API:

- `src/agent/tools.py` exposes `transcribe`, `extract_features`, and
  `predict_risk` as callable tools over a shared `ScreeningContext` (so
  intermediate results don't need to round-trip through the LLM's tool-call
  arguments), plus two terminal tools: `recommend_retake` and
  `finalize_assessment`.
- `src/agent/orchestrator.py:run_agentic_screening` runs a **bounded** loop
  (hard cap of 4 iterations -- deliberately not open-ended, to keep cost and
  latency predictable). The agent decides, from the actual tool results,
  whether to proceed through the full pipeline or call `recommend_retake`
  instead -- e.g. if the transcript comes back empty, or a tool call itself
  fails (a malformed audio file surfaces as a tool error, which the agent
  then reasons over rather than the server crashing). On success, the agent
  synthesizes `finalize_assessment`'s explanation itself, replacing the
  static templated strings in `src/model/predict.py` with a response
  tailored to that specific result.
- The Anthropic client is dependency-injected into `run_agentic_screening`,
  so `tests/test_agent_orchestrator.py` exercises the full tool-calling loop,
  retake branch, and error handling with a scripted fake client -- no real
  API calls, no cost, no `ANTHROPIC_API_KEY` required in CI.
- Without `ANTHROPIC_API_KEY` configured, the endpoint returns 503
  (`AgentNotConfiguredError`) rather than crashing; `/api/v1/screen` is
  unaffected either way.

## SDK (`sdk/neuroaid_client/`)

A small, typed client package (kept in-repo, `pip install -e sdk/`) so any
downstream consumer -- human code or another LLM agent -- can call the API
reliably: typed exceptions mirroring `src/api/errors.py`, retry/backoff on
transient failures. `sdk/neuroaid_client/tool_schema.py` additionally exports
the screening operation as an Anthropic/OpenAI-style tool-use schema, so an
**external** LLM agent (distinct from `src/agent/`'s own server-side
orchestrator) can consume it as a tool -- demonstrated end-to-end in
`sdk/examples/downstream_llm_demo.py`.

## Components
- `src/api`: Flask app, request validation, error handling (production surface)
- `src/agent`: LLM tool-use orchestration layer for `/api/v1/agent/screen`
- `src/audio`: ephemeral upload handling + diagnostics + acoustic features
- `src/speech_to_text`: Whisper wrapper + text cleaning
- `src/nlp`: phoneme mapping, spaCy pipeline, feature extraction
- `src/model`: training + prediction utilities
- `src/web`: Streamlit demo client (calls the API over HTTP, no internal imports)
- `src/utils`: shared helpers and PDF export
- `sdk/`: typed client SDK + LLM tool-use schema for downstream consumers
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
