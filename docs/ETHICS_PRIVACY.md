# Ethics & Privacy

NeuroAid AI is an educational screening tool and is not a diagnostic system.

## Informed consent (minors)
Before recording or uploading audio of a minor, obtain consent from a parent/guardian and the school/teacher where applicable.

Suggested consent language:
“I understand this tool provides informational screening signals and is not a diagnosis. I consent to recording the student’s voice for the purpose of generating a screening report.”

## Data handling
- Uploaded audio is written to a private temp file for the duration of a
  single screening request and is always deleted immediately afterward
  (`src/audio/recorder.py:ephemeral_audio_file`), whether the request
  succeeds or fails. Nothing is persisted server-side by default.
- Transcripts and features are returned in the API response and are not
  logged or stored server-side.
- The user can export a PDF report locally (optional, opt-in per request).
- No data is sent to third parties: transcription runs locally via Whisper
  (`mode=local`, the default). A `mode=none` option exists only to test the
  UI/feature pipeline without running transcription at all.
- Uploads are capped in size and duration (`src/config.py`) so a single
  request can't exhaust server resources or silently accept multi-hour
  recordings.

## Risks / limitations
- Speech recognition errors can produce false signals
- Dyslexia is complex; screening signals are not definitive
- Dialect, multilingualism, speech disorders, anxiety, and environmental noise can affect results
- This tool should not be used for high-stakes decisions

## Bias mitigation (practical steps)
- Provide clear disclaimers in UI
- Encourage repeated attempts in a calm setting
- Offer multilingual prompt options in future versions
- Evaluate performance across varied accents and speech rates

## Compliance notes (high-level)
- COPPA/FERPA considerations: avoid storing personally identifiable audio data
- Keep pilot runs anonymous where possible
- Use aggregated metrics for any research write-ups

## Model validity disclosure
No real dyslexia-labeled dataset is freely available for a project at this
scale -- genuine ones require paid licensing or IRB-gated research access,
appropriately, given it's pediatric health data. `models/risk_model.joblib`
is trained on a documented, rule-based **synthetic** label (see
`data/generated/README.md`), not real clinical outcomes. Every API response
carrying a `risk_assessment` includes a disclaimer, and this is not a
validated diagnostic tool.
