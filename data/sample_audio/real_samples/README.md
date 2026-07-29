# Real sample audio

5 short clips (16kHz mono WAV) extracted from the **LibriSpeech `dev-clean`** corpus
(Panayotov et al., 2015), used here as stand-in real read-speech audio so the
transcription -> phoneme -> acoustic-feature pipeline can be exercised end-to-end on
genuine recordings during development, testing, and synthetic-label generation
(see `scripts/generate_training_data.py`).

- **Source:** https://www.openslr.org/12 (speaker 2277, chapter 149896)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Attribution:** LibriSpeech (c) 2014 by Vassil Panayotov, http://creativecommons.org/licenses/by/4.0/

`transcripts.json` holds the corpus's ground-truth transcript for each clip, used as
the "prompt" reference text when exercising the feature pipeline.

**Note:** LibriSpeech is adult audiobook narration, not child speech, and carries no
dyslexia-related labels of any kind. It is used purely to have real (non-synthetic)
audio flowing through the pipeline. See `docs/ETHICS_PRIVACY.md` and the root
`README.md` for why no dyslexia-labeled dataset is used, and how training labels
in this project are generated synthetically and disclosed as such.
