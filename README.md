# NeuroAid A

AI-powered dyslexia pre-screener using speech recognition and phoneme analysis.

## Tech Stack
- Python 3.10+
- OpenAI Whisper
- SpaCy
- CMU Pronouncing Dictionary
- Scikit-learn
- Streamlit / Gradio

## Setup
```bash
git clone <repo-url>
cd neuroaid-ai
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
