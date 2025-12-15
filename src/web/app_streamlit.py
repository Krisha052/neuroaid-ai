import streamlit as st
from pathlib import Path

from src.config import CONFIG
from src.audio.recorder import save_uploaded_audio
from src.audio.diagnostics import wav_info
from src.speech_to_text.whisper_wrapper import transcribe
from src.speech_to_text.text_cleaning import clean_transcript, tokenize
from src.nlp.feature_extractor import extract_features
from src.utils.pdf_report import export_pdf_report

def load_prompts(path: Path) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections[current] = []
            continue
        if current:
            sections[current].append(line)
    return sections

st.set_page_config(page_title="NeuroAid AI", layout="centered")

st.title("NeuroAid AI")
st.caption("An informational early literacy screening assistant (not a diagnosis).")

prompts = load_prompts(CONFIG.sample_prompts_path)
prompt_set = st.selectbox("Choose a prompt set", list(prompts.keys()))
prompt_lines = prompts[prompt_set]

st.write("### Prompt")
st.write("\n".join(prompt_lines))

uploaded = st.file_uploader("Upload a WAV file (recommended for MVP)", type=["wav"])

mode = st.selectbox("Transcription mode", ["local", "none"], index=0)
st.info("If installs are still in progress, choose 'none' to test the UI and feature pipeline without transcription.")

if uploaded:
    audio_path = save_uploaded_audio(uploaded, CONFIG.sample_audio_dir, filename="uploaded.wav")
    st.success(f"Saved audio: {audio_path}")
    try:
        info = wav_info(audio_path)
        st.write("Audio diagnostics:", info)
        duration = float(info["duration_sec"])
    except Exception:
        duration = 0.0

    if st.button("Analyze"):
        # Transcribe
        tr = transcribe(audio_path, mode=mode, model_size=CONFIG.whisper_model_size)
        cleaned = clean_transcript(tr.text)
        spoken_words = tokenize(cleaned)

        # Prompt words: treat prompt as the reference text
        prompt_text = " ".join(prompt_lines)
        prompt_words = tokenize(clean_transcript(prompt_text))

        # Features
        feat_res = extract_features(prompt_words, spoken_words, duration_sec=duration)

        st.write("### Transcript")
        st.write(tr.text if tr.text else "(no transcript)")

        st.write("### Feature Summary (MVP)")
        st.json(feat_res.features)

        st.write("### Notes")
        st.json(feat_res.notes)

        # PDF
        if st.checkbox("Generate PDF report"):
            CONFIG.reports_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = CONFIG.reports_dir / "neuroaid_report.pdf"
            export_pdf_report(pdf_path, {**feat_res.features, **feat_res.notes})
            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF report", f, file_name="neuroaid_report.pdf", mime="application/pdf")

st.write("---")
st.write("### Resources (examples)")
st.write("- If concerns persist, consult a reading specialist or school psychologist for a formal evaluation.")
st.write("- Practice in low-stress settings; repeat screenings on different days for consistency.")
