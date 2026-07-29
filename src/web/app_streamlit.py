import requests
import streamlit as st

from src.config import API_BASE_URL

st.set_page_config(page_title="NeuroAid AI", layout="centered")

st.title("NeuroAid AI")
st.caption("An informational early literacy screening assistant (not a diagnosis).")
st.caption(f"Talking to API at `{API_BASE_URL}` -- this UI is a thin client; "
           "all processing happens in the Flask REST API.")

try:
    prompts = requests.get(f"{API_BASE_URL}/api/v1/prompts", timeout=5).json()
except requests.RequestException:
    st.error(f"Could not reach the NeuroAid API at {API_BASE_URL}. "
             "Make sure it's running (see README).")
    st.stop()

prompt_set = st.selectbox("Choose a prompt set", list(prompts.keys()))
st.write("### Prompt")
st.write(prompts[prompt_set])

uploaded = st.file_uploader("Upload a WAV file (recommended for MVP)", type=["wav"])
mode = st.selectbox("Transcription mode", ["local", "none"], index=0)
st.info("If installs are still in progress, choose 'none' to test the UI and "
        "feature pipeline without transcription.")

if uploaded and st.button("Analyze"):
    with st.spinner("Calling NeuroAid API..."):
        resp = requests.post(
            f"{API_BASE_URL}/api/v1/screen",
            files={"file": (uploaded.name, uploaded.getvalue())},
            data={"prompt_set": prompt_set, "transcription_mode": mode},
            timeout=120,
        )

    if not resp.ok:
        st.error(f"API error ({resp.status_code}): {resp.json().get('error', resp.text)}")
    else:
        result = resp.json()

        st.write("### Transcript")
        st.write(result["transcript"] or "(no transcript)")

        st.write("### Risk Assessment")
        risk = result.get("risk_assessment")
        if risk:
            st.metric("Risk band", risk["risk_band"].capitalize())
            st.write(risk["summary"])
            if risk.get("contributing_factors"):
                st.write("Contributing factors: " + ", ".join(risk["contributing_factors"]))
            st.caption(risk["disclaimer"])
        else:
            st.warning(result["notes"].get(
                "risk_model_missing",
                "No trained risk model available; showing raw features only."
            ))

        with st.expander("Feature details (advanced)"):
            st.json(result["features"])
            st.json(result["notes"])

        if st.button("Generate PDF report"):
            pdf_resp = requests.post(
                f"{API_BASE_URL}/api/v1/screen/report",
                files={"file": (uploaded.name, uploaded.getvalue())},
                data={"prompt_set": prompt_set, "transcription_mode": mode},
                timeout=120,
            )
            if pdf_resp.ok:
                st.download_button("Download PDF report", pdf_resp.content,
                                    file_name="neuroaid_report.pdf", mime="application/pdf")
            else:
                st.error("Could not generate PDF report.")

st.write("---")
st.write("### Resources (examples)")
st.write("- If concerns persist, consult a reading specialist or school psychologist "
         "for a formal evaluation.")
st.write("- Practice in low-stress settings; repeat screenings on different days for consistency.")
