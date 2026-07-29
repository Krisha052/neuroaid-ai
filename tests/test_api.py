import io
from pathlib import Path

import pytest

from src.api.app import create_app

REAL_SAMPLE = (
    Path(__file__).resolve().parents[1]
    / "data" / "sample_audio" / "real_samples" / "2277-149896-0033.wav"
)

@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()

def test_health(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"

def test_prompts_lists_sections(client):
    resp = client.get("/api/v1/prompts")
    assert resp.status_code == 200
    assert isinstance(resp.get_json(), dict)

def test_screen_requires_file(client):
    resp = client.post("/api/v1/screen", data={"prompt_text": "hello"})
    assert resp.status_code == 400
    assert resp.get_json()["type"] == "InvalidInputError" or "error" in resp.get_json()

def test_screen_requires_prompt(client):
    with open(REAL_SAMPLE, "rb") as f:
        resp = client.post(
            "/api/v1/screen",
            data={"file": (f, "sample.wav")},
            content_type="multipart/form-data",
        )
    assert resp.status_code == 400

def test_screen_end_to_end_with_no_transcription(client):
    with open(REAL_SAMPLE, "rb") as f:
        resp = client.post(
            "/api/v1/screen",
            data={
                "file": (f, "sample.wav"),
                "prompt_text": "then he rang the bell no answer",
                "transcription_mode": "none",
            },
            content_type="multipart/form-data",
        )
    assert resp.status_code == 200
    body = resp.get_json()
    assert "features" in body
    assert "wpm" in body["features"]
    # risk_assessment is either a plain-language result or None with a note
    # explaining why, depending on whether models/risk_model.joblib exists.
    assert "risk_assessment" in body

def test_screen_rejects_non_wav_bytes(client):
    resp = client.post(
        "/api/v1/screen",
        data={
            "file": (io.BytesIO(b"not a real wav file"), "sample.wav"),
            "prompt_text": "hello world",
        },
        content_type="multipart/form-data",
    )
    assert resp.status_code == 422
