import threading
import time
from pathlib import Path

import pytest
from neuroaid_client import NeuroAidAPIError, NeuroAidClient

from src.api.app import create_app

REAL_SAMPLE = (
    Path(__file__).resolve().parents[1]
    / "data" / "sample_audio" / "real_samples" / "2277-149896-0033.wav"
)
TEST_PORT = 8799

@pytest.fixture(scope="module")
def live_server():
    """Runs the real Flask app in a background thread on a real port, so the
    SDK is tested over actual HTTP rather than mocked -- the whole point is
    proving a real client can reliably talk to a real server."""
    app = create_app()
    thread = threading.Thread(
        target=app.run, kwargs={"port": TEST_PORT, "use_reloader": False}, daemon=True,
    )
    thread.start()
    time.sleep(1.0)
    yield f"http://localhost:{TEST_PORT}"

def test_sdk_health(live_server):
    client = NeuroAidClient(base_url=live_server)
    assert client.health()["status"] == "ok"

def test_sdk_prompts_lists_demo_set(live_server):
    client = NeuroAidClient(base_url=live_server)
    prompts = client.prompts()
    assert "LIBRISPEECH_SAMPLE_DEMO" in prompts

def test_sdk_screen_end_to_end(live_server):
    client = NeuroAidClient(base_url=live_server)
    result = client.screen(
        REAL_SAMPLE, prompt_text="then he rang the bell no answer",
        transcription_mode="none",
    )
    assert "wpm" in result.features

def test_sdk_requires_a_prompt(live_server):
    client = NeuroAidClient(base_url=live_server)
    with pytest.raises(ValueError):
        client.screen(REAL_SAMPLE)

def test_sdk_raises_typed_api_error_for_unknown_prompt_set(live_server):
    client = NeuroAidClient(base_url=live_server)
    with pytest.raises(NeuroAidAPIError) as exc_info:
        client.screen(REAL_SAMPLE, prompt_set="NOT_A_REAL_SET")
    assert exc_info.value.status_code == 400
