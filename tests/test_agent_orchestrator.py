from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from src.agent.errors import AgentError, MaxIterationsExceededError
from src.agent.orchestrator import run_agentic_screening

REAL_SAMPLE = (
    Path(__file__).resolve().parents[1]
    / "data" / "sample_audio" / "real_samples" / "2277-149896-0033.wav"
)
MATCHING_PROMPT = "then he rang the bell no answer"

@dataclass
class FakeBlock:
    type: str
    text: str = ""
    name: str = ""
    input: Dict[str, Any] = field(default_factory=dict)
    id: str = "tool_1"

class FakeMessages:
    """Stands in for `anthropic.Anthropic().messages`, returning one scripted
    response per call so tests can script exact agent behavior without any
    real API calls."""

    def __init__(self, scripted_responses: List[List[FakeBlock]]):
        self._responses = list(scripted_responses)
        self.call_count = 0

    def create(self, **kwargs) -> Any:
        response = self._responses[self.call_count]
        self.call_count += 1
        return SimpleNamespace(content=response)

class FakeAnthropicClient:
    def __init__(self, scripted_responses: List[List[FakeBlock]]):
        self.messages = FakeMessages(scripted_responses)

def test_full_happy_path_calls_all_tools_and_finalizes():
    scripted = [
        [FakeBlock(type="tool_use", name="transcribe", input={}, id="t1")],
        [FakeBlock(type="tool_use", name="extract_features", input={}, id="t2")],
        [FakeBlock(type="tool_use", name="predict_risk", input={}, id="t3")],
        [FakeBlock(type="tool_use", name="finalize_assessment",
                    input={"explanation": "Reading looked solid overall."}, id="t4")],
    ]
    client = FakeAnthropicClient(scripted)

    result = run_agentic_screening(client, REAL_SAMPLE, MATCHING_PROMPT,
                                    transcription_mode="none")

    assert result["agent_outcome"] == "finalized"
    assert result["agent_explanation"] == "Reading looked solid overall."
    assert "wpm" in result["features"]
    tool_calls = [t["tool"] for t in result["agent_trace"] if "tool" in t]
    assert tool_calls == ["transcribe", "extract_features", "predict_risk", "finalize_assessment"]

def test_agent_recommends_retake_instead_of_forcing_assessment():
    scripted = [
        [FakeBlock(type="tool_use", name="transcribe", input={}, id="t1")],
        [FakeBlock(type="tool_use", name="recommend_retake",
                    input={"reason": "transcript looked unreliable"}, id="t2")],
    ]
    client = FakeAnthropicClient(scripted)

    result = run_agentic_screening(client, REAL_SAMPLE, MATCHING_PROMPT,
                                    transcription_mode="none")

    assert result["agent_outcome"] == "retake_recommended"
    assert result["agent_explanation"] == "transcript looked unreliable"
    # predict_risk was never reached
    assert result["risk_assessment"] is None

def test_tool_failure_is_fed_back_and_agent_can_retake(tmp_path):
    bad_wav = tmp_path / "bad.wav"
    bad_wav.write_bytes(b"not a real wav file")

    scripted = [
        [FakeBlock(type="tool_use", name="transcribe", input={}, id="t1")],
        [FakeBlock(type="tool_use", name="recommend_retake",
                    input={"reason": "audio could not be read"}, id="t2")],
    ]
    client = FakeAnthropicClient(scripted)

    result = run_agentic_screening(client, bad_wav, MATCHING_PROMPT,
                                    transcription_mode="none")

    assert result["agent_outcome"] == "retake_recommended"

def test_agent_raises_after_max_iterations_without_terminal_tool():
    scripted = [
        [FakeBlock(type="tool_use", name="transcribe", input={}, id=f"t{i}")]
        for i in range(10)
    ]
    client = FakeAnthropicClient(scripted)

    with pytest.raises(MaxIterationsExceededError):
        run_agentic_screening(client, REAL_SAMPLE, MATCHING_PROMPT, transcription_mode="none")

def test_agent_raises_if_no_tool_call_ever_made():
    scripted = [[FakeBlock(type="text", text="I'm not sure what to do.")]]
    client = FakeAnthropicClient(scripted)

    with pytest.raises(MaxIterationsExceededError):
        run_agentic_screening(client, REAL_SAMPLE, MATCHING_PROMPT, transcription_mode="none")

def test_agent_errors_on_unknown_tool_name():
    scripted = [[FakeBlock(type="tool_use", name="not_a_real_tool", input={}, id="t1")]]
    client = FakeAnthropicClient(scripted)

    with pytest.raises(AgentError):
        run_agentic_screening(client, REAL_SAMPLE, MATCHING_PROMPT, transcription_mode="none")
