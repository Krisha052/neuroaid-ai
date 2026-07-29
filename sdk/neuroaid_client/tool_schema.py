"""
Exposes NeuroAid's screening operation as an Anthropic/OpenAI-style tool-use
JSON schema, so any downstream LLM agent -- not just NeuroAid's own
orchestrator (src/agent/) -- can call it as a tool via NeuroAidClient. See
sdk/README.md for a worked example using a second, independent LLM call.
"""
from typing import Any, Dict

from .client import NeuroAidClient

SCREEN_TOOL_SCHEMA: Dict[str, Any] = {
    "name": "neuroaid_screen_reading",
    "description": (
        "Screen a short reading-aloud audio recording against a reference "
        "prompt for early literacy / dyslexia risk signals. Returns a "
        "transcript, engineered features, and (if available) a "
        "plain-language risk assessment. Informational only, not a medical "
        "diagnosis."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "audio_path": {
                "type": "string",
                "description": "Local path to a WAV file.",
            },
            "prompt_text": {
                "type": "string",
                "description": "Reference text the speaker was asked to read aloud.",
            },
        },
        "required": ["audio_path", "prompt_text"],
    },
}

def call_screen_tool(
    tool_input: Dict[str, Any], base_url: str = "http://localhost:8000"
) -> Dict[str, Any]:
    """
    Executes the tool call described by SCREEN_TOOL_SCHEMA. Wire this up as
    the tool executor in any LLM agent's tool-use loop.
    """
    client = NeuroAidClient(base_url=base_url)
    result = client.screen(tool_input["audio_path"], prompt_text=tool_input["prompt_text"])
    return {
        "transcript": result.transcript,
        "features": result.features,
        "risk_assessment": result.risk_assessment.__dict__ if result.risk_assessment else None,
    }
