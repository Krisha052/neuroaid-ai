import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Protocol

from src.config import AGENT_MODEL, ANTHROPIC_API_KEY

from .errors import (
    AgentError,
    AgentNotConfiguredError,
    AgentUpstreamError,
    MaxIterationsExceededError,
)
from .tools import TOOL_EXECUTORS, TOOL_SCHEMAS, ScreeningContext

logger = logging.getLogger("neuroaid.agent")

SYSTEM_PROMPT = (
    "You are the orchestration agent for NeuroAid AI, a dyslexia pre-screening "
    "prototype -- an informational screening signal, not a diagnosis. Given a "
    "reading prompt and an uploaded audio recording, use the available tools "
    "to: transcribe the audio, extract reading-fluency features, and (if the "
    "sample is analyzable) predict a risk assessment. If the transcript is "
    "empty, the recording is degenerate, or a tool reports an error, call "
    "recommend_retake instead of forcing an assessment on bad input. Once you "
    "have enough information, call finalize_assessment with a short, "
    "plain-language, non-alarming explanation suitable for a parent or "
    "teacher."
)

MAX_ITERATIONS = 4

class AnthropicClientProtocol(Protocol):
    """
    The orchestrator only needs an object shaped like the real
    `anthropic.Anthropic` client (i.e. exposing `.messages.create(...)`), so
    production code can pass the real SDK client while tests inject a fake
    with zero real API calls or network access.
    """
    messages: Any

def build_anthropic_client() -> "AnthropicClientProtocol":
    if not ANTHROPIC_API_KEY:
        raise AgentNotConfiguredError("ANTHROPIC_API_KEY is not configured on this server.")
    import anthropic
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def run_agentic_screening(
    client: AnthropicClientProtocol,
    audio_path: Path,
    prompt_text: str,
    transcription_mode: str = "local",
) -> Dict[str, Any]:
    ctx = ScreeningContext(
        audio_path=audio_path, prompt_text=prompt_text,
        transcription_mode=transcription_mode,
    )
    trace: List[Dict[str, Any]] = []
    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": f"Reading prompt: {prompt_text!r}. Begin the screening."}
    ]

    for iteration in range(MAX_ITERATIONS):
        try:
            response = client.messages.create(
                model=AGENT_MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )
        except Exception as exc:
            import anthropic
            if isinstance(exc, anthropic.APIError):
                logger.error("Anthropic API error: %s", exc)
                raise AgentUpstreamError(f"Anthropic API error: {exc}") from exc
            raise

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        messages.append({"role": "assistant", "content": response.content})

        if not tool_use_blocks:
            text = " ".join(b.text for b in response.content if b.type == "text")
            trace.append({"iteration": iteration, "note": "agent returned no tool call",
                          "text": text})
            break

        tool_results = []
        terminal = None
        for block in tool_use_blocks:
            trace.append({"iteration": iteration, "tool": block.name, "input": block.input})

            if block.name == "recommend_retake":
                terminal = {"outcome": "retake_recommended", "reason": block.input.get("reason")}
            elif block.name == "finalize_assessment":
                terminal = {"outcome": "finalized", "explanation": block.input.get("explanation")}
            else:
                executor = TOOL_EXECUTORS.get(block.name)
                if executor is None:
                    raise AgentError(f"Unknown tool requested by agent: {block.name}")
                try:
                    result = executor(ctx)
                except Exception as exc:
                    logger.warning("Tool %s failed: %s", block.name, exc)
                    result = {"error": str(exc)}
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })

        if terminal:
            trace.append({"iteration": iteration, "terminal": terminal})
            return _build_response(ctx, trace, terminal)

        messages.append({"role": "user", "content": tool_results})

    raise MaxIterationsExceededError(
        f"Agent did not reach a final decision within {MAX_ITERATIONS} iterations."
    )

def _build_response(ctx: ScreeningContext, trace: List[Dict[str, Any]],
                     terminal: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "audio_diagnostics": {"duration_sec": ctx.duration_sec},
        "transcript": ctx.transcript,
        "features": ctx.features,
        "notes": ctx.notes,
        "risk_assessment": ctx.risk_assessment,
        "agent_outcome": terminal["outcome"],
        "agent_explanation": terminal.get("explanation") or terminal.get("reason"),
        "agent_trace": trace,
    }
