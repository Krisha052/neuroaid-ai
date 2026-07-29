"""
Demonstrates a *downstream* LLM -- independent of NeuroAid's own orchestrator
in src/agent/ -- consuming the screening pipeline as a tool via the SDK.

This is NOT the same agent as src/agent/orchestrator.py: that one lives
server-side and IS the screening pipeline's own orchestration. This script
is a separate, external LLM client (as any third-party integration would be)
proving SCREEN_TOOL_SCHEMA + NeuroAidClient work for arbitrary callers.

Usage:
    ANTHROPIC_API_KEY=... python sdk/examples/downstream_llm_demo.py \\
        --audio ../data/sample_audio/real_samples/2277-149896-0033.wav \\
        --prompt "Then he rang the bell. No answer."

Requires the NeuroAid API running locally (see root README) and `anthropic`
installed (`pip install anthropic`).
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuroaid_client import SCREEN_TOOL_SCHEMA, call_screen_tool  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY to run this example.")

    import anthropic

    client = anthropic.Anthropic()
    messages = [{
        "role": "user",
        "content": (
            f"A student read this prompt aloud: {args.prompt!r}. The recording "
            f"is at {args.audio}. Use the neuroaid_screen_reading tool to check "
            "it and summarize the result for a teacher in one sentence."
        ),
    }]

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        tools=[SCREEN_TOOL_SCHEMA],
        messages=messages,
    )

    tool_use = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_use is None:
        print("Model didn't call the tool:", response.content)
        return

    result = call_screen_tool(tool_use.input, base_url=args.base_url)
    print("Tool result:", json.dumps(result, indent=2))

    messages.append({"role": "assistant", "content": response.content})
    messages.append({
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": tool_use.id,
            "content": json.dumps(result),
        }],
    })

    final = client.messages.create(
        model="claude-sonnet-4-5", max_tokens=512,
        tools=[SCREEN_TOOL_SCHEMA], messages=messages,
    )
    print("\nDownstream LLM summary:")
    print(" ".join(b.text for b in final.content if b.type == "text"))

if __name__ == "__main__":
    main()
