# neuroaid-client

A small, typed Python SDK for the NeuroAid AI Flask REST API, kept in-repo
(not published to PyPI). Two things live here:

1. **`NeuroAidClient`** -- a client with typed results, typed exceptions
   mirroring the API's error types, and automatic retry/backoff on
   transient failures.
2. **`SCREEN_TOOL_SCHEMA` / `call_screen_tool`** -- the screening operation
   exposed as an Anthropic/OpenAI-style tool-use schema, so *any* downstream
   LLM agent (not just NeuroAid's own server-side orchestrator in
   `src/agent/`) can reliably consume it as a callable tool.

## Install

```bash
pip install -e sdk/
```

## Direct use

```python
from neuroaid_client import NeuroAidClient

client = NeuroAidClient(base_url="http://localhost:8000")
result = client.screen(
    "data/sample_audio/real_samples/2277-149896-0033.wav",
    prompt_text="Then he rang the bell. No answer.",
)
print(result.transcript)
print(result.risk_assessment.risk_band if result.risk_assessment else "no model available")
```

## Downstream LLM consumption

`sdk/examples/downstream_llm_demo.py` shows a *second, independent* LLM call
(separate from NeuroAid's own orchestrator) using `SCREEN_TOOL_SCHEMA` to call
the screening pipeline as a tool and summarize the result:

```bash
export ANTHROPIC_API_KEY=...
python sdk/examples/downstream_llm_demo.py \
  --audio data/sample_audio/real_samples/2277-149896-0033.wav \
  --prompt "Then he rang the bell. No answer."
```

Requires the API running locally (`gunicorn --bind=0.0.0.0:8000 wsgi:app`)
and `pip install anthropic`.
