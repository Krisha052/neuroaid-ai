import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests

_KNOWN_RISK_FIELDS = {
    "risk_score", "risk_band", "summary", "disclaimer", "contributing_factors", "notes",
}

class NeuroAidClientError(Exception):
    """Base class for SDK-level errors (network failures, exhausted retries)."""

class NeuroAidAPIError(NeuroAidClientError):
    """The API responded with a typed error (see src/api/errors.py)."""

    def __init__(self, status_code: int, error_type: str, message: str):
        super().__init__(f"[{status_code} {error_type}] {message}")
        self.status_code = status_code
        self.error_type = error_type

@dataclass
class RiskAssessment:
    risk_score: float
    risk_band: str
    summary: str
    disclaimer: str
    contributing_factors: Optional[List[str]] = None
    notes: Optional[str] = None

@dataclass
class ScreeningResult:
    transcript: str
    features: Dict[str, float]
    notes: Dict[str, Any]
    risk_assessment: Optional[RiskAssessment]
    raw: Dict[str, Any]

class NeuroAidClient:
    """
    Typed client for the NeuroAid AI Flask REST API. Designed so any
    downstream application -- including LLM agents via `tool_schema.py` --
    can consume speech-based screening reliably: typed exceptions mirroring
    the API's error types, and automatic retry with backoff on transient
    (network / 5xx) failures.
    """

    def __init__(self, base_url: str = "http://localhost:8000",
                 timeout: float = 120.0, max_retries: int = 2, backoff_sec: float = 1.5):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_sec = backoff_sec

    def health(self) -> Dict[str, Any]:
        return self._request("GET", "/api/v1/health")

    def prompts(self) -> Dict[str, str]:
        return self._request("GET", "/api/v1/prompts")

    def screen(
        self,
        audio_path: Union[str, Path],
        *,
        prompt_set: Optional[str] = None,
        prompt_text: Optional[str] = None,
        transcription_mode: str = "local",
        agentic: bool = False,
    ) -> ScreeningResult:
        if not prompt_set and not prompt_text:
            raise ValueError("Provide prompt_set or prompt_text.")

        path = "/api/v1/agent/screen" if agentic else "/api/v1/screen"
        data = {"transcription_mode": transcription_mode}
        if prompt_set:
            data["prompt_set"] = prompt_set
        if prompt_text:
            data["prompt_text"] = prompt_text

        with open(audio_path, "rb") as f:
            raw = self._request(
                "POST", path, data=data,
                files={"file": (Path(audio_path).name, f)},
            )

        return self._to_result(raw)

    def _to_result(self, raw: Dict[str, Any]) -> ScreeningResult:
        risk = raw.get("risk_assessment")
        risk_assessment = None
        if risk:
            filtered = {k: v for k, v in risk.items() if k in _KNOWN_RISK_FIELDS}
            risk_assessment = RiskAssessment(**filtered)

        return ScreeningResult(
            transcript=raw.get("transcript", ""),
            features=raw.get("features", {}),
            notes=raw.get("notes", {}),
            risk_assessment=risk_assessment,
            raw=raw,
        )

    def _request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.request(
                    method, f"{self.base_url}{path}", timeout=self.timeout, **kwargs
                )
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(self.backoff_sec * (attempt + 1))
                    continue
                raise NeuroAidClientError(f"Network error calling {path}: {exc}") from exc

            if resp.ok:
                return resp.json()

            if resp.status_code >= 500 and attempt < self.max_retries:
                time.sleep(self.backoff_sec * (attempt + 1))
                continue

            content_type = resp.headers.get("content-type", "")
            body = resp.json() if content_type.startswith("application/json") else {}
            raise NeuroAidAPIError(
                resp.status_code, body.get("type", "UnknownError"),
                body.get("error", resp.text),
            )

        raise NeuroAidClientError(f"Exhausted retries calling {path}") from last_exc
