"""Optional live model client: re-ask one image region and score the answer.

This closes the loop the dashboard otherwise only inspects. A supervisor looking
at a silent failure can put the same crop and the same prompt back to the model —
the adapted checkpoint, a different adapter, or a stronger model — and see
whether the point action changes, without leaving the audit view.

It talks to any OpenAI-compatible chat-completions endpoint that accepts image
content, which is what `vllm serve` exposes for Qwen3-VL. Nothing is bundled and
nothing runs locally: if no endpoint is configured the feature stays off, because
a dashboard that quietly substitutes a different model than the one under audit
would be worse than one that cannot re-ask at all.

Serve the checkpoint under audit, for example:

    vllm serve Qwen/Qwen3-VL-8B-Instruct \\
      --enable-lora --lora-modules staged_grpo=/path/to/adapter \\
      --port 8000 --served-model-name staged_grpo

then point the dashboard at http://<host>:8000/v1.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from pathlib import Path

import requests

DEFAULT_TIMEOUT = 120

# Ample for the checkpoints under audit, which answer directly and stop at their
# own EOS, so the budget is never the binding constraint. A *reasoning* model is a
# different matter: it spends the budget on a thinking trace first and can need
# several times this to reach the answer, which is what the override is for.
DEFAULT_MAX_TOKENS = 2048

ENV_BASE_URL = "SAR_MODEL_BASE_URL"
ENV_MODEL = "SAR_MODEL_NAME"
ENV_KEY = "SAR_MODEL_API_KEY"
ENV_MAX_TOKENS = "SAR_MODEL_MAX_TOKENS"


def max_tokens_from_env(default: int = DEFAULT_MAX_TOKENS) -> int:
    """Completion budget per call, overridable for models that think before answering."""
    try:
        value = int(os.environ.get(ENV_MAX_TOKENS, "") or default)
    except ValueError:
        return default
    return value if value > 0 else default

# The two prompts the paired evaluation uses. Kept verbatim in shape so a live
# re-ask asks the same thing the released records were produced by.
GROUNDING_PROMPT = (
    "Identify all mitochondria in this electron microscopy image. "
    'Return one point per mitochondrion as <point x="0.000" y="0.000" alt="mitochondrion"/> '
    "with normalized coordinates in [0, 1]."
)


def answer_prompt(question: str, choices: list[str]) -> str:
    options = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
    return (
        f"{question}\n{options}\n\n"
        "Answer with the single letter of the correct option."
    )


@dataclass
class Endpoint:
    base_url: str = ""
    model: str = ""
    api_key: str = ""
    timeout: int = DEFAULT_TIMEOUT

    @classmethod
    def from_env(cls) -> "Endpoint":
        return cls(
            base_url=os.environ.get(ENV_BASE_URL, "").rstrip("/"),
            model=os.environ.get(ENV_MODEL, ""),
            api_key=os.environ.get(ENV_KEY, ""),
        )

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.model)


@dataclass
class Reply:
    text: str
    ok: bool = True
    error: str = ""
    latency_s: float = 0.0
    raw_model: str = ""


class ModelError(RuntimeError):
    pass


def _data_uri(image_path: Path) -> str:
    suffix = image_path.suffix.lower().lstrip(".") or "png"
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    return f"data:image/{mime};base64,{base64.b64encode(image_path.read_bytes()).decode()}"


def ask(
    endpoint: Endpoint,
    image_path: Path,
    prompt: str,
    *,
    max_tokens: int | None = None,
    temperature: float = 0.0,
) -> Reply:
    """Send one image + prompt. Returns a Reply; never raises for HTTP errors."""
    if not endpoint.configured:
        return Reply("", ok=False, error="No model endpoint configured.")
    if not image_path or not Path(image_path).exists():
        return Reply("", ok=False, error="This record has no image to send.")

    headers = {"Content-Type": "application/json"}
    if endpoint.api_key:
        headers["Authorization"] = f"Bearer {endpoint.api_key}"

    body = {
        "model": endpoint.model,
        "temperature": temperature,
        "max_tokens": max_tokens_from_env() if max_tokens is None else max_tokens,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _data_uri(Path(image_path))}},
                {"type": "text", "text": prompt},
            ],
        }],
    }

    import time
    started = time.monotonic()
    try:
        response = requests.post(
            f"{endpoint.base_url}/chat/completions",
            json=body, headers=headers, timeout=endpoint.timeout,
        )
    except requests.RequestException as exc:
        return Reply("", ok=False, error=f"Could not reach {endpoint.base_url}: {exc}")
    latency = time.monotonic() - started

    if response.status_code != 200:
        detail = response.text[:300]
        return Reply("", ok=False, error=f"HTTP {response.status_code}: {detail}",
                     latency_s=latency)
    try:
        payload = response.json()
        choice = payload["choices"][0]
        text = choice["message"]["content"]
    except (ValueError, KeyError, IndexError) as exc:
        return Reply("", ok=False, error=f"Unexpected response shape: {exc}", latency_s=latency)

    # An empty answer must be reported, not scored. A reasoning model that spends
    # its whole budget on the thinking trace returns HTTP 200 with no content, and
    # scoring that silently would put a fabricated 0.000 beside the released value.
    if not str(text or "").strip():
        reason = str(choice.get("finish_reason") or "")
        detail = (f"the response hit the token limit before answering — the model is "
                  f"probably emitting a reasoning trace; raise {ENV_MAX_TOKENS} "
                  f"(currently {max_tokens_from_env()}) or serve it with thinking disabled"
                  if reason == "length"
                  else f"the endpoint returned an empty message (finish_reason={reason!r})")
        return Reply("", ok=False, error=detail, latency_s=latency)

    return Reply(text=str(text), latency_s=latency, raw_model=str(payload.get("model", "")))


@dataclass
class LiveResult:
    """One live re-ask of a record, scored the same way the release was."""
    answer_text: str = ""
    answer_letter: str = ""
    answer_correct: int | None = None
    points: list = field(default_factory=list)
    obj_recall: float | None = None
    point_f1: float | None = None
    latency_s: float = 0.0
    model_reported: str = ""
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def reask(endpoint: Endpoint, record, image_path: Path, tau: float) -> LiveResult:
    """Re-ask both channels for one record and score them against its labels."""
    from . import scoring

    out = LiveResult()
    choices = list(record["vqa_choices"] or [])

    if choices:
        reply = ask(endpoint, image_path, answer_prompt(record["vqa_question"], choices))
        out.latency_s += reply.latency_s
        out.model_reported = reply.raw_model or out.model_reported
        if reply.ok:
            out.answer_text = reply.text.strip()
            out.answer_letter = scoring.parse_option_letter(reply.text, len(choices))
            expected = str(record["expected_letter"] or "").strip().upper()
            if out.answer_letter and expected:
                out.answer_correct = int(out.answer_letter == expected)
        else:
            out.errors.append(f"answer channel: {reply.error}")
    else:
        out.errors.append("answer channel: this record exports no MCQ options.")

    reply = ask(endpoint, image_path, GROUNDING_PROMPT)
    out.latency_s += reply.latency_s
    out.model_reported = reply.raw_model or out.model_reported
    if reply.ok:
        out.points = scoring.parse_points(reply.text)
        gt = list(record["gt_centroids"] or [])
        if gt or out.points:
            metrics = scoring.point_metrics(out.points, gt)
            out.obj_recall = metrics["obj_recall"]
            out.point_f1 = metrics["point_f1"]
    else:
        out.errors.append(f"action channel: {reply.error}")

    return out
