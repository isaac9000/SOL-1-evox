"""
Wrapper around skydiscover-run that adds token usage tracking consistent
with SOL-1-BES. Patches the OpenAI client (used by skydiscover for Anthropic
calls) to intercept responses and write token_usage.json into the run output dir.
"""

import json
import os
import sys
import threading
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Token usage tracking (consistent with SOL-1-BES)
# ---------------------------------------------------------------------------

_token_lock = threading.Lock()
_token_totals: dict = {"input_tokens": 0, "output_tokens": 0, "api_calls": 0}
_token_file: Path | None = None


def _flush_tokens() -> None:
    if _token_file is not None:
        _token_file.write_text(json.dumps(_token_totals, indent=2))


def _record(usage) -> None:
    with _token_lock:
        # OpenAI SDK uses prompt_tokens/completion_tokens; map to BES naming
        _token_totals["input_tokens"] += getattr(usage, "prompt_tokens", 0)
        _token_totals["output_tokens"] += getattr(usage, "completion_tokens", 0)
        _token_totals["api_calls"] += 1
        _flush_tokens()


def _patch_openai() -> None:
    import openai

    _orig = openai.resources.chat.completions.Completions.create

    def _patched_create(self, *args, **kwargs):
        response = _orig(self, *args, **kwargs)
        if getattr(response, "usage", None):
            _record(response.usage)
        return response

    openai.resources.chat.completions.Completions.create = _patched_create


def _build_output_dir(search_type: str, initial_program_path: str) -> str:
    problem_name = os.path.basename(os.path.dirname(os.path.abspath(initial_program_path))) or "unknown"
    timestamp = datetime.now().strftime("%m%d_%H%M")
    return os.path.join("outputs", search_type, f"{problem_name}_{timestamp}")


if __name__ == "__main__":
    # Determine output dir using same formula as skydiscover so token_usage.json
    # lands in the same directory as the run artifacts.
    initial_program = "attn_bwd/starting_point.py"
    search_type = "adaevolve"

    # Allow --output override from CLI args (passed through from run_agent.sh)
    output_dir = None
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg in ("--output", "-o") and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
            break

    if output_dir is None:
        output_dir = _build_output_dir(search_type, initial_program)

    os.makedirs(output_dir, exist_ok=True)

    _token_file = Path(output_dir) / "token_usage.json"

    _patch_openai()

    # Hand off to skydiscover CLI
    from skydiscover.cli import main
    sys.exit(main())
