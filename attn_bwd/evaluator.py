"""
SkyDiscover evaluator for the attn_bwd kernel optimization problem.

Submits the candidate kernel to the deployed Modal B200 evaluator and
returns a score for SkyDiscover's search loop.

Score = 756 / geomean_us  (~1.0 at baseline, ~9.3 at SOL target).
Returns combined_score=0.0 on correctness failure or crash.

Deploy the Modal evaluator once before running:
    uv run modal deploy eval_modal_attn_bwd.py
"""

import json
import os
import re
import subprocess
import sys
import tempfile

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_EVAL_SCRIPT = os.path.join(_SCRIPT_DIR, "run_eval.py")
BASELINE_GEOMEAN_US = 756.0


def evaluate(program_path: str) -> dict:
    """Evaluate a candidate kernel file. Returns SkyDiscover metrics dict."""
    with tempfile.NamedTemporaryFile(
        suffix=".json", delete=False, dir=_SCRIPT_DIR
    ) as f:
        results_path = f.name

    try:
        ret = subprocess.run(
            [
                sys.executable,
                _EVAL_SCRIPT,
                os.path.abspath(program_path),
                "-o", results_path,
            ],
            cwd=_SCRIPT_DIR,
            timeout=660,
            capture_output=True,
            text=True,
        )

        try:
            with open(results_path) as f:
                md = json.load(f)
        except Exception as e:
            return {"combined_score": 0.0, "error": f"Could not read results: {e}"}

        text = md if isinstance(md, str) else ""

        if "> ❌ Testing failed" in text or "> ❌ Benchmarking failed" in text:
            err_m = re.search(r"## Error:\s*```\s*(.*?)\s*```", text, re.DOTALL)
            detail = err_m.group(1).strip()[:400] if err_m else "correctness failure"
            return {"combined_score": 0.0, "error": detail, "geomean_us": 0.0}

        m = re.search(r"Geometric mean: ⏱ ([\d.]+)", text)
        if m and ret.returncode == 0:
            geomean_us = float(m.group(1))
            return {
                "combined_score": BASELINE_GEOMEAN_US / geomean_us,
                "geomean_us": geomean_us,
            }

        err_m = re.search(r"## Error:\s*```\s*(.*?)\s*```", text, re.DOTALL)
        error = (
            err_m.group(1).strip()[:500] if err_m
            else f"eval exited {ret.returncode}: {ret.stderr[:200]}"
        )
        return {"combined_score": 0.0, "error": error, "geomean_us": 0.0}

    except subprocess.TimeoutExpired:
        return {
            "combined_score": 0.0,
            "error": "Evaluation timed out after 660s",
            "geomean_us": 0.0,
        }
    except Exception as e:
        return {"combined_score": 0.0, "error": str(e), "geomean_us": 0.0}
    finally:
        try:
            os.unlink(results_path)
        except OSError:
            pass
