#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

echo "=== attn_bwd kernel optimization (SkyDiscover) ==="
echo "Deploying evaluator (no-op if already deployed)..."
uv run modal deploy eval_modal_attn_bwd.py

echo ""
echo "Launching SkyDiscover..."
uv run python run_evox.py \
    attn_bwd/starting_point.py \
    attn_bwd/evaluator.py \
    --config attn_bwd/config.yaml \
    "$@"
