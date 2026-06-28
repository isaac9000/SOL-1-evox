"""
Compare advisor vs openevolve vs evox vs BES — SOLExecBench Problem 1 (attn_bwd).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Advisor (SOL-1-advisor, opus-4-6, run 20260628_083819) ───────────────────
# From attn_bwd/runs/20260628_083819_attn_bwd_starting_point/results.tsv
# Token usage: 17,321,452 input + 111,760 output = 17.4M total, 125 LLM calls
ADV_TOKENS = {"input": 17_321_452, "output": 111_760, "api_calls": 125}

adv_raw = [
    (0,  3428.25, "keep"),
    (1,  0.0,     "crash"),
    (2,  0.0,     "crash"),
    (3,  2194.37, "keep"),
    (4,  1175.61, "keep"),
    (5,  689.58,  "keep"),
    (6,  0.0,     "crash"),
    (7,  594.31,  "keep"),
    (8,  0.0,     "crash"),
    (9,  655.98,  "discard"),
    (10, 463.73,  "keep"),
    (11, 427.14,  "keep"),
    (12, 410.51,  "keep"),
    (13, 0.0,     "crash"),
    (14, 0.0,     "crash"),
    (15, 505.83,  "discard"),
    (16, 757.25,  "discard"),
    (17, 0.0,     "crash"),
    (18, 408.12,  "keep"),
    (19, 408.64,  "discard"),
    (20, 1003.11, "discard"),
    (21, 797.74,  "discard"),
    (22, 409.42,  "discard"),
    (23, 5691.07, "discard"),
    (24, 409.36,  "discard"),
    (25, 409.80,  "discard"),
]
adv_iters = [r[0] for r in adv_raw]
adv_times = [r[1] for r in adv_raw]
adv_kinds = [r[2] for r in adv_raw]

# ── EvoX (SOL-1-evox, sonnet-4-6, 25 iters, AdaEvolve) ──────────────────────
# From outputs/adaevolve/attn_bwd_0628_0841/adaevolve_iteration_stats.jsonl
# Only have global_best per iteration (not individual program scores).
# Token usage: 233,822 input + 129,317 output = 363K total, 28 API calls
EVOX_TOKENS = {"input": 233_822, "output": 129_317, "api_calls": 28}

# Global best geomean_us per iteration (tracks only when best improves)
evox_global_best = {
    0:  3437.26,  # baseline starting_point.py
    1:  2201.449,
    2:  2201.449,
    3:  2201.449,
    4:  2134.306,
    5:  1755.282,
    6:  1079.355,
    7:  1070.978,
    8:  1070.978,
    9:  1070.978,
    10: 1070.978,
    11: 1070.978,
    12: 1070.978,
    13: 1070.978,
    14: 1070.978,
    15: 1070.978,
    16: 1070.978,
    17: 1070.978,
    18: 1070.978,
    19: 1070.978,
    20: 1070.978,
    21: 1070.978,
    22: 1057.927,
    23: 1057.927,
    24: 1057.927,
    25: 1048.431,
}

# Extract improvement (keep) events
evox_iters, evox_times, evox_kinds = [], [], []
prev_best = float("inf")
for it in sorted(evox_global_best):
    gm = evox_global_best[it]
    evox_iters.append(it)
    evox_times.append(gm)
    if gm < prev_best:
        evox_kinds.append("keep")
        prev_best = gm
    else:
        evox_kinds.append("discard")  # no per-program data; skip from scatter

# ── BES (SOL-1-BES, 25 gens) ─────────────────────────────────────────────────
# From attn_bwd/results/attn_bwd_bes/gen_*/results/metrics.json
# Token usage: 276,602 input + 122,634 output = 399K total, 63 API calls
BES_TOKENS = {"input": 276_602, "output": 122_634, "api_calls": 63}

bes_raw_scored = [
    (0,  3439.29),
    (1,  3183.992),
    (2,  None),      # crash
    (3,  None),      # crash
    (4,  7869.662),
    (5,  None),      # crash
    (6,  1950.067),
    (7,  2077.879),
    (8,  3017.027),
    (9,  None),      # crash
    (10, 3448.472),
    (11, 1177.998),
    (12, 610.176),
    (13, 3368.553),
    (14, None),      # crash
    (15, None),      # crash
    (16, 573.035),
    (17, 1183.54),
    (18, None),      # crash
    (19, 3373.155),
    (20, 4171.16),
    (21, 573.014),
    (22, 2425.677),
    (23, None),      # crash
    (24, None),      # crash
]
bes_iters, bes_times, bes_kinds = [], [], []
bes_best = float("inf")
for it, gm in bes_raw_scored:
    bes_iters.append(it)
    if gm is None:
        bes_times.append(0.0)
        bes_kinds.append("crash")
    else:
        bes_times.append(gm)
        if gm < bes_best:
            bes_best = gm
            bes_kinds.append("keep")
        else:
            bes_kinds.append("discard")

# ── OpenEvolve (SOL-1-openevolve, sonnet-4-6, 25 iters) ──────────────────────
# From attn_bwd/openevolve_runs/run1/checkpoints/*/programs/*.json
# Token usage: 364,272 input + 46,251 output = 410K total, 25 API calls
# Note: OpenEvolve scores = 1,000,000/geomean_us internally; plot uses geomean_us
OE_TOKENS = {"input": 364_272, "output": 46_251, "api_calls": 25}

oe_raw = [
    (0,  3447.252),  # SEED
    (1,  3546.849),
    (2,  1770.172),
    (3,  1008.933),
    (4,  1752.387),
    (5,  1005.300),
    (6,  2300.618),
    (7,   987.618),
    (8,     0.000),  # crash
    (9,   989.850),
    (10, 1721.238),
    (11,  987.199),
    (12, 1716.781),
    (13,  990.586),
    (14, 1720.768),
    (15,  986.448),
    (16, 2085.185),
    (17,  901.993),
    (18, 2074.985),
    (19,  901.665),
    (20,  916.453),
    (21,  900.979),
    (22,  921.634),
    (23,  899.337),
    (24,  917.706),
    (25,  903.295),
]
oe_iters, oe_times, oe_kinds = [], [], []
oe_best = float("inf")
for it, gm in oe_raw:
    oe_iters.append(it)
    if gm == 0.0:
        oe_times.append(0.0)
        oe_kinds.append("crash")
    else:
        oe_times.append(gm)
        if gm < oe_best:
            oe_best = gm
            oe_kinds.append("keep")
        else:
            oe_kinds.append("discard")

# ── Best-over-time step lines ─────────────────────────────────────────────────
def best_step(iters, times, kinds):
    bx, by = [], []
    best = float("inf")
    for it, t, k in sorted(zip(iters, times, kinds)):
        if k == "keep" and t > 0:
            best = t
        if best < float("inf"):
            bx.append(it)
            by.append(best)
    return bx, by

adv_bx,  adv_by  = best_step(adv_iters, adv_times, adv_kinds)
evox_bx, evox_by = best_step(evox_iters, evox_times, evox_kinds)
bes_bx,  bes_by  = best_step(bes_iters, bes_times, bes_kinds)
oe_bx,   oe_by   = best_step(oe_iters, oe_times, oe_kinds)

adv_best     = min(t for t, k in zip(adv_times, adv_kinds) if k == "keep" and t > 0)
evox_best    = min(evox_by) if evox_by else float("inf")
bes_best_val = min(bes_by) if bes_by else float("inf")
oe_best_val  = min(oe_by) if oe_by else float("inf")

# ── Y-axis (negative latency, clip outliers) ──────────────────────────────────
CLIP_US = 5000.0
all_valid = [
    t for t in adv_times + evox_times + bes_times + oe_times
    if 0 < t <= CLIP_US
]
y_hi = -(min(all_valid) * 0.82)
y_lo = -(CLIP_US * 1.08)

def ny(t):
    return max(-t, y_lo) if t > 0 else y_lo

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 8))
fig.subplots_adjust(top=0.75)

# ── Advisor — green ───────────────────────────────────────────────────────────
adv_kx = [it for it, k in zip(adv_iters, adv_kinds) if k == "keep"]
adv_ky = [ny(adv_times[i]) for i, k in enumerate(adv_kinds) if k == "keep"]
adv_dx = [it for it, k in zip(adv_iters, adv_kinds) if k == "discard"]
adv_dy = [ny(adv_times[i]) for i, k in enumerate(adv_kinds) if k == "discard"]
adv_cx = [it for it, k in zip(adv_iters, adv_kinds) if k == "crash"]

if adv_kx:
    ax.scatter(adv_kx, adv_ky, c="#22c55e", s=70, zorder=5,
               edgecolors="white", linewidths=0.5, label="advisor keep")
if adv_dx:
    ax.scatter(adv_dx, adv_dy, c="#86efac", s=40, zorder=4,
               edgecolors="white", linewidths=0.3, alpha=0.8, label="advisor discard")
if adv_bx:
    ax.step(adv_bx, [-t for t in adv_by], where="post", color="#22c55e",
            linewidth=2, label="advisor best", zorder=6)

# ── EvoX — orange (best-over-time only, no per-program scatter) ───────────────
evox_kx = [it for it, k in zip(evox_iters, evox_kinds) if k == "keep"]
evox_ky = [ny(evox_times[i]) for i, k in enumerate(evox_kinds) if k == "keep"]
if evox_kx:
    ax.scatter(evox_kx, evox_ky, c="#f97316", s=70, zorder=5,
               edgecolors="white", linewidths=0.5, label="evox improvement")
if evox_bx:
    ax.step(evox_bx, [-t for t in evox_by], where="post", color="#f97316",
            linewidth=2, label="evox best", zorder=6)

# ── BES — purple ──────────────────────────────────────────────────────────────
bes_kx = [it for it, k in zip(bes_iters, bes_kinds) if k == "keep"]
bes_ky = [ny(bes_times[i]) for i, k in enumerate(bes_kinds) if k == "keep"]
bes_dx = [it for it, k in zip(bes_iters, bes_kinds) if k == "discard"]
bes_dy = [ny(bes_times[i]) for i, k in enumerate(bes_kinds) if k == "discard"]
bes_cx = [it for it, k in zip(bes_iters, bes_kinds) if k == "crash"]

if bes_kx:
    ax.scatter(bes_kx, bes_ky, c="#a855f7", s=70, zorder=5,
               edgecolors="white", linewidths=0.5, label="BES keep")
if bes_dx:
    ax.scatter(bes_dx, bes_dy, c="#d8b4fe", s=40, zorder=4,
               edgecolors="white", linewidths=0.3, alpha=0.8, label="BES discard")
if bes_bx:
    ax.step(bes_bx, [-t for t in bes_by], where="post", color="#a855f7",
            linewidth=2, label="BES best", zorder=6)

# ── OpenEvolve — blue ────────────────────────────────────────────────────────
oe_kx = [it for it, k in zip(oe_iters, oe_kinds) if k == "keep"]
oe_ky = [ny(oe_times[i]) for i, k in enumerate(oe_kinds) if k == "keep"]
oe_dx = [it for it, k in zip(oe_iters, oe_kinds) if k == "discard"]
oe_dy = [ny(oe_times[i]) for i, k in enumerate(oe_kinds) if k == "discard"]
oe_cx = [it for it, k in zip(oe_iters, oe_kinds) if k == "crash"]

if oe_kx:
    ax.scatter(oe_kx, oe_ky, c="#3b82f6", s=70, zorder=5,
               edgecolors="white", linewidths=0.5, label="openevolve keep")
if oe_dx:
    ax.scatter(oe_dx, oe_dy, c="#93c5fd", s=40, zorder=4,
               edgecolors="white", linewidths=0.3, alpha=0.8, label="openevolve discard")
if oe_bx:
    ax.step(oe_bx, [-t for t in oe_by], where="post", color="#3b82f6",
            linewidth=2, label="openevolve best", zorder=6)

# ── Crashes (advisor + BES + openevolve) ─────────────────────────────────────
all_cx = adv_cx + bes_cx + oe_cx
if all_cx:
    ax.scatter(all_cx, [y_lo] * len(all_cx), c="#fbbf24", s=40, zorder=3,
               marker="x", linewidths=1.5,
               label=f"crash ({len(all_cx)})", alpha=0.8)

ax.set_ylim(y_lo * 1.05, y_hi)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
ax.set_xlabel("Iteration / Generation #", fontsize=12)
ax.set_ylabel("Negative Latency (−μs)", fontsize=12)
ax.grid(True, alpha=0.3)

# ── Legend above the plot ─────────────────────────────────────────────────────
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=5,
          framealpha=0.9, fontsize=10, borderaxespad=0)

# ── Best-time band ────────────────────────────────────────────────────────────
fig.text(
    0.5, 0.92,
    f"Advisor best: {adv_best:.2f} μs    |    "
    f"OpenEvolve best: {oe_best_val:.2f} μs    |    "
    f"BES best: {bes_best_val:.2f} μs    |    "
    f"EvoX best: {evox_best:.2f} μs",
    ha="center", va="top", fontsize=11, fontweight="bold", color="#1e3a5f",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
              edgecolor="#6b7280", alpha=0.9),
)

# ── Title ─────────────────────────────────────────────────────────────────────
fig.text(
    0.5, 0.995,
    "advisor vs openevolve vs evox vs BES — SOLExecBench Problem 1",
    ha="center", va="top", fontsize=14, fontweight="bold",
)

# ── Token usage annotations ───────────────────────────────────────────────────
token_lines = [
    ("advisor",    "#22c55e", ADV_TOKENS,  0.01),
    ("openevolve", "#3b82f6", OE_TOKENS,   0.26),
    ("BES",        "#a855f7", BES_TOKENS,  0.51),
    ("evox",       "#f97316", EVOX_TOKENS, 0.76),
]
for label, color, tok, xfrac in token_lines:
    total = tok["input"] + tok["output"]
    total_disp = f"{total/1_000_000:.1f}M" if total >= 1_000_000 else f"{total/1_000:.0f}K"
    text = (
        f"{label}\n"
        f"{tok['api_calls']} LLM calls\n"
        f"~{total_disp} tokens"
    )
    ax.annotate(
        text,
        xy=(xfrac, 0.02), xycoords="axes fraction",
        ha="left", va="bottom", fontsize=8.5, color=color,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=color, alpha=0.85),
    )

# ── Outlier note ──────────────────────────────────────────────────────────────
ax.annotate(
    f"(outliers > {CLIP_US:.0f} μs shown at floor)",
    xy=(0.5, 0.10), xycoords="axes fraction",
    ha="center", va="bottom", fontsize=9, color="#6b7280",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
              edgecolor="#d1d5db", alpha=0.8),
)

# ── Baseline & SOL reference lines ────────────────────────────────────────────
ax.axhline(-756, color="#9ca3af", linewidth=1.0, linestyle="--", alpha=0.5,
           label="baseline ≈756 μs")
ax.axhline(-82, color="#10b981", linewidth=1.0, linestyle="--", alpha=0.5,
           label="SOL ≈82 μs")

out = "/workspace/SOL-1-evox/comparison.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
