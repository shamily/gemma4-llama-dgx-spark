#!/usr/bin/env python3
"""
Benchmark thinking vs no-thinking mode for Gemma 4 via the llama.cpp server API.

Requires the server to be running with a thinking-capable model (26B or 31B).
Start it with: docker compose up server

Usage:
  python3 scripts/run_bench_thinking.py
  HOST=localhost PORT=8080 REPS=5 python3 scripts/run_bench_thinking.py

Output: markdown table saved to results/bench_thinking_<timestamp>.md
"""

import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HOST = os.environ.get("HOST", "localhost")
PORT = os.environ.get("PORT", "8080")
BASE_URL = f"http://{HOST}:{PORT}"
REPS = int(os.environ.get("REPS", "5"))
RESULTS_DIR = Path("results")

# Five prompts that span reasoning depth: math, logic, factual, causal, creative.
# Kept short so generation length stays comparable between thinking on/off.
PROMPTS = [
    ("math",     "What is 347 × 28? Give only the numeric answer."),
    ("logic",    "A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball. "
                 "How much does the ball cost in cents?"),
    ("factual",  "What is the capital of Australia?"),
    ("causal",   "If all bloops are razzles and all razzles are lazzles, are all bloops lazzles? "
                 "Answer yes or no and explain in one sentence."),
    ("creative", "Write a haiku about inference speed."),
]


def wait_for_server(timeout: int = 60) -> None:
    print(f"Waiting for server at {BASE_URL}/health ...", flush=True)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE_URL}/health", timeout=2):
                print("  Server is up.")
                return
        except Exception:
            time.sleep(2)
    print("ERROR: server did not become healthy in time.", file=sys.stderr)
    sys.exit(1)


def query(prompt_text: str, thinking: bool) -> dict:
    payload: dict = {
        "model": "gemma-4",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": prompt_text},
        ],
        "max_tokens": 1024,
        "temperature": 0.0,
    }
    if not thinking:
        # thinking_budget=0 suppresses chain-of-thought for Gemma 4 on llama.cpp.
        # Requires llama.cpp built after the thinking-budget feature was merged.
        payload["thinking_budget"] = 0

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {body}") from e
    elapsed = time.monotonic() - t0

    msg = result["choices"][0]["message"]
    usage = result.get("usage", {})
    timings = result.get("timings", {})

    thinking_text = msg.get("reasoning_content") or ""
    content_text = msg.get("content") or ""

    return {
        "elapsed_s": elapsed,
        "thinking_chars": len(thinking_text),
        "content_chars": len(content_text),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "gen_tps": timings.get("predicted_per_second", 0.0),
        "has_thinking": bool(thinking_text),
    }


def run_suite(thinking: bool) -> list[dict]:
    mode = "ON " if thinking else "OFF"
    print(f"\n  Running thinking={mode} — {len(PROMPTS)} prompts × {REPS} reps", flush=True)
    results = []
    for i, (label, prompt) in enumerate(PROMPTS):
        for r in range(REPS):
            print(f"    [{label}] rep {r + 1}/{REPS} ...", end=" ", flush=True)
            row = query(prompt, thinking=thinking)
            row["label"] = label
            row["thinking_mode"] = thinking
            results.append(row)
            print(f"{row['elapsed_s']:.1f}s  {row['gen_tps']:.1f} t/s"
                  f"  think={'yes' if row['has_thinking'] else 'no '}"
                  f"  tokens={row['completion_tokens']}")
    return results


def stats(values: list[float]) -> tuple[float, float]:
    """Return (mean, stdev). Stdev is 0 if fewer than 2 samples."""
    if not values:
        return 0.0, 0.0
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return m, s


def summarise(rows: list[dict]) -> dict:
    elapsed   = [r["elapsed_s"] for r in rows]
    gen_tps   = [r["gen_tps"] for r in rows if r["gen_tps"] > 0]
    comp_tok  = [r["completion_tokens"] for r in rows]
    think_ch  = [r["thinking_chars"] for r in rows]
    has_think = sum(1 for r in rows if r["has_thinking"])

    return {
        "n": len(rows),
        "thinking_active": has_think,
        "elapsed_mean": stats(elapsed)[0],
        "elapsed_std":  stats(elapsed)[1],
        "gen_tps_mean": stats(gen_tps)[0],
        "gen_tps_std":  stats(gen_tps)[1],
        "comp_tokens_mean": stats(comp_tok)[0],
        "think_chars_mean": stats(think_ch)[0],
    }


def fetch_model_id() -> str:
    try:
        with urllib.request.urlopen(f"{BASE_URL}/v1/models", timeout=5) as r:
            data = json.loads(r.read())
        return data["data"][0]["id"]
    except Exception:
        return "unknown"


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"bench_thinking_{timestamp}.md"

    wait_for_server()
    model_id = fetch_model_id()
    print(f"  Model: {model_id}")

    rows_on  = run_suite(thinking=True)
    rows_off = run_suite(thinking=False)

    on  = summarise(rows_on)
    off = summarise(rows_off)

    # -- verify no-thinking actually worked ----------------------------------
    if off["thinking_active"] > 0:
        print(f"\n  WARNING: {off['thinking_active']}/{off['n']} responses still had "
              f"reasoning_content with thinking_budget=0. "
              f"This llama.cpp build may not support thinking_budget. "
              f"Results reflect actual behaviour.", file=sys.stderr)

    # -- build report --------------------------------------------------------
    lines = [
        f"# Thinking vs No-Thinking Benchmark — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        f"**Model:** {model_id}",
        f"**Server:** {HOST}:{PORT}",
        f"**Prompts:** {len(PROMPTS)} × {REPS} repetitions = {len(PROMPTS) * REPS} samples per mode",
        "",
        "## Summary",
        "",
        "| Mode | Latency mean | Latency std | Gen speed | Completion tokens | Thinking chars |",
        "|------|-------------|-------------|-----------|-------------------|----------------|",
        f"| thinking ON  | {on['elapsed_mean']:.2f} s | ±{on['elapsed_std']:.2f} s "
        f"| {on['gen_tps_mean']:.1f} ± {on['gen_tps_std']:.1f} t/s "
        f"| {on['comp_tokens_mean']:.0f} "
        f"| {on['think_chars_mean']:.0f} |",
        f"| thinking OFF | {off['elapsed_mean']:.2f} s | ±{off['elapsed_std']:.2f} s "
        f"| {off['gen_tps_mean']:.1f} ± {off['gen_tps_std']:.1f} t/s "
        f"| {off['comp_tokens_mean']:.0f} "
        f"| {off['think_chars_mean']:.0f} |",
        "",
    ]

    # latency and throughput deltas
    if on["elapsed_mean"] > 0 and off["elapsed_mean"] > 0:
        latency_ratio = on["elapsed_mean"] / off["elapsed_mean"]
        lines += [
            "## Analysis",
            "",
            f"- Thinking ON is **{latency_ratio:.1f}×** slower end-to-end than thinking OFF.",
        ]
        if off["thinking_active"] == 0:
            lines.append("- Thinking was successfully suppressed (`thinking_budget=0` honoured).")
        else:
            lines.append(
                f"- **Note:** `thinking_budget=0` was NOT fully honoured "
                f"({off['thinking_active']}/{off['n']} responses still had reasoning content). "
                f"Upgrade llama.cpp for full thinking suppression."
            )
        lines.append("")

    # per-prompt detail
    lines += [
        "## Per-prompt detail",
        "",
        "| Prompt | Mode | Latency (s) | Gen t/s | Completion tokens | Has thinking |",
        "|--------|------|------------|---------|-------------------|--------------|",
    ]
    for label, _ in PROMPTS:
        for thinking, all_rows in [(True, rows_on), (False, rows_off)]:
            subset = [r for r in all_rows if r["label"] == label]
            lats = [r["elapsed_s"] for r in subset]
            tps  = [r["gen_tps"] for r in subset if r["gen_tps"] > 0]
            comp = [r["completion_tokens"] for r in subset]
            ht   = sum(1 for r in subset if r["has_thinking"])
            mode_label = "ON " if thinking else "OFF"
            lines.append(
                f"| {label} | {mode_label} "
                f"| {stats(lats)[0]:.2f} ± {stats(lats)[1]:.2f} "
                f"| {stats(tps)[0]:.1f} ± {stats(tps)[1]:.1f} "
                f"| {stats(comp)[0]:.0f} "
                f"| {ht}/{len(subset)} |"
            )

    report = "\n".join(lines) + "\n"
    out_path.write_text(report)

    print(f"\n==> Results saved to {out_path}")
    print("\n" + report)


if __name__ == "__main__":
    main()
