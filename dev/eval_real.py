"""Local evaluation against a REAL target model on the GB10 GPU.

The competition's leaderboard runs our attack against gpt_oss / gemma. This harness
mirrors that locally so we can iterate in minutes instead of ~5h Kaggle reruns.

Key trick: `eval_attack` rebuilds the agent for every candidate replay, which would
reload the 20B model each time. We load the backend ONCE and share it across a
factory, so the weights stay resident on the GPU.

Also sets KAGGLE_IS_COMPETITION_RERUN so attack.py uses the full --budget-s (it caps
to 90s otherwise, which is our visible-commit safety valve).

Usage:
  .venv/bin/python dev/eval_real.py --model gpt_oss --guardrail optimal --budget-s 300
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pathlib
import time

os.environ.setdefault("KAGGLE_IS_COMPETITION_RERUN", "1")  # unlock full budget in attack.py

from aicomp_sdk.attacks.baselines.guardrail_allow import Guardrail as AllowAll
from aicomp_sdk.attacks.contracts import AttackRunConfig
from aicomp_sdk.evaluation.ops import (
    EVALUATION_DEFAULT_MAX_TOOL_HOPS,
    AttackEvalOptions,
    eval_attack,
    summarize_attack_findings,
)
from aicomp_sdk.guardrails.optimal import Guardrail as OptimalGuardrail

GUARDRAILS = {"optimal": OptimalGuardrail, "allow": AllowAll}


def _load_attack_cls():
    path = pathlib.Path(__file__).resolve().parent.parent / "attack.py"
    spec = importlib.util.spec_from_file_location("attack_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AttackAlgorithm


def _shared_agent_factory(model: str):
    """Build the target-model backend once; return a factory that reuses it."""
    if model == "gpt_oss":
        from aicomp_sdk.agents.gpt_oss_agent import GPTOSSAgent, build_gpt_oss_backend

        print(f"Loading {model} backend once (GB10)...", flush=True)
        t = time.time()
        backend = build_gpt_oss_backend()
        print(f"  loaded in {time.time() - t:.0f}s", flush=True)
        return lambda: GPTOSSAgent(backend=backend)
    if model in ("gemma", "gemma_4"):
        from aicomp_sdk.agents.gemma4_agent import Gemma4Agent, build_gemma4_backend
        from aicomp_sdk.agents.gemma_agent import GemmaAgent, build_gemma_backend

        print(f"Loading {model} backend once (GB10)...", flush=True)
        t = time.time()
        if model == "gemma":
            backend = build_gemma_backend()
            print(f"  loaded in {time.time() - t:.0f}s", flush=True)
            return lambda: GemmaAgent(backend=backend)
        backend = build_gemma4_backend()
        print(f"  loaded in {time.time() - t:.0f}s", flush=True)
        return lambda: Gemma4Agent(backend=backend)
    # Fallback: deterministic (no GPU) via the SDK factory.
    from aicomp_sdk.agents import build_agent_factory

    return build_agent_factory(model)


def main() -> None:
    ap = argparse.ArgumentParser(description="Local real-model attack evaluation")
    ap.add_argument("--model", default="gpt_oss")
    ap.add_argument("--guardrail", default="optimal", choices=sorted(GUARDRAILS))
    ap.add_argument("--budget-s", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    agent_factory = _shared_agent_factory(args.model)
    options = AttackEvalOptions(
        run_config=AttackRunConfig(
            time_budget_s=float(args.budget_s),
            max_steps=10**9,
            max_tool_hops=EVALUATION_DEFAULT_MAX_TOOL_HOPS,
        ),
        env_seed=args.seed,
        guardrail_factory=GUARDRAILS[args.guardrail],
    )

    t = time.time()
    findings = eval_attack(_load_attack_cls(), args.budget_s, agent_factory, "sandbox", options=options)
    elapsed = time.time() - t

    summary = summarize_attack_findings(findings)
    hist: dict[str, int] = {}
    for f in findings:
        for p in f.get("predicates", []):
            hist[p["predicate"]] = hist.get(p["predicate"], 0) + 1
    print(f"\n=== {args.model} x {args.guardrail} ({elapsed:.0f}s) ===")
    print("SUMMARY:", summary)
    print("PREDICATES:", hist)
    # Show a few concrete winning chains for inspection.
    for f in findings[:5]:
        print("  WIN:", [m[:60] for m in f.get("user_messages", [])],
              "->", [p["predicate"] for p in f.get("predicates", [])])


if __name__ == "__main__":
    main()
    # Force teardown: the HF backend / eval threads can otherwise linger and hold GPU
    # memory (orphan processes). Flush and hard-exit so the GPU is released immediately.
    import sys as _s
    _s.stdout.flush()
    _s.stderr.flush()
    os._exit(0)
