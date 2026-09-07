"""Unattended overnight optimizer for attack.py.

Loads the target model (gpt_oss / gemma) ONCE onto the GB10, then scores a sweep of
attack-config variants in-process using the SDK's real scoring function
(aicomp_sdk.scoring.score_attack). Keeps the best config, writes it to
dev/best_config.json, and logs a ranked table. Does NOT submit anything.

The variants are pure config dicts fed through AttackEvalOptions.attack_config, so
attack.py is never rewritten mid-run — the current file's AttackAlgorithm just reads
the knobs. New strategies added to attack.py (selectable via a config key) are swept
the same way.

Usage:
  .venv/bin/python dev/overnight_optimize.py --model gpt_oss --guardrail optimal \
      --budget-s 150 --variants dev/variants.json --deadline-min 420
"""
from __future__ import annotations

import argparse
import importlib.util
import json
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
from aicomp_sdk.scoring import score_attack

ROOT = pathlib.Path(__file__).resolve().parent.parent
GUARDRAILS = {"optimal": OptimalGuardrail, "allow": AllowAll}


def _load_attack_cls():
    path = ROOT / "attack.py"
    spec = importlib.util.spec_from_file_location("attack_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AttackAlgorithm


def _shared_agent_factory(model: str):
    if model == "gpt_oss":
        from aicomp_sdk.agents.gpt_oss_agent import GPTOSSAgent, build_gpt_oss_backend
        print(f"Loading {model} backend once (GB10)...", flush=True)
        t = time.time(); backend = build_gpt_oss_backend(torch_dtype="bfloat16")  # avoid broken mxfp4 triton path on GB10
        print(f"  loaded in {time.time()-t:.0f}s", flush=True)
        return lambda: GPTOSSAgent(backend=backend)
    if model in ("gemma", "gemma_4"):
        from aicomp_sdk.agents.gemma4_agent import Gemma4Agent, build_gemma4_backend
        from aicomp_sdk.agents.gemma_agent import GemmaAgent, build_gemma_backend
        print(f"Loading {model} backend once (GB10)...", flush=True)
        t = time.time()
        if model == "gemma":
            backend = build_gemma_backend()
            print(f"  loaded in {time.time()-t:.0f}s", flush=True)
            return lambda: GemmaAgent(backend=backend)
        backend = build_gemma4_backend()
        print(f"  loaded in {time.time()-t:.0f}s", flush=True)
        return lambda: Gemma4Agent(backend=backend)
    from aicomp_sdk.agents import build_agent_factory
    return build_agent_factory(model)


def _score_one(attack_cls, cfg, budget_s, agent_factory, guardrail, seed):
    options = AttackEvalOptions(
        run_config=AttackRunConfig(
            time_budget_s=float(budget_s), max_steps=10**9,
            max_tool_hops=EVALUATION_DEFAULT_MAX_TOOL_HOPS,
        ),
        env_seed=seed,
        guardrail_factory=guardrail,
        attack_config=dict(cfg),
    )
    t = time.time()
    findings = eval_attack(attack_cls, budget_s, agent_factory, "sandbox", options=options)
    score = score_attack(findings)
    summ = summarize_attack_findings(findings)
    return score, len(findings), summ, time.time() - t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt_oss")
    ap.add_argument("--guardrail", default="optimal", choices=sorted(GUARDRAILS))
    ap.add_argument("--budget-s", type=float, default=150.0, help="per-variant attack budget")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--variants", default=str(ROOT / "dev" / "variants.json"))
    ap.add_argument("--deadline-min", type=float, default=420.0, help="stop starting variants after N minutes")
    ap.add_argument("--out", default=str(ROOT / "dev" / "best_config.json"))
    ap.add_argument("--results", default=str(ROOT / "dev" / "optimize_results.jsonl"))
    args = ap.parse_args()

    variants = json.loads(pathlib.Path(args.variants).read_text())
    print(f"loaded {len(variants)} variant(s) from {args.variants}", flush=True)

    attack_cls = _load_attack_cls()
    agent_factory = _shared_agent_factory(args.model)
    guardrail = GUARDRAILS[args.guardrail]

    start = time.time()
    deadline = start + args.deadline_min * 60.0
    results = []
    resfile = open(args.results, "a")
    best = None
    for idx, v in enumerate(variants):
        if time.time() >= deadline:
            print(f"[deadline] stopping after {idx} variants", flush=True); break
        name = v.get("name", f"variant{idx}")
        cfg = v.get("config", {})
        try:
            score, n, summ, secs = _score_one(attack_cls, cfg, args.budget_s, agent_factory, guardrail, args.seed)
        except Exception as e:  # never let one variant kill the sweep
            print(f"[{idx+1}/{len(variants)}] {name}: ERROR {e!r}", flush=True)
            resfile.write(json.dumps({"name": name, "config": cfg, "error": repr(e)}) + "\n"); resfile.flush()
            continue
        rec = {"name": name, "config": cfg, "score": score, "n_findings": n, "elapsed_s": round(secs, 1), "summary": summ}
        results.append(rec)
        resfile.write(json.dumps(rec) + "\n"); resfile.flush()
        elapsed_min = (time.time() - start) / 60.0
        print(f"[{idx+1}/{len(variants)}] {name}: score={score:.3f}  findings={n}  ({secs:.0f}s)  [{elapsed_min:.0f}min elapsed]", flush=True)
        if best is None or score > best["score"]:
            best = rec
            print(f"    ^ NEW BEST: {name} @ {score:.3f}", flush=True)

    resfile.close()
    print("\n" + "=" * 60)
    if best:
        print(f"BEST: {best['name']}  score={best['score']:.3f}  findings={best['n_findings']}")
        print(f"CONFIG: {json.dumps(best['config'])}")
        pathlib.Path(args.out).write_text(json.dumps(best, indent=2))
        print(f"wrote {args.out}")
    else:
        print("no successful variants scored")
    print("Ranked results:")
    for r in sorted(results, key=lambda x: -x["score"]):
        print(f"  {r['score']:8.3f}  {r['name']}")


if __name__ == "__main__":
    main()
