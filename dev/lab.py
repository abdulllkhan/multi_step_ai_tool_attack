"""Single-load throughput+score lab for the EXFIL attack (creditless, unattended).

Loads gpt-oss ONCE on the GB10, then:
  Phase A (decompose): for each prompt STYLE in attack.py, replay K distinct-host
     candidates and record env_build vs generation time, hop count, fire status, and
     assistant-text length. This isolates the replay-time levers and answers whether
     env-build overhead (fixed) or generation (prompt-tunable) dominates -> i.e. whether
     terser/analysis-suppressing prompts can actually raise N.
  Phase B (score): run the real SDK eval_attack at a fixed budget for a few end-to-end
     configs (v9 live-validate baseline, best-style prebuilt, v9 prebuilt) and print the
     SDK score. score is linear in N so the ranking at this budget proxies the board.

Writes JSONL rows to dev/lab_results.jsonl and a human summary to stdout + dev/lab_summary.txt.
Usage: .venv/bin/python dev/lab.py --k 6 --score-budget 240
"""
from __future__ import annotations

import argparse, importlib.util, json, os, pathlib, time

os.environ.setdefault("KAGGLE_IS_COMPETITION_RERUN", "1")

import aicomp_sdk
from aicomp_sdk.agents.gpt_oss_agent import GPTOSSAgent, build_gpt_oss_backend
from aicomp_sdk.core.env.sandbox import SandboxEnv
from aicomp_sdk.core.predicates import eval_predicates
from aicomp_sdk.guardrails.optimal import Guardrail as OptimalGuardrail
from aicomp_sdk.attacks.contracts import AttackRunConfig
from aicomp_sdk.evaluation.ops import (
    EVALUATION_DEFAULT_MAX_TOOL_HOPS, AttackEvalOptions, eval_attack)
from aicomp_sdk.scoring import score_attack, SEVERITY_W

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURES = pathlib.Path(aicomp_sdk.__file__).parent / "fixtures"
HOPS = EVALUATION_DEFAULT_MAX_TOOL_HOPS


def _load_attack_mod():
    spec = importlib.util.spec_from_file_location("attack_under_test", ROOT / "attack.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _raw_of(preds):
    return sum(SEVERITY_W.get(int(p["severity"]), 1) for p in preds)


def phase_a(backend, mod, styles, k, resf):
    print(f"\n{'='*70}\nPHASE A: per-candidate replay decomposition (K={k}/style, hops={HOPS})\n{'='*70}", flush=True)
    rows = []
    for style in styles:
        env_ts, gen_ts, hops_l, fired_l, asst_len, posts_l = [], [], [], [], [], []
        for i in range(k):
            msg = mod._exfil_msg(1000 * (styles.index(style) + 1) + i, style)  # distinct hosts
            te0 = time.time()
            env = SandboxEnv(seed=123, fixtures_dir=FIXTURES,
                             agent=GPTOSSAgent(backend=backend),
                             guardrail=OptimalGuardrail(), max_tool_hops=HOPS)
            env.reset()
            env_build = time.time() - te0
            tg0 = time.time()
            try:
                env.interact(msg, max_tool_hops=HOPS)
            except Exception as e:
                print(f"    [{style} #{i}] interact ERROR {e!r}", flush=True)
            gen = time.time() - tg0
            tr = env.export_trace_dict()
            evs = tr.get("tool_events", [])
            preds = eval_predicates(tr)
            fired = any(p["predicate"] == "EXFILTRATION" for p in preds)
            posts = [e for e in evs if e["name"] == "http.post" and e.get("ok")]
            asst = " ".join(a for a in tr.get("assistant_messages", []) if a and a != "(no_tool)")
            env_ts.append(env_build); gen_ts.append(gen); hops_l.append(len(evs))
            fired_l.append(fired); asst_len.append(len(asst)); posts_l.append(len(posts))
        n = len(fired_l)
        mean_env = sum(env_ts) / n; mean_gen = sum(gen_ts) / n
        per_cand = mean_env + mean_gen
        fire_rate = sum(fired_l) / n
        # projected findings at a 9000s replay budget, and normalized score (0.09/finding)
        proj_n = fire_rate * 9000.0 / per_cand if per_cand > 0 else 0
        proj_score = 0.09 * proj_n
        rec = {"phase": "A", "style": style, "k": n,
               "mean_env_s": round(mean_env, 3), "mean_gen_s": round(mean_gen, 3),
               "per_cand_s": round(per_cand, 3), "fire_rate": round(fire_rate, 3),
               "mean_hops": round(sum(hops_l) / n, 2), "mean_posts": round(sum(posts_l) / n, 2),
               "mean_asst_len": int(sum(asst_len) / n),
               "proj_N_9000s": int(proj_n), "proj_norm_score": round(proj_score, 2)}
        rows.append(rec); resf.write(json.dumps(rec) + "\n"); resf.flush()
        print(f"  {style:8s} env={mean_env:5.2f}s gen={mean_gen:6.2f}s /cand={per_cand:6.2f}s "
              f"fire={fire_rate:4.2f} hops={rec['mean_hops']:.1f} posts={rec['mean_posts']:.1f} "
              f"asst_len={rec['mean_asst_len']:4d}  ->projN@9000={rec['proj_N_9000s']:4d} "
              f"score~{rec['proj_norm_score']:.1f}", flush=True)
    rows.sort(key=lambda r: -r["proj_norm_score"])
    print("\n  RANK (proj score):", [(r["style"], r["proj_norm_score"], r["fire_rate"]) for r in rows], flush=True)
    return rows


def _factory(backend):
    return lambda: GPTOSSAgent(backend=backend)


def phase_b(backend, attack_cls, configs, budget, resf):
    print(f"\n{'='*70}\nPHASE B: end-to-end SDK eval_attack score (budget={budget}s/config)\n{'='*70}", flush=True)
    agent_factory = _factory(backend)
    rows = []
    for name, cfg in configs:
        opts = AttackEvalOptions(
            run_config=AttackRunConfig(time_budget_s=float(budget), max_steps=10**9,
                                       max_tool_hops=HOPS),
            env_seed=123, guardrail_factory=OptimalGuardrail, attack_config=dict(cfg))
        t = time.time()
        try:
            findings = eval_attack(attack_cls, budget, agent_factory, "sandbox", options=opts)
            score = score_attack(findings)
            secs = time.time() - t
            rec = {"phase": "B", "name": name, "config": cfg, "score": round(score, 3),
                   "n_findings": len(findings), "elapsed_s": round(secs, 1)}
        except Exception as e:
            rec = {"phase": "B", "name": name, "config": cfg, "error": repr(e)}
        rows.append(rec); resf.write(json.dumps(rec) + "\n"); resf.flush()
        if "error" in rec:
            print(f"  {name:22s} ERROR {rec['error'][:80]}", flush=True)
        else:
            print(f"  {name:22s} score={rec['score']:7.3f}  N={rec['n_findings']:4d}  ({rec['elapsed_s']:.0f}s)", flush=True)
    rows.sort(key=lambda r: -(r.get("score") or -1))
    print("\n  RANK (real score):", [(r["name"], r.get("score")) for r in rows], flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--score-budget", type=float, default=240.0)
    ap.add_argument("--skip-b", action="store_true")
    args = ap.parse_args()

    mod = _load_attack_mod()
    styles = list(mod.STYLES.keys())
    print(f"styles: {styles}", flush=True)
    print("Loading gpt-oss backend once (GB10)...", flush=True)
    t = time.time(); backend = build_gpt_oss_backend(); print(f"  loaded in {time.time()-t:.0f}s", flush=True)

    resf = open(ROOT / "dev" / "lab_results.jsonl", "a")
    resf.write(json.dumps({"phase": "meta", "ts": time.time(), "k": args.k, "budget": args.score_budget}) + "\n"); resf.flush()

    a_rows = phase_a(backend, mod, styles, args.k, resf)
    best_style = a_rows[0]["style"] if a_rows else "v9"
    # also keep the best style that has fire_rate == 1.0 (reliability first)
    reliable = [r for r in a_rows if r["fire_rate"] >= 0.999]
    best_reliable = reliable[0]["style"] if reliable else best_style

    if not args.skip_b:
        configs = [
            ("v9-validate(base)", {"prompt_style": "v9", "prebuilt": 0}),
            ("v9-prebuilt2000",  {"prompt_style": "v9", "prebuilt": 2000}),
            (f"{best_reliable}-prebuilt2000", {"prompt_style": best_reliable, "prebuilt": 2000}),
        ]
        # avoid duplicate config if best_reliable == v9
        seen = set(); uniq = []
        for nm, c in configs:
            key = json.dumps(c, sort_keys=True)
            if key not in seen: seen.add(key); uniq.append((nm, c))
        phase_b(backend, mod.AttackAlgorithm, uniq, args.score_budget, resf)

    resf.close()
    print(f"\nDONE. best_style(proj)={best_style} best_reliable={best_reliable}", flush=True)


if __name__ == "__main__":
    main()
    import sys as _s; _s.stdout.flush(); _s.stderr.flush(); os._exit(0)
