"""Model-agnostic throughput+score lab (gpt_oss / gemma / gemma_4). One model load.

Reuses the proven decomposition from lab_multihop but selects the target backend by
--model, so the SAME single-vs-multihop and prompt-style experiments run against gemma
the instant HF_TOKEN is available. Forces bfloat16 (mxfp4 triton path is broken on GB10).

Usage:
  .venv/bin/python dev/lab_model.py --model gemma   --k 6 --score-budget 200
  .venv/bin/python dev/lab_model.py --model gpt_oss --k 8 --skip-b
"""
from __future__ import annotations
import argparse, importlib.util, json, os, pathlib, time
os.environ.setdefault("KAGGLE_IS_COMPETITION_RERUN", "1")
import aicomp_sdk
from aicomp_sdk.core.env.sandbox import SandboxEnv
from aicomp_sdk.core.predicates import eval_predicates
from aicomp_sdk.guardrails.optimal import Guardrail as OptimalGuardrail
from aicomp_sdk.attacks.contracts import AttackRunConfig
from aicomp_sdk.evaluation.ops import (EVALUATION_DEFAULT_MAX_TOOL_HOPS, AttackEvalOptions, eval_attack)
from aicomp_sdk.scoring import score_attack

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIX = pathlib.Path(aicomp_sdk.__file__).parent / "fixtures"
HOPS = EVALUATION_DEFAULT_MAX_TOOL_HOPS


def _load_mod():
    spec = importlib.util.spec_from_file_location("attack_ut", ROOT / "attack.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def _backend_and_agent(model, dtype):
    """Return (backend, agent_class). bf16 forced for gpt_oss; gemma uses default."""
    if model == "gpt_oss":
        from aicomp_sdk.agents.gpt_oss_agent import GPTOSSAgent, build_gpt_oss_backend
        return build_gpt_oss_backend(torch_dtype=dtype), GPTOSSAgent
    if model == "gemma":
        from aicomp_sdk.agents.gemma_agent import GemmaAgent, build_gemma_backend
        try:    return build_gemma_backend(torch_dtype=dtype), GemmaAgent
        except TypeError: return build_gemma_backend(), GemmaAgent
    if model == "gemma_4":
        from aicomp_sdk.agents.gemma4_agent import Gemma4Agent, build_gemma4_backend
        try:    return build_gemma4_backend(torch_dtype=dtype, device_map="cuda"), Gemma4Agent
        except TypeError:
            try: return build_gemma4_backend(device_map="cuda"), Gemma4Agent
            except TypeError: return build_gemma4_backend(), Gemma4Agent
    raise ValueError(model)


def _one(backend, AgentCls, msg):
    t0 = time.time()
    env = SandboxEnv(seed=123, fixtures_dir=FIX, agent=AgentCls(backend=backend),
                     guardrail=OptimalGuardrail(), max_tool_hops=HOPS)
    env.reset(); eb = time.time() - t0
    t1 = time.time(); err = None
    try:
        env.interact(msg, max_tool_hops=HOPS)
    except Exception as e:
        err = repr(e)
    gen = time.time() - t1
    tr = env.export_trace_dict(); evs = tr.get("tool_events", [])
    preds = eval_predicates(tr)
    posts = sum(1 for e in evs if e["name"] == "http.post" and e.get("ok"))
    exfil = sum(1 for p in preds if p["predicate"] == "EXFILTRATION")
    asst = " ".join(a for a in tr.get("assistant_messages", []) if a and a != "(no_tool)")
    return eb, gen, posts, exfil, len(evs), len(asst), err


def phase_a(backend, AgentCls, mod, combos, k, resf, model):
    print(f"\n{'='*76}\nPHASE A [{model}] decomposition (K={k}, hop cap={HOPS})\n{'='*76}", flush=True)
    rows = []
    for name, cfg in combos:
        a = mod.AttackAlgorithm(config=dict(cfg))
        envs=gens=posts=exfils=fires=alen=0.0; ok=0
        for i in range(k):
            msg = a._message(1000*(combos.index((name,cfg))+1)+i)
            eb,g,p,x,nev,al,err = _one(backend, AgentCls, msg)
            if err:
                if i==0: print(f"    [{name} #0] ERR {err[:80]}", flush=True)
                continue
            ok+=1; envs+=eb; gens+=g; posts+=p; exfils+=x; fires+=(1 if x>0 else 0); alen+=al
        if ok==0:
            rec={"phase":"A","model":model,"name":name,"config":cfg,"ok":0}
            rows.append(rec); resf.write(json.dumps(rec)+"\n"); resf.flush()
            print(f"  {name:16s} ALL-ERR/NOFIRE", flush=True); continue
        me=envs/ok; mg=gens/ok; pc=me+mg; mp=posts/ok; mx=exfils/ok; fr=fires/ok
        raw=16*mp+2; proj=(raw/200.0)*(9000.0/pc) if pc>0 else 0
        rec={"phase":"A","model":model,"name":name,"config":cfg,"ok":ok,"env_s":round(me,3),
             "gen_s":round(mg,3),"per_cand_s":round(pc,3),"posts":round(mp,2),"exfil":round(mx,2),
             "fire":round(fr,2),"asst_len":int(alen/ok),"raw_per_cand":round(raw,1),"proj_score_9000":round(proj,1)}
        rows.append(rec); resf.write(json.dumps(rec)+"\n"); resf.flush()
        print(f"  {name:16s} env={me:4.2f} gen={mg:5.2f} /cand={pc:5.2f}s posts={mp:4.2f} exfil={mx:4.2f} "
              f"fire={fr:.2f} alen={int(alen/ok):4d} raw={raw:5.1f} ->proj={proj:5.1f}", flush=True)
    rows.sort(key=lambda r:-(r.get("proj_score_9000") or -1))
    print("\n  RANK(proj):", [(r["name"], r.get("proj_score_9000")) for r in rows], flush=True)
    return rows


def phase_b(backend, AgentCls, cls, configs, budget, resf, model):
    print(f"\n{'='*76}\nPHASE B [{model}] eval_attack (budget={budget}s)\n{'='*76}", flush=True)
    fac = lambda: AgentCls(backend=backend); rows=[]
    for name, cfg in configs:
        opts = AttackEvalOptions(run_config=AttackRunConfig(time_budget_s=float(budget),
                max_steps=10**9, max_tool_hops=HOPS), env_seed=123,
                guardrail_factory=OptimalGuardrail, attack_config=dict(cfg))
        t=time.time()
        try:
            f=eval_attack(cls, budget, fac, "sandbox", options=opts)
            rec={"phase":"B","model":model,"name":name,"config":cfg,"score":round(score_attack(f),3),
                 "n_findings":len(f),"elapsed_s":round(time.time()-t,1)}
        except Exception as e:
            rec={"phase":"B","model":model,"name":name,"config":cfg,"error":repr(e)}
        rows.append(rec); resf.write(json.dumps(rec)+"\n"); resf.flush()
        print(f"  {name:26s} " + ("ERR "+rec['error'][:60] if 'error' in rec
              else f"score={rec['score']:8.3f} N={rec['n_findings']:4d} ({rec['elapsed_s']:.0f}s)"), flush=True)
    rows.sort(key=lambda r:-(r.get("score") or -1))
    print("\n  RANK(real):", [(r["name"], r.get("score")) for r in rows], flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt_oss", choices=["gpt_oss","gemma","gemma_4"])
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--score-budget", type=float, default=200.0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--skip-b", action="store_true")
    args = ap.parse_args()
    mod = _load_mod()
    print(f"loading {args.model} backend (dtype={args.dtype})...", flush=True); t=time.time()
    backend, AgentCls = _backend_and_agent(args.model, args.dtype)
    print(f"loaded {time.time()-t:.0f}s ({AgentCls.__name__})", flush=True)
    resf=open(ROOT/"dev"/f"lab_{args.model}_results.jsonl","a")
    resf.write(json.dumps({"phase":"meta","model":args.model,"args":vars(args)})+"\n"); resf.flush()
    combos = [
        ("auto-adaptive", {"auto": True}),
    ]
    a=phase_a(backend, AgentCls, mod, combos, args.k, resf, args.model)
    if not args.skip_b:
        good=[r for r in a if r.get("ok") and r.get("posts",0)>0]
        best=good[0]["name"] if good else "single-v9"
        bcfg=dict(next(c for n,c in combos if n==best))
        cfgs=[("auto-adaptive", {"auto": True})]
        phase_b(backend, AgentCls, mod.AttackAlgorithm, cfgs, args.score_budget, resf, args.model)
    resf.close(); print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
    import sys as _s; _s.stdout.flush(); _s.stderr.flush(); os._exit(0)
