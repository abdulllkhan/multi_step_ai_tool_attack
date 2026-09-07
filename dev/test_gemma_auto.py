"""Focused: does the ADAPTIVE attack (auto=True) fire on gemma end-to-end via eval_attack?
Short budget to complete fast."""
import os, time, importlib.util, pathlib
os.environ.setdefault("KAGGLE_IS_COMPETITION_RERUN","1")
os.environ.setdefault("GEMMA_MODEL_ID","unsloth/gemma-3-4b-it")
import aicomp_sdk
from aicomp_sdk.agents.gemma_agent import GemmaAgent, build_gemma_backend
from aicomp_sdk.attacks.contracts import AttackRunConfig
from aicomp_sdk.evaluation.ops import EVALUATION_DEFAULT_MAX_TOOL_HOPS, AttackEvalOptions, eval_attack
from aicomp_sdk.guardrails.optimal import Guardrail as OptimalGuardrail
from aicomp_sdk.scoring import score_attack
ROOT=pathlib.Path(__file__).resolve().parent.parent
spec=importlib.util.spec_from_file_location("a",ROOT/"attack.py"); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print("loading gemma mirror...",flush=True); t=time.time()
backend=build_gemma_backend(torch_dtype="bfloat16"); print(f"loaded {time.time()-t:.0f}s",flush=True)
fac=lambda: GemmaAgent(backend=backend)
for budget in (120.0,):
    opts=AttackEvalOptions(run_config=AttackRunConfig(time_budget_s=budget,max_steps=10**9,
        max_tool_hops=EVALUATION_DEFAULT_MAX_TOOL_HOPS), env_seed=123,
        guardrail_factory=OptimalGuardrail, attack_config={"auto":True})
    t=time.time()
    try:
        f=eval_attack(m.AttackAlgorithm, budget, fac, "sandbox", options=opts)
        print(f"GEMMA auto-adaptive budget={budget}: score={score_attack(f):.3f} N={len(f)} ({time.time()-t:.0f}s)",flush=True)
        for x in f[:2]:
            print("  win msg:", [u[:70] for u in x.get('user_messages',[])], "->",[p['predicate'] for p in x.get('predicates',[])])
    except Exception as e:
        print("ERR", repr(e),flush=True)
os._exit(0)
