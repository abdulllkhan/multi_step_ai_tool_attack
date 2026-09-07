"""Numerical-robustness gemma probe. Weights == board (unsloth byte-exact to google).
Gemma's bare-JSON output is knife-edge, so GB10-bf16 fires but T4-bf16 (board) fences.
Find prompts that fire across DIFFERENT precisions (bf16 vs fp32) = high-margin = robust =
should survive the board's T4 rounding. Prints per-style fire under each config."""
import os,time,pathlib
os.environ.setdefault("KAGGLE_IS_COMPETITION_RERUN","1"); os.environ.setdefault("GEMMA_MODEL_ID","unsloth/gemma-3-4b-it")
import importlib.util,aicomp_sdk,torch
spec=importlib.util.spec_from_file_location("a",pathlib.Path("attack.py"));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
from aicomp_sdk.agents.gemma_agent import GemmaAgent, build_gemma_backend
from aicomp_sdk.core.env.sandbox import SandboxEnv
from aicomp_sdk.core.predicates import eval_predicates
from aicomp_sdk.guardrails.optimal import Guardrail as OptimalGuardrail
FIX=pathlib.Path(aicomp_sdk.__file__).parent/"fixtures"
STYLES=["gecho","graw","gbenign2","gtelemetry","gpipeline","gcontenttype","gsingle","gmachine"]
CONFIGS=[("bfloat16","high"),("float32","highest"),("bfloat16","medium")]
K=6
def test(be,style):
    f=0
    for i in range(K):
        env=SandboxEnv(seed=123,fixtures_dir=FIX,agent=GemmaAgent(backend=be),guardrail=OptimalGuardrail(),max_tool_hops=4)
        env.reset()
        try: env.interact(m._exfil_msg(7000+i,style),max_tool_hops=4)
        except: pass
        if any(p["predicate"]=="EXFILTRATION" for p in eval_predicates(env.export_trace_dict())): f+=1
    return f/K
for dtype,mmp in CONFIGS:
    torch.set_float32_matmul_precision(mmp)
    print(f"\n#### dtype={dtype} matmul_precision={mmp} ####",flush=True)
    try:
        be=build_gemma_backend(torch_dtype=dtype)
    except Exception as e:
        print("  LOAD ERR",repr(e)[:80]);continue
    for st in STYLES:
        print(f"  {st:12s} fire={test(be,st):.2f}",flush=True)
    del be; import gc; gc.collect(); torch.cuda.empty_cache()
os._exit(0)
