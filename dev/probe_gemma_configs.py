"""Reproduce the board's gemma behavior locally by varying dtype/attention (T4-like).
Board = google gemma-3-4b on T4 fences JSON (gecho scored 0). Local mirror bf16/GB10 does
NOT fence (gecho 100%). If some config here fences too, it's a board-predictive setup to
tune against. Prints per-style fire rate + a raw output sample under each config."""
import os,time,pathlib,itertools
os.environ.setdefault("KAGGLE_IS_COMPETITION_RERUN","1")
os.environ.setdefault("GEMMA_MODEL_ID","unsloth/gemma-3-4b-it")
import importlib.util
ROOT=pathlib.Path(__file__).resolve().parent.parent
spec=importlib.util.spec_from_file_location("a",ROOT/"attack.py");m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
import aicomp_sdk
from aicomp_sdk.agents.gemma_agent import GemmaAgent, build_gemma_backend
from aicomp_sdk.core.env.sandbox import SandboxEnv
from aicomp_sdk.core.predicates import eval_predicates
from aicomp_sdk.guardrails.optimal import Guardrail as OptimalGuardrail
FIX=pathlib.Path(aicomp_sdk.__file__).parent/"fixtures"
STYLES=["gecho","gbenign2","gcopy","graw","gsingle","gmachine"]
CONFIGS=[("bfloat16","auto"),("float16","eager"),("float16","sdpa"),("bfloat16","eager")]
K=6
def test(backend, style):
    fires=0; sample=""
    for i in range(K):
        env=SandboxEnv(seed=123,fixtures_dir=FIX,agent=GemmaAgent(backend=backend),guardrail=OptimalGuardrail(),max_tool_hops=4)
        env.reset()
        try: env.interact(m._exfil_msg(9000+i, style),max_tool_hops=4)
        except Exception as e: pass
        tr=env.export_trace_dict()
        if any(p["predicate"]=="EXFILTRATION" for p in eval_predicates(tr)): fires+=1
        if i==0:
            a=" ".join(x for x in tr.get("assistant_messages",[]) if x and x!="(no_tool)"); sample=a[:70]
    return fires/K, sample
for dtype,attn in CONFIGS:
    print(f"\n########## config: dtype={dtype} attn={attn} ##########",flush=True)
    try:
        kw={"torch_dtype":dtype}
        if attn!="auto": kw["attn_implementation"]=attn
        t=time.time(); be=build_gemma_backend(**kw); print(f"  loaded {time.time()-t:.0f}s",flush=True)
    except Exception as e:
        print(f"  LOAD ERR {e!r}",flush=True); continue
    for st in STYLES:
        fr,samp=test(be,st)
        print(f"  {st:10s} fire={fr:.2f}  raw0={samp!r}",flush=True)
    del be
    import torch,gc; gc.collect(); torch.cuda.empty_cache()
os._exit(0)
