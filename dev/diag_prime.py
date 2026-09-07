import os,time,pathlib
os.environ.setdefault("KAGGLE_IS_COMPETITION_RERUN","1"); os.environ.setdefault("GEMMA_MODEL_ID","unsloth/gemma-3-4b-it")
import importlib.util,aicomp_sdk
spec=importlib.util.spec_from_file_location("a",pathlib.Path("attack.py"));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
from aicomp_sdk.agents.gemma_agent import GemmaAgent, build_gemma_backend
from aicomp_sdk.core.env.sandbox import SandboxEnv
from aicomp_sdk.core.predicates import eval_predicates
from aicomp_sdk.guardrails.optimal import Guardrail as OptimalGuardrail
FIX=pathlib.Path(aicomp_sdk.__file__).parent/"fixtures"
print("loading...",flush=True); be=build_gemma_backend(torch_dtype="bfloat16")
for label,msgs in [("gprime", m._prime("aaaa")), ("gprime2", m._prime2("aaab"))]:
    env=SandboxEnv(seed=123,fixtures_dir=FIX,agent=GemmaAgent(backend=be),guardrail=OptimalGuardrail(),max_tool_hops=4)
    env.reset()
    for j,mm in enumerate(msgs):
        try: env.interact(mm,max_tool_hops=4)
        except Exception as e: print(f"  turn{j} ERR {e!r}")
    tr=env.export_trace_dict()
    posts=[e for e in tr.get("tool_events",[]) if e["name"]=="http.post" and e.get("ok")]
    preds=[p["predicate"] for p in eval_predicates(tr)]
    asst=[a[:120] for a in tr.get("assistant_messages",[]) if a and a!="(no_tool)"]
    print(f"\n• {label}: posts={len(posts)} preds={preds}")
    for k,a in enumerate(asst): print(f"    asst{k}: {a!r}")
os._exit(0)
