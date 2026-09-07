import os,time,importlib.util,pathlib
os.environ.setdefault("KAGGLE_IS_COMPETITION_RERUN","1")
spec=importlib.util.spec_from_file_location("a","attack.py");m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
print("loading gpt-oss...",flush=True)
from aicomp_sdk.agents.gpt_oss_agent import GPTOSSAgent,build_gpt_oss_backend
from aicomp_sdk.attacks.contracts import AttackRunConfig
from aicomp_sdk.evaluation.ops import EVALUATION_DEFAULT_MAX_TOOL_HOPS,AttackEvalOptions,eval_attack
from aicomp_sdk.guardrails.optimal import Guardrail as OptimalGuardrail
from aicomp_sdk.scoring import score_attack
be=build_gpt_oss_backend(torch_dtype="bfloat16");fac=lambda:GPTOSSAgent(backend=be)
print("loaded, evaluating (multihop baked, empty config)...",flush=True)
opts=AttackEvalOptions(run_config=AttackRunConfig(time_budget_s=45.0,max_steps=10**9,max_tool_hops=EVALUATION_DEFAULT_MAX_TOOL_HOPS),env_seed=123,guardrail_factory=OptimalGuardrail,attack_config={})
try:
    f=eval_attack(m.AttackAlgorithm,45.0,fac,"sandbox",options=opts)
    tot=sum(sum(1 for e in x.get("trace",{}).get("tool_events",[]) if e["name"]=="http.post" and e.get("ok")) for x in f)
    print(f"RESULT v14 gpt-oss: score={score_attack(f):.3f} findings={len(f)} total_posts={tot}",flush=True)
except Exception as e:
    import traceback; traceback.print_exc()
os._exit(0)
