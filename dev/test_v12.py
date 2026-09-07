"""v12 mechanics: gpt-oss must still lock fn (no regression); gemma portfolio must probe+lock a firing style."""
import os,sys,time,importlib.util,pathlib
os.environ.setdefault("KAGGLE_IS_COMPETITION_RERUN","1")
ROOT=pathlib.Path(__file__).resolve().parent.parent
spec=importlib.util.spec_from_file_location("a",ROOT/"attack.py");m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
from aicomp_sdk.attacks.contracts import AttackRunConfig
from aicomp_sdk.evaluation.ops import EVALUATION_DEFAULT_MAX_TOOL_HOPS,AttackEvalOptions,eval_attack
from aicomp_sdk.guardrails.optimal import Guardrail as OptimalGuardrail
from aicomp_sdk.scoring import score_attack
def run_model(which):
    if which=="gpt_oss":
        from aicomp_sdk.agents.gpt_oss_agent import GPTOSSAgent,build_gpt_oss_backend
        be=build_gpt_oss_backend(torch_dtype="bfloat16");fac=lambda:GPTOSSAgent(backend=be)
    else:
        os.environ["GEMMA_MODEL_ID"]="unsloth/gemma-3-4b-it"
        from aicomp_sdk.agents.gemma_agent import GemmaAgent,build_gemma_backend
        be=build_gemma_backend(torch_dtype="bfloat16");fac=lambda:GemmaAgent(backend=be)
    opts=AttackEvalOptions(run_config=AttackRunConfig(time_budget_s=float(sys.argv[2]) if len(sys.argv)>2 else 100.0,max_steps=10**9,max_tool_hops=EVALUATION_DEFAULT_MAX_TOOL_HOPS),
        env_seed=123,guardrail_factory=OptimalGuardrail,attack_config={"auto":True})
    t=time.time();f=eval_attack(m.AttackAlgorithm,float(sys.argv[2]) if len(sys.argv)>2 else 100.0,fac,"sandbox",options=opts)
    msgs=f[0].get("user_messages",[""]) if f else [""]
    print(f"[{which}] auto score={score_attack(f):.3f} N={len(f)} ({time.time()-t:.0f}s) lockedMsg={msgs[0][:55]!r}",flush=True)
run_model(sys.argv[1])
os._exit(0)
