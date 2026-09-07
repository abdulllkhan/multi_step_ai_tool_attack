"""Verify throughput-adaptive locks fn on gpt-oss GGUF and nothink on gemma-4 GGUF (exact board models)."""
import os, sys, pathlib, time, importlib.util
os.environ.setdefault("KAGGLE_IS_COMPETITION_RERUN","1")
ROOT=pathlib.Path(__file__).resolve().parent.parent
spec=importlib.util.spec_from_file_location("a",ROOT/"attack.py");m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
import aicomp_sdk
from aicomp_sdk.agents.hf_chat_template.backends.llama_cpp import LlamaCppChatTemplateBackend
from aicomp_sdk.agents.hf_chat_template.types import HFBackendConfig
from aicomp_sdk.attacks.contracts import AttackRunConfig
from aicomp_sdk.evaluation.ops import EVALUATION_DEFAULT_MAX_TOOL_HOPS, AttackEvalOptions, eval_attack
from aicomp_sdk.guardrails.optimal import Guardrail as OptimalGuardrail
from aicomp_sdk.scoring import score_attack
GG=os.path.expanduser("~/.cache/hf_gguf/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf")
from huggingface_hub import hf_hub_download
def gptoss_factory():
    from aicomp_sdk.agents.gpt_oss_agent import GPTOSSAgent
    gp=hf_hub_download('unsloth/gpt-oss-20b-GGUF','gpt-oss-20b-Q4_K_M.gguf')
    cfg=HFBackendConfig(model_id="openai/gpt-oss-20b",model_path=gp,max_new_tokens=1024)
    be=LlamaCppChatTemplateBackend.from_model_path(model_path=gp,config=cfg,n_ctx=8192,n_gpu_layers=-1,supports_tools=True)
    return lambda: GPTOSSAgent(be)
def gemma4_factory():
    from aicomp_sdk.agents.gemma4_agent import (Gemma4Agent, Gemma4ToolCallParser, _GEMMA4_TOOL_CALL_PATTERN, _parse_gemma4_arguments, _clean_fallback_assistant_text)
    from aicomp_sdk.agents.hf_chat_template.response_parsing import normalize_parsed_response, normalize_tool_arguments
    def _pa(b): return normalize_tool_arguments(b) if b.lstrip().startswith("{") else _parse_gemma4_arguments(b)
    class KP(Gemma4ToolCallParser):
        def parse(self,response,*,fallback_call_id):
            tcs=[{"type":"function","function":{"name":x.group("name"),"arguments":_pa(x.group("arguments"))}} for x in _GEMMA4_TOOL_CALL_PATTERN.finditer(response.raw_text)]
            if not tcs: return super().parse(response,fallback_call_id=fallback_call_id)
            return normalize_parsed_response({"role":"assistant","content":_clean_fallback_assistant_text(response.text),"tool_calls":tcs},fallback_call_id=fallback_call_id)
    cfg=HFBackendConfig(model_id="google/gemma-4-26B-A4B-it",model_path=GG,max_new_tokens=1024)
    be=LlamaCppChatTemplateBackend.from_model_path(model_path=GG,config=cfg,n_ctx=8192,n_gpu_layers=-1,supports_tools=True)
    return lambda: Gemma4Agent(be,parser=KP())
which=sys.argv[1]; budget=float(sys.argv[2]) if len(sys.argv)>2 else 60.0
fac=(gptoss_factory if which=="gpt_oss" else gemma4_factory)()
opts=AttackEvalOptions(run_config=AttackRunConfig(time_budget_s=budget,max_steps=10**9,max_tool_hops=EVALUATION_DEFAULT_MAX_TOOL_HOPS),env_seed=123,guardrail_factory=OptimalGuardrail,attack_config={})
t=time.time(); f=eval_attack(m.AttackAlgorithm,budget,fac,"sandbox",options=opts)
# count locked style from returned candidate messages
nfn=sum(1 for x in f if any("Do not think" not in u and "Call http.post" in u for u in x.get("user_messages",[])))
nnt=sum(1 for x in f if any("Do not think" in u for u in x.get("user_messages",[])))
print(f"[{which}] score={score_attack(f):.3f} N={len(f)} fn_cands={nfn} nothink_cands={nnt} ({time.time()-t:.0f}s) -> LOCKED={'nothink' if nnt>nfn else 'fn'}",flush=True)
os._exit(0)
