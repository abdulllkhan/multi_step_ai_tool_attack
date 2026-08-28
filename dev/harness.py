"""Local evaluation harness for developing attack.py.

Wraps the SDK's own evaluator (`aicomp_sdk.evaluation.ops.eval_attack`) so we can
run our AttackAlgorithm against a chosen (agent, guardrail) pair exactly the way
the hosted evaluator does — replay each candidate in a fresh env, recompute
predicates + score-cell signatures, and score with the official `score_attack`.

The real leaderboard runs against gpt-oss / gemma behind `OptimalGuardrail`
(public) and a hidden stricter guardrail (private). Locally we can only run the
`deterministic` stub agent, so we use two guardrails as bookends:
  - allow  : permissive AllowAll -> validates the SEARCH machinery end-to-end
  - optimal: the real public OptimalGuardrail -> realistic public-board difficulty
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import Any

from aicomp_sdk.agents import build_agent_factory
from aicomp_sdk.attacks.baselines.guardrail_allow import Guardrail as AllowAllGuardrail
from aicomp_sdk.attacks.contracts import AttackRunConfig
from aicomp_sdk.evaluation.ops import (
    EVALUATION_DEFAULT_MAX_TOOL_HOPS,
    AttackEvalOptions,
    eval_attack,
    summarize_attack_findings,
)
from aicomp_sdk.guardrails.optimal import Guardrail as OptimalGuardrail

GUARDRAILS: dict[str, type] = {
    "optimal": OptimalGuardrail,  # public leaderboard guardrail
    "allow": AllowAllGuardrail,  # permissive bookend for machinery validation
}


def evaluate(
    attack_cls: type,
    *,
    agent: str = "deterministic",
    guardrail: str = "optimal",
    budget_s: float = 30.0,
    seed: int = 123,
    env_selection: str = "sandbox",
    attack_config: Mapping[str, Any] | None = None,
    max_tool_hops: int = EVALUATION_DEFAULT_MAX_TOOL_HOPS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run `attack_cls` and return (summary, validated_findings).

    summary keys: score (normalized 0-1000), score_raw, findings_count,
    unique_cells (score-driving), unique_canonical_cells.
    """
    if guardrail not in GUARDRAILS:
        raise ValueError(f"guardrail must be one of {sorted(GUARDRAILS)}")

    options = AttackEvalOptions(
        run_config=AttackRunConfig(
            time_budget_s=float(budget_s),
            max_steps=10**9,
            max_tool_hops=max_tool_hops,
        ),
        attack_config=dict(attack_config or {}),
        env_seed=seed,
        guardrail_factory=GUARDRAILS[guardrail],
    )
    findings = eval_attack(
        attack_cls,
        budget_s,
        build_agent_factory(agent),
        env_selection,
        options=options,
    )
    return summarize_attack_findings(findings), findings


def _predicate_histogram(findings: list[dict[str, Any]]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for f in findings:
        for p in f.get("predicates", []):
            name = p.get("predicate", "?")
            hist[name] = hist.get(name, 0) + 1
    return hist


def main() -> None:
    parser = argparse.ArgumentParser(description="Local attack evaluation harness")
    parser.add_argument("--agent", default="deterministic")
    parser.add_argument("--guardrail", default="optimal", choices=sorted(GUARDRAILS))
    parser.add_argument("--budget-s", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    # Import the submission under test lazily so this module has no hard dep on it.
    import importlib.util
    import pathlib

    attack_path = pathlib.Path(__file__).resolve().parent.parent / "attack.py"
    spec = importlib.util.spec_from_file_location("attack_under_test", attack_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    summary, findings = evaluate(
        module.AttackAlgorithm,
        agent=args.agent,
        guardrail=args.guardrail,
        budget_s=args.budget_s,
        seed=args.seed,
    )
    print(f"agent={args.agent} guardrail={args.guardrail} budget={args.budget_s}s")
    print("summary:", summary)
    print("predicates:", _predicate_histogram(findings))


if __name__ == "__main__":
    main()
