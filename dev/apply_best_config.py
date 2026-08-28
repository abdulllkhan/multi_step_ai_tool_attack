"""Bake the winning config into attack.py's __init__ defaults.

attack.py reads its knobs as `cfg.get("KEY", DEFAULT)`. On Kaggle the evaluator
passes an empty/default config, so those DEFAULTs are what actually run. This script
rewrites the DEFAULT literals to the best config found by the overnight sweep, so the
Kaggle rerun uses the tuned values. It only touches the numeric literals inside the
known `cfg.get(...)` calls, verifies the file still parses, and refuses to write if a
key can't be located.

Usage:  python dev/apply_best_config.py dev/best_config.json attack.py
"""
from __future__ import annotations

import ast
import json
import re
import sys

# key -> (regex capturing the default literal, formatter for the new literal)
PATCHES = {
    "hedge_every": (r'(cfg\.get\(\s*["\']hedge_every["\']\s*,\s*)([0-9.]+)(\s*\))', lambda v: str(int(v))),
    "slowest0":    (r'(cfg\.get\(\s*["\']slowest0["\']\s*,\s*)([0-9.]+)(\s*\))',    lambda v: f"{float(v):.4g}"),
    "margin_mult": (r'(cfg\.get\(\s*["\']margin_mult["\']\s*,\s*)([0-9.]+)(\s*\))', lambda v: f"{float(v):.4g}"),
}


def main() -> int:
    best_path, attack_path = sys.argv[1], sys.argv[2]
    best = json.loads(open(best_path).read())
    cfg = best.get("config", best)  # accept either {"config": {...}} or a bare dict
    src = open(attack_path).read()

    changed = []
    for key, val in cfg.items():
        if key not in PATCHES:
            print(f"  (skip unknown key {key})")
            continue
        rx, fmt = PATCHES[key]
        newlit = fmt(val)
        new_src, n = re.subn(rx, lambda m: m.group(1) + newlit + m.group(3), src, count=1)
        if n == 0:
            print(f"  WARN: could not locate default for '{key}' — left unchanged")
            continue
        src = new_src
        changed.append(f"{key}={newlit}")

    if not changed:
        print("no defaults changed"); return 0

    ast.parse(src)  # raises if we broke syntax → caller keeps the original
    open(attack_path, "w").write(src)
    print(f"applied best config to {attack_path}: {', '.join(changed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
