"""Bake a prompt_style (and optional k_mode) as the DEFAULT in attack.py __init__.

Kaggle passes an empty config, so the __init__ DEFAULTS are what run in the scored
rerun. This flips `cfg.get("prompt_style", "<old>")` -> "<new>" (and optionally k_mode),
verifies the file still parses AND that the default AttackAlgorithm now emits the new
style, then rebuilds the submission notebook.

Usage: python dev/bake_style.py fn            # set prompt_style default to fn
       python dev/bake_style.py fn --check    # dry-run, don't write
"""
from __future__ import annotations
import ast, importlib.util, pathlib, re, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
AP = ROOT / "attack.py"


def _load_default_msg(path):
    spec = importlib.util.spec_from_file_location("a_check", path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.AttackAlgorithm(config={})._message(0)


def main():
    style = sys.argv[1]
    check = "--check" in sys.argv[2:]
    src = AP.read_text()
    rx = r'(cfg\.get\(\s*["\']prompt_style["\']\s*,\s*["\'])([a-z0-9]+)(["\']\s*\))'
    m = re.search(rx, src)
    if not m:
        print("ERROR: prompt_style default not found"); return 1
    old = m.group(2)
    new_src = re.sub(rx, lambda mm: mm.group(1) + style + mm.group(3), src, count=1)
    ast.parse(new_src)
    if check:
        print(f"[check] would change prompt_style default {old} -> {style}"); return 0
    AP.write_text(new_src)
    # verify the running default now emits the requested style
    msg = _load_default_msg(AP)
    print(f"baked prompt_style default {old} -> {style}")
    print("default _message(0):", repr(msg)[:120])
    out = subprocess.run([sys.executable, str(ROOT/"notebook"/"build_notebook.py")],
                         capture_output=True, text=True)
    print(out.stdout.strip() or out.stderr.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
