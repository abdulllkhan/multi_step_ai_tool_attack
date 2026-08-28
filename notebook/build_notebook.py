"""Generate the Kaggle submission notebook from attack.py.

The competition privately re-runs this notebook with a hidden test set and extracts
`/kaggle/working/submission.csv`. So the notebook must (1) put the competition
dataset on sys.path, (2) write our attack.py to /kaggle/working/, and (3) run the
competition's JED inference server, which produces submission.csv. This mirrors the
official getting-started notebook. Regenerate after editing attack.py:

    .venv/bin/python notebook/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ATTACK = ROOT / "attack.py"
OUT = ROOT / "notebook" / "submission.ipynb"


def _code_cell(source_lines: list[str]) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": f"cell-code-{abs(hash(tuple(source_lines))) % 10**8}",
        "metadata": {},
        "outputs": [],
        "source": source_lines,
    }


def _md_cell(source_lines: list[str]) -> dict:
    return {
        "cell_type": "markdown",
        "id": f"cell-md-{abs(hash(tuple(source_lines))) % 10**8}",
        "metadata": {},
        "source": source_lines,
    }


def build() -> None:
    attack_lines = ATTACK.read_text(encoding="utf-8").splitlines(keepends=True)

    intro = _md_cell(
        [
            "# AI Agent Security — Multi-Step Tool Attacks\n",
            "\n",
            "**Approach: adaptive Go-Explore over tool-call cells.** This notebook puts the\n",
            "competition dataset on `sys.path`, writes our `attack.py` to `/kaggle/working/`,\n",
            "then runs the competition inference server, which emits `submission.csv`.\n",
            "The hosted re-run substitutes the hidden test set (real models + guardrails).\n",
        ]
    )

    setup = _code_cell(
        [
            "import sys, glob\n",
            "from pathlib import Path\n",
            "\n",
            "# Avoid argparse conflicts inside Kaggle notebooks.\n",
            "sys.argv = [sys.argv[0]]\n",
            "\n",
            "# The competition dataset contains kaggle_evaluation/ and aicomp_sdk/ at its root.\n",
            "for candidate in glob.glob('/kaggle/input/**/kaggle_evaluation', recursive=True):\n",
            "    dataset_root = str(Path(candidate).parent)\n",
            "    if dataset_root not in sys.path:\n",
            "        sys.path.insert(0, dataset_root)\n",
            "    print('Dataset root:', dataset_root)\n",
            "    break\n",
            "\n",
            "print('Setup complete')\n",
        ]
    )

    writefile = _code_cell(["%%writefile /kaggle/working/attack.py\n", *attack_lines])

    serve = _code_cell(
        [
            "# Run the competition's evaluation server so it writes submission.csv.\n",
            "# .run(): during Kaggle's private re-run (KAGGLE_IS_COMPETITION_RERUN set) it\n",
            "# blocks and lets the real gateway drive scoring against gpt_oss/gemma; in this\n",
            "# visible commit it runs a LOCAL gateway that writes /kaggle/working/submission.csv.\n",
            "import os, importlib, pkgutil\n",
            "\n",
            "# For the visible commit only, force the deterministic (GPU-free) agent so the\n",
            "# local gateway doesn't try to load the 20B models and OOM. The scored rerun is\n",
            "# untouched, so it still evaluates gpt_oss/gemma at full budget.\n",
            "if os.getenv('KAGGLE_IS_COMPETITION_RERUN') is None:\n",
            "    os.environ.setdefault('AICOMP_MODEL_NAMES', 'deterministic')\n",
            "\n",
            "import kaggle_evaluation\n",
            "pkg = next((m.name for m in pkgutil.iter_modules(kaggle_evaluation.__path__)\n",
            "            if m.name.startswith('jed_attack')), 'jed_attack_134815')\n",
            "server_mod = importlib.import_module(\n",
            "    f'kaggle_evaluation.{pkg}.jed_attack_inference_server')\n",
            "server_mod.JEDAttackInferenceServer().run()\n",
            "\n",
            "# Visible commit only: the code-competition submit validates submission.csv's\n",
            "# row ids, so write the canonical leaderboard rows (placeholder scores). The\n",
            "# private rerun overwrites this with real scores (this code does NOT run during\n",
            "# the rerun, because .serve() blocks inside .run() until the gateway ends it).\n",
            "if os.getenv('KAGGLE_IS_COMPETITION_RERUN') is None:\n",
            "    import csv\n",
            "    with open('/kaggle/working/submission.csv', 'w', newline='') as _f:\n",
            "        _w = csv.writer(_f)\n",
            "        _w.writerow(['Id', 'Score'])\n",
            "        for _rid in ('gpt_oss_public', 'gpt_oss_private',\n",
            "                     'gemma_public', 'gemma_private'):\n",
            "            _w.writerow([_rid, 0.0])\n",
            "    print('Wrote submission.csv with canonical leaderboard row ids')\n",
        ]
    )

    nb = {
        "cells": [intro, setup, writefile, serve],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({len(attack_lines)} attack.py lines embedded; 4 cells)")


if __name__ == "__main__":
    build()
