# Local coding agent for the hackathon — setup & usage

Your DGX Spark now runs a local, uncensored coding **agent** (not just a model) that reads
this repo, edits files, runs commands, and understands the competition. This is the doc
for using it. Everything is local, free, and offline.

---

## What was set up, and why

**Harness: Qwen Code CLI** (`qwen`) — already installed and configured.

It won a head-to-head against OpenCode, Crush, Aider, Goose, Codex CLI, Cline, and
Continue for *this* job. The deciding factor is your model: an 80B MoE with only ~3B
active params, whose #1 failure mode is malformed tool calls. Qwen Code is the Qwen
team's own reference harness, so its tool-call format matches Qwen3-Coder natively →
the fewest broken-tool-call failures. It has the full agent loop (read/edit files, run
shell, MCP), a project context file, a headless mode, and zero legal baggage.

- **Runner-up:** OpenCode (`sst/opencode`) — reach for it if you want a more polished
  autonomous-agent UX. **Fallback:** Aider (installed) — deterministic diffs for pure
  edit bursts.
- On the "leaked Claude Code in Rust" idea: a real accidental source exposure happened
  in early 2026, and Anthropic DMCA'd an earlier reverse-engineer. Claude Code is
  proprietary; the "Rust rewrite" repos are mixed-provenance and legally toxic. **Not
  used, not advisable.** The open harnesses above are better *and* clean.

**Context file: `QWEN.md`** (in this repo) — the reason the agent is competent instead
of "learning from scratch." It encodes the mission, the **exact scoring formula**, the
SDK predicate/guardrail mechanics, your current strategy, and the improvement
directions. Qwen Code auto-loads it. Keep it updated as your understanding grows — it's
the single biggest lever on agent quality.

**Verified working** (probed on this repo): the agent read `QWEN.md` + `attack.py`,
reproduced the scoring formula, proposed a valid improvement (distinct URL paths → more
score-cells), and created a file. Read + reason + write all confirmed.

---

## Quick start

```bash
cd ~/Documents/multi_step_tool_attack
qwen                       # interactive agent in this repo (auto-loads QWEN.md)
```

Ask it things like:
- "Read attack.py and add a variant that puts a unique path on each POST URL, so each
  finding lands in a new score-cell. Keep the interface and the 90s budget cap."
- "Explain why the optimal guardrail zeroes out a naive agent, using the SDK."

**Headless / one-shot** (good for scripting):
```bash
qwen -p "summarize attack.py's strategy in 5 bullets"          # read-only, prints, exits
qwen -p --yolo "implement <change>; keep attack.py importable"  # auto-runs tools (see safety)
```

Connection is preconfigured in `~/.qwen/.env` (local server, model
`qwen3-coder-next-uncensored`). Nothing else to set.

---

## Safety modes (important for autonomous runs)

`qwen` asks approval for each file-write/shell command by default — safe. `--yolo`
auto-executes **everything at your user privileges with no sandbox** (you saw the
warning during setup). For unattended or risky work, sandbox it:

```bash
qwen --sandbox -p "..."        # or set QWEN_SANDBOX=1
```

Always work on a git branch so anything the agent does is reversible:
```bash
git switch -c agent/work-$(date +%m%d)
```

---

## The feedback loop (make the agent *measure*, not guess)

The competition's fitness is a local eval against the real target model:
```bash
bash dev/score.sh 150        # runs attack.py vs gpt-oss-20b, prints SUMMARY + score
```

**GPU memory caveat (important):** the target model (`gpt-oss-20b`, ~26 GB) and the Qwen
coding agent (Q8_0, ~85 GB) do **not** both fit in the 107 GB GPU at once. So:

- **Write code** with the Qwen agent (server up), then **stop it to score**:
  ```bash
  systemctl --user stop qwen-server.service   # free the GPU
  bash dev/score.sh 150                        # eval vs gpt-oss
  systemctl --user start qwen-server.service   # bring the agent back
  ```
- **Want both resident at once** (agent writes *and* scores in one session)? Run Qwen at
  a smaller quant so it coexists with gpt-oss: switch `MODEL_FILE` to the **Q4_K_M**
  (~49 GB) coder quant in `~/Documents/qwen-models/config/server.env` — then 49 + 26 GB
  fits with headroom. Q4 isn't downloaded yet; ask and I'll pull it (~50 GB).

> Note: the local eval was validated to *load* gpt-oss (116 s) but one smoke run exited
> before printing a score — if `dev/score.sh` comes back empty, run
> `.venv/bin/python dev/eval_real.py --model gpt_oss --guardrail optimal --budget-s 120`
> directly and check the tail for a traceback; it may need a small fix in the SDK's
> shared-backend path. This is separate from the coding agent, which works.

---

## Managing the model server

```bash
systemctl --user status qwen-server.service      # is it up?
systemctl --user restart qwen-server.service     # after a config change
tail -f ~/Documents/qwen-models/logs/server.log  # watch it
```
Config: `~/Documents/qwen-models/config/server.env` (quant ladder + LAN/API-key details).
Also reachable from your Mac over Tailscale at `http://spark-qwen:8080/v1`.

---

## Fallbacks

**Aider** (deterministic diffs, git-native) — already pointed at the local model:
```bash
cd ~/Documents/multi_step_tool_attack
aider attack.py --read QWEN.md         # uses ~/.aider.conf.yml + ~/.env
```

**OpenCode** (if you want to try the runner-up):
```bash
npm i -g opencode-ai
# ~/.config/opencode/opencode.json: OpenAI-compatible provider, baseURL
#   http://127.0.0.1:8080/v1, model key "qwen3-coder-next-uncensored", context 131072
```

---

## Files added by this setup (uncommitted, on branch MS-2)
- `QWEN.md` — agent context (the important one)
- `dev/score.sh` — one-command local fitness check
- `dev/overnight_optimize.py`, `dev/variants.json`, `dev/apply_best_config.py`,
  `dev/run_overnight.sh` — the (optional) unattended config-optimizer, stage-only

Commit them when you're happy: `git add QWEN.md dev/ && git commit -m "local coding-agent harness"`.
