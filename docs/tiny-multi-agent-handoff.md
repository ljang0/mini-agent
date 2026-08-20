# Handoff: CUA and browsing benchmarks for tiny-multi-agent

Work done 2026-08-17/19 in the sibling project [tiny-multi-agent](https://github.com/kohjingyu/tiny-multi-agent)
(branch `feat/cua-and-browsing-compositions`, draft PR
[#17](https://github.com/kohjingyu/tiny-multi-agent/pull/17)). This note lives in mini-agent because mini-agent's
OSWorld/BrowseComp adapters were the working reference; the canonical, always-current copy is
`docs/handoff-cua-browsing.md` on that branch. Author: ljang, with Claude Code.

## What was built

tiny-multi-agent previously had one task domain (SWE: SWE-bench, ProgramBench). The branch adds computer use
(OSWorld-Verified) and browsing (BrowseComp, BrowseComp-Plus), each in exactly ProgramBench's shape, so all bundled
multi-agent scaffolds (peer-team, blocking-subagents, async-subagents) run on them unchanged and
`--harness single-agent` runs a project-owned minimal agent:

- `agents/mini_cua_agent.py` (30 code lines) and `agents/mini_browser_agent.py` (8) — the mini-swe-agent loop,
  model- and environment-agnostic; the CUA one only shows the starting screenshot before the first call and carries
  screenshots inside tool results.
- `environments/desktop_env.py` — one official OSWorld `DesktopEnv` per scaffold slot; `environments/
  computer_actions.py` — the tool contract from OSWorld's reference Muse Spark agent (batched `computer.computer`
  with relative [0, 1000] coordinates converted to one pyautogui script per call, `computer.stop`, "infeasible" →
  FAIL), with the reference system prompt/step budget/screenshot history in `config/osworld.yaml`.
- `environments/web_env.py` — `search`/`open`/terminal `answer` over a pluggable backend: SerpAPI live Google, or
  the official BrowseComp-Plus BM25 Lucene index (pinned Anserini jar, 512-token snippets).
- Thin adapters `benchmarks/{osworld,browsecomp,browsecomp_plus}.py`; runners
  `tiny-multi-agent {osworld,browsecomp,browsecomp-plus}`; grading stays official (OSWorld's per-task evaluator;
  OpenAI's BrowseComp grader prompt behind `tiny-multi-agent-browsecomp-grade`; BCP's evaluator reads our run files).
- Runtime enablers, kept small: `runners/team.py` (`AgentTeam`, the domain-neutral composition factored out of the
  SWE-only `MiniSweAgentTeam`), image-bearing tool results, `max_history_images` request-payload cap,
  `harnesses/single_agent.py`. 277 tests pass; ruff clean; import boundaries enforced by architecture tests.

## Results (Meta Muse Spark `super_nova_ext`, Responses API at api.ai.meta.com)

OSWorld-Verified, 361-task no-Google-Drive split, 100 steps, Apptainer/QEMU VMs on Babel, one attempt per task:

| loop | scored | mean |
| --- | --- | --- |
| OSWorld's reference Muse Spark agent (vendored temporarily to calibrate, then removed) | 346/361 | 78.3% |
| `mini_cua_agent` (`--harness single-agent`) | 361/361 | **74.4%** |
| runtime session through the multi-agent composition (`SingleAgent` scaffold) | 360/361 | 66.8% |

Meta reports 80.8 at 200 steps/episode; the remaining knob is `-c osworld.yaml -c agent.step_limit=200`.
Endpoint findings that cost ~20% of tasks until fixed: stored Responses requests are capped at ~10 MB (→
`store: false`); the first post-reset screenshot can be a partial PNG (→ 60 s settle + whole-PNG re-read); the
endpoint speaks the Responses API natively including `input_image` in tool results and encrypted reasoning passback.
Multi-agent smokes (peer-team 2 desktops) score 1.0; no multi-agent full runs yet.

Browsing: BrowseComp single-agent answers in the official format live; BrowseComp-Plus emits official run files.
A full BCP single-agent run is in flight (830 queries; ~3.8M tokens/query at a 5M cumulative budget — browsing keeps
all snippets in context, so mini-swe-agent's 1M default dies after ~20 calls). BrowseComp live at scale is gated on
SerpAPI budget (~25k searches); grading with a real judge is gated on an OpenAI key.

## Operational state (Babel)

- Checkout `~/tiny-multi-agent` (branch above); node-local venv `/scratch/ljang/tma-venv` — rebuild after node hops:
  `uv venv --python 3.12`, `uv pip install -r ~/odysseys/osworld_runner/requirements.txt`, then
  `uv pip install -e '.[programbench,browsecomp-plus]' ruff` (two steps; OSWorld pins an old transformers).
- Runs and keys on NFS: `/data/user_data/ljang/tma-runs/` (`osworld-full/`, `bcp-full/`, each with `model.env`,
  mode 600). Not visible from login nodes. Batch scripts also in-repo at `scripts/babel/*.sbatch` (resumable:
  finished tasks are skipped on resubmit); score with `scripts/osworld_scores.py`.
- Assets: OSWorld checkout `~/odysseys/osworld_runner` (in-tree apptainer provider), qcow2 at
  `/data/user_data/ljang/osworld_vms/` (stage to `/scratch`), SIF at `/data/user_data/ljang/apptainer/images/`,
  BCP index+jar+decrypted queries at `/data/user_data/ljang/browsecomp-plus/`.
- 2026-08-19: the `cpu` partition began requiring a GPU; scripts now use `general` with `--gres=gpu:1`.

## Open items before merge

1. Split PR #17 (46 files, +5.0k) into runtime / OSWorld / browsing; dedupe the two web runners' shared typer
   options and process-instance scaffolding.
2. Decide the default CUA action space: minimal pyautogui-code tool (bash-like) vs the current structured reference
   contract (keep the other as an option).
3. OSWorld at 200 steps; multi-agent full runs; grade the BCP run officially; BrowseComp live (budget decision).
4. Untested: Docker `DesktopEnv` provider; real-judge grading end to end.
