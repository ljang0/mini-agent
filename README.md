# mini-agent

[![CI](https://github.com/ljang0/mini-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/ljang0/mini-agent/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/mini-agent-cmu)](https://pypi.org/project/mini-agent-cmu/)
[![Python](https://img.shields.io/pypi/pyversions/mini-agent-cmu)](https://pypi.org/project/mini-agent-cmu/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

`mini-agent` is one small agent loop for three environment families: software
engineering, web research, and computer use. Provider codecs, tools, benchmark
loaders, graders, storage, and multi-agent scheduling stay outside that loop.

The design follows the useful constraint in
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent): keep the agent
boring and make everything else replaceable. This project is alpha software. It
is infrastructure for controlled experiments, not a claim that one normalized
prompt reproduces every provider's proprietary harness.

## Quickstart

```bash
pip install mini-agent-cmu
```

(The distribution is `mini-agent-cmu` — PyPI blocks the bare name as too
similar to an unrelated existing project — but the import stays
`mini_agent` and the command stays `mini-agent`.)

Run a research agent over a local JSONL corpus — no VM, no benchmark checkout,
no search API key; retrieval is deterministic in-process BM25:

```bash
export OPENAI_API_KEY=...
mini-agent run --environment web --web-backend jsonl \
  --corpus examples/corpus.jsonl \
  --task 'Which document explains when the wind turbine furls?' \
  --model openai/MODEL
```

Each corpus line is one JSON object with `docid` (or `id`) and `text` (or
`contents`); see [`examples/corpus.jsonl`](examples/corpus.jsonl). The run
prints its output directory on start — `tail -f <output>/trace.jsonl` streams
every model and tool event live.

Or drive the loop as a library with no key at all
([`examples/library_quickstart.py`](examples/library_quickstart.py) is the
complete runnable version):

```python
import asyncio
from mini_agent import MiniAgent, RunContext, ScriptedModel, ModelResponse
from mini_agent.models import build_model  # also exported as mini_agent.build_model

model = ScriptedModel([ModelResponse("done")])  # or build_model("openai/MODEL")
agent = MiniAgent(
    model=model,
    environment=my_environment,  # any BaseEnvironment subclass
    system_prompt="Solve the task with the available tools.",
    max_steps=64,
    context=RunContext(),
)
result = asyncio.run(agent.run("the task"))
```

See [docs/library.md](docs/library.md) for custom environments, budgets, and
accounting.

## Design contract

- `MiniAgent` owns only linear history, model calls, declared tool calls, and
  stopping.
- A SWE environment exposes one `bash` tool. A web environment exposes one
  `browser` tool; general runs enable `search` and `open`, while fixed
  BrowseComp-Plus is search-only. A computer environment exposes one batched
  `computer` tool using native screenshot pixels.
- OpenAI Responses, OpenAI-compatible Chat Completions, and Anthropic Messages
  are thin maintained codecs. Meta deployments require an explicit endpoint and
  a `--protocol` choice; protocol- or model-specific behavior belongs downstream
  unless it is truly universal.
- Every maintained run path uses the same concurrency-safe budget ledger and
  streaming trace recorder.
- Multi-agent mode adds one `agent` tool with `spawn`, `send`, blocking or
  non-blocking `inbox`, `wait`, descendant `stop`, and explicit `adopt` actions.
  Every worker is still an ordinary `MiniAgent`.
- Evaluation control data, expected answers, qrels, and verifiers never enter an
  agent environment or prompt.

Prefer delegation on by default? `pip install multi-mini-agent` adds a
`multi-mini-agent` console script — the identical CLI with `--multi-agent`
implied for `profile`/`run`/`eval` (a thin companion distribution from
[`packaging/multi-mini-agent`](packaging/multi-mini-agent), not a second
codebase).

The recursive scheduler has no topology depth constant. It is still finite:
global and per-agent budgets, active-agent and total-agent limits, mailbox limits,
tool limits, timeouts, and environment leases all apply. It is a minimal
inference topology; it does not claim to reproduce any policy-trained
multi-agent method.

## Install

Python 3.10 through 3.13 is supported. The upstream-locked `web-fixed` extra
supports Python 3.10 through 3.12 because Pyjnius 1.6.1 does not publish a
Python 3.13 wheel.

```bash
python -m pip install -e .

# Only install the extras needed on this machine.
python -m pip install -e '.[web-live]'
python -m pip install -e '.[web-fixed]'
```

SWE-bench grading requires an exact editable upstream checkout because the tag's
built wheel omits tracked runtime resource files used by the official harness:

```bash
git clone https://github.com/SWE-bench/SWE-bench.git /src/SWE-bench
git -C /src/SWE-bench checkout 726c5461e2ef52d83cf1ea2107870a8bb3328d57
python -m pip install -e /src/SWE-bench
```

Upstream did not publish version 4.1.0 to PyPI, so a plain
`swebench==4.1.0` requirement is not installable. Grading verifies the installed
editable source and its required resource inventory before and after use; a
wheel built from the tag is intentionally rejected as incomplete.

Computer benchmarks use their pinned upstream checkouts and environment
dependencies; there is no synthetic catch-all `computer` extra.

The fixed-web extra installs only the pinned Rust tokenizer, its Hub downloader,
and the JNI bridge. It reads the same Qwen `tokenizer.json` used upstream without
installing a general model framework. The 185 MB Anserini fat JAR and Lucene
index are explicit, hashed benchmark assets rather than an excuse to install
Pyserini's unused dense-retrieval and server stack.

Models use `provider/model` syntax:

- `openai/MODEL` uses `OPENAI_API_KEY` and OpenAI Responses by default.
- `anthropic/MODEL` uses `ANTHROPIC_API_KEY` and Anthropic Messages.
- `meta/MODEL` uses `MODEL_API_KEY`; it requires `--base-url` because
  mini-agent does not guess a Meta deployment's hostname or protocol support.
  Without `--protocol` it remains the opt-in Responses-compatible adapter.

`--protocol chat-completions` selects the OpenAI-compatible Chat Completions
adapter for `openai/` and `meta/` models. It resends the full wire transcript
each turn (no `previous_response_id`) and never sets sampling parameters, so
the server's defaults always apply. `--provider-header NAME=VALUE` attaches
non-secret static headers (for example a deployment-required session id) to
every provider call; credential-looking header names are rejected.

Transcript-replay codecs (Chat Completions and Anthropic Messages) keep only
the newest `--max-history-images` screenshots when replaying history (default
4); older image blocks become a fixed text placeholder, declared as a
translation loss and recorded in run manifests. This is what makes long
computer-use runs feasible on those paths — a 64-step run replays at most K
images per call instead of every prior screenshot. Pass
`--max-history-images unlimited` to restore full replay. The Responses path
keeps continuation server-side and rejects the option.

Transient provider failures (408/429/5xx and transport errors) are retried
with jittered backoff, honoring `Retry-After`; `--provider-retries N` bounds
the attempts (`0` disables) and `--provider-timeout` overrides the 300-second
per-request default. Retries never extend a run past its wall-clock budget,
and retry counts appear in trace events.

Override a trusted compatible endpoint with `--base-url` and its credential name
with `--api-key-env`. Put credentials only in environment variables, never in
`--provider-body` or `--provider-header`.

For a live evaluation intended to be reproducible, pass the exact model value
the provider promises to return as `--expected-provider-model` (and
`--grader-expected-provider-model` for BrowseComp). The adapter hashes every
observed response model, rejects an unexpected snapshot, and rejects a snapshot
change within one agent run. Without this option a requested alias is recorded
honestly as an alias, not treated as an immutable model-card identity.

The OpenAI Responses adapter chains tool turns with `previous_response_id`.
Accordingly, it rejects `{"store": false}`: Zero Data Retention requires a
different downstream adapter that replays every response item, including encrypted
reasoning items, instead of pretending this continuation mode is compatible.

## CLI

The public surface is intentionally small:

```text
mini-agent profile
mini-agent run --environment swe|web
mini-agent eval --benchmark swebench|programbench|browsecomp|browsecomp-plus|osworld-v1|osworld-v2
mini-agent grade --benchmark swebench|programbench|browsecomp-plus
mini-agent doctor
```

Resolve the maintained baseline without making a model call:

```bash
mini-agent profile \
  --application web \
  --profile default \
  --model openai/MODEL
```

The provider-neutral, versioned agent contract is available as canonical JSON:

```bash
mini-agent profile \
  --application web \
  --model openai/MODEL \
  --multi-agent \
  --format agent-spec
```

Use `--format translation-report` for an explicit field-level loss report. The
report includes the selected provider codec's declared losses — the
OpenAI-protocol codecs drop the tool-result error flag and relocate tool-result
images into a synthetic user message, and every codec restricts tools to the
generic function kind — so `exact` is only reported when the codec declares
none. Even then the claim is deliberately scoped to declared fields; it is
never a claim of behavioral, policy-training, timing, tool, or benchmark
equivalence.

Downstream Python adapters can load that document with `AgentSpecV1.from_json`
and call `spec.bind(model=..., model_id=..., environment=...,
environment_id=...)`. Binding checks the explicit model/domain identities,
runtime tool names, communication action enum, step limit, prompt, and shared
budget before constructing the same `MiniAgent`; endpoint credentials and
benchmark assets stay in their respective downstream constructors.

Run a SWE task. Single-agent mode edits the selected workspace directly:

```bash
mini-agent run \
  --environment swe \
  --workspace /path/to/repository \
  --task 'Fix the failing tests and verify the change.' \
  --model openai/MODEL \
  --home /path/to/durable/mini-agent \
  --scratch /path/to/local-scratch/mini-agent
```

With `--multi-agent`, every SWE worker gets an independent private repository
copy. The root's selected state is exported as `patch.diff`; the source workspace
is not changed.

`--harness` selects a named multi-agent topology instead, so runs that differ
only in coordination structure can be compared:

```bash
mini-agent eval --benchmark programbench --harness fixed-team --team-size 3 \
  --max-active-agents 3 --max-total-agents 3 --model-concurrency 3 \
  --per-agent-model-calls 64 --agent-git-share ...
```

The choices are `single`, `fixed-team` (peers with a designated lead),
`orchestrator` (a coordinator with no task tools, delegating to blocking
subagents), `async-subagents` (long-lived subagents that idle between
instructions), and `recursive` — the free-form mesh `--multi-agent` selects.
Every result records what each agent spent and how much of it went into
coordination. See [docs/harnesses.md](docs/harnesses.md).

Run live web research with SerpAPI, or use JSONL BM25 for a small deterministic
fixture:

```bash
export SERPAPI_API_KEY=...
mini-agent run \
  --environment web \
  --web-backend serpapi \
  --page-reader http \
  --task 'Answer the question and cite the returned references.' \
  --model openai/MODEL

mini-agent run \
  --environment web \
  --web-backend jsonl \
  --corpus /path/to/corpus.jsonl \
  --task 'Find the relevant document.' \
  --model openai/MODEL
```

Use `--max-model-calls`, token limits, `--max-cost-usd`, wall time, tool limits,
and agent limits for every paid run. A cost cap requires explicit input and output
prices; unknown or incomplete usage fails closed.

## Evaluations

`eval` defaults to one task as a canary. Use `--all` deliberately. Generation is
separate from official grading.

```bash
# SWE-bench generation in independent persistent containers.
mini-agent eval \
  --benchmark swebench \
  --dataset /data/swebench-verified.jsonl \
  --runtime docker \
  --model openai/MODEL \
  --output /path/to/durable/mini-agent/runs/swe-canary \
  --scratch /path/to/local-scratch/mini-agent-swe

# ProgramBench generation: offline `--network none` cleanroom containers.
mini-agent eval \
  --benchmark programbench \
  --checkout /src/ProgramBench \
  --runtime docker \
  --model openai/MODEL \
  --output /path/to/durable/mini-agent/runs/programbench-canary

# Live BrowseComp plus an independently configured private-answer grader.
mini-agent eval \
  --benchmark browsecomp \
  --dataset /data/browsecomp.csv \
  --model openai/MODEL \
  --grader-model openai/GRADER_MODEL \
  --output /path/to/durable/mini-agent/runs/browsecomp-canary

# Fixed-corpus BrowseComp-Plus generation.
mini-agent eval \
  --benchmark browsecomp-plus \
  --dataset /data/browsecomp-plus/queries.tsv \
  --index /data/browsecomp-plus/index \
  --anserini-jar /data/browsecomp-plus/anserini-1.1.1-fatjar.jar \
  --snippet-tokenizer Qwen/Qwen3-0.6B \
  --snippet-tokenizer-revision COMMIT \
  --model openai/MODEL

# OSWorld lifecycle and hidden evaluation from an exact pinned checkout.
# On a Docker-less KVM host, run the official container image with Apptainer.
mini-agent eval \
  --benchmark osworld-v1 \
  --checkout /src/OSWorld \
  --provider-name docker \
  --runtime apptainer \
  --path-to-vm /assets/osworld/Ubuntu.qcow2 \
  --osworld-apptainer-image /assets/osworld/osworld-docker.sif \
  --model openai/MODEL
```

Every evaluation writes an immutable manifest fingerprint, one shared redacted
JSONL trace, atomic task results with commit markers, accounting, timing, and
hash-bound benchmark-specific artifacts. Artifact bytes and their atomic rename
are synced before a commit marker is treated as durable. `--resume` accepts the
exact same task data, configuration, and limits, and restores already-spent
accounting.
If a crash leaves an uncommitted task after a model or tool operation started,
resume refuses to guess its spend or side effects; start a new output rather
than silently repeating it. The start record is synced to durable storage before
the provider or environment request begins.

```text
run/
  manifest.json
  trace.jsonl
  summary.json
  instances/<hashed-task-id>/result.json
  instances/<hashed-task-id>/completed.json
  predictions.jsonl                 # SWE-bench
  official_run/                     # ProgramBench
  official_runs/                    # BrowseComp-Plus
```

Official graders are explicit commands:

```bash
mini-agent grade \
  --benchmark swebench \
  --evaluation /path/to/evaluation \
  --dataset /data/SWE-bench_Verified.jsonl \
  --run-id mini-agent-canary

mini-agent grade \
  --benchmark programbench \
  --evaluation /path/to/evaluation \
  --checkout /src/ProgramBench

mini-agent grade \
  --benchmark browsecomp-plus \
  --evaluation /path/to/evaluation \
  --checkout /src/BrowseComp-Plus \
  --ground-truth /data/ground_truth.jsonl \
  --qrel-evidence /data/qrel_evidence.txt \
  --judge-model /models/Qwen3-32B-materialized
```

Grading accepts only a fingerprint-valid matching evaluation manifest and local
content-hashed inputs. It snapshots predictions/runs and hidden answer data into
a private `0700` grade directory before invoking upstream. SWE-bench requires
the current Python to contain exactly `swebench==4.1.0`. Image verification uses
the same Docker SDK, explicit `DOCKER_HOST`, and allowlisted environment as the
upstream grader; the generation manifest's runtime command is inert provenance.
ProgramBench requires the pinned checkout plus `programbench==1.2.4`, and runs
the official `programbench eval` with `--output` outside the hashed submission
snapshot so the evaluator cannot mutate its own input.
BrowseComp-Plus checks
the pinned checkout and lockfile plus the lock's direct grader versions
(`numpy==1.26.4`, `tqdm==4.67.1`, `vllm==0.9.0.1`). Its judge model must be a
materialized, symlink-free local directory, not a mutable model name. The grade
manifest, captured stdout/stderr, return code, and hash inventory of official
outputs are private evidence; do not publish the grade directory.

See [benchmark fidelity](docs/benchmarks.md) for exact pins, prerequisites, and
intentional differences from upstream model harnesses.

## Storage and preflight

Keep durable evidence and disposable machine state separate:

```bash
export MINI_AGENT_HOME=/path/to/durable/mini-agent
export MINI_AGENT_SCRATCH=/path/to/local-scratch/mini-agent

mini-agent doctor --target storage
mini-agent doctor --target swebench --runtime docker
mini-agent doctor --target web --web-mode live --page-reader http
mini-agent doctor \
  --target computer \
  --checkout /src/OSWorld \
  --osworld-version v1 \
  --runtime apptainer \
  --path-to-vm /assets/osworld/Ubuntu.qcow2 \
  --osworld-apptainer-image /assets/osworld/osworld-docker.sif
```

`doctor` is non-paid and machine-readable. It reports prerequisites; it does not
claim that inference, a VM launch, or official grading passed.

See [machine images](docs/machine-images.md) for exact upstream pins,
acquisition commands, hashes from the validated node, and the optional adjacent
provenance sidecar contract.

### Environment variables

| Variable | Read by | Purpose |
|---|---|---|
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `MODEL_API_KEY` | provider codecs | default credentials for `openai/`, `anthropic/`, and `meta/` models; override the name with `--api-key-env` |
| `MINI_AGENT_HOME` | storage | durable root for runs, assets, and caches (default `~/.local/share/mini-agent`) |
| `MINI_AGENT_SCRATCH` | storage | disposable node-local work area (default `$MINI_AGENT_HOME/work`) |
| `SERPAPI_API_KEY` | live web backend | SerpAPI credential for `--web-backend serpapi` |
| `DOCKER_HOST` | SWE-bench / OSWorld docker paths | explicit Docker engine endpoint; required for official SWE-bench grading |

## What results mean

The adapters preserve task loading, environment isolation, output schemas, and
hidden evaluator boundaries where documented. The generic tool topology and
baseline prompt are intentionally not the official provider-specific leaderboard
harnesses. In particular:

- BrowseComp-Plus uses one search-only `browser` action rather than the upstream
  provider-specific search name, while retaining fixed retrieval and official
  run/grader artifacts. Full-document `open` is intentionally absent because it
  is opt-in, not the pinned upstream runner's default.
- OSWorld uses a batched native-pixel `computer` action protocol adapted onto the
  official reset/step/evaluate lifecycle.
- Multi-agent mode is a bounded scheduler, not best-of-N, a trained recursive
  policy, or a simulation of proprietary team runtimes.

A passing smoke test is internal QA, not evidence that a configuration is a
release winner. Current machine evidence and blockers live in
[validation](docs/validation.md).

## Development

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m compileall -q src tests
python3 -m ruff check src tests
python3 -m mypy src/mini_agent
PYTHONPATH=src python3 -m mini_agent profile \
  --application web --profile default --model openai/test-model
```

See [architecture](docs/architecture.md), the pinned [reference
ledger](docs/references.md), [contributing](CONTRIBUTING.md), and
[security](SECURITY.md).
