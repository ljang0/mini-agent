# mini-agent

One small agent loop for SWE, offline web research, computer use, and bounded
multi-agent experiments.

```python
from mini_agent import MiniAgent

agent = MiniAgent(
    model=model,
    environment=environment,
    system_prompt=system_prompt,
    max_steps=64,
    context=context,
)
result = await agent.run(task)
```

The complete domain-neutral loop is [108 readable lines](https://github.com/ljang0/mini-agent/blob/main/src/mini_agent/agent.py).
It maintains a linear history, asks the model for actions, executes only declared
environment tools, records budget/trace events, and stops. Provider codecs,
parsers, environments, datasets, graders, and orchestration remain outside it.

## What works

| Domain | Agent tools | Reproducible evaluation boundary |
|---|---|---|
| SWE | `bash(command)` | private local copy or persistent SWE-bench Docker container; binary patch captured before cleanup; official 4.1.0 prediction/grader contract |
| Web | `search(query)` | BrowseComp-Plus fixed corpus, canonical top-5 Lucene BM25, 512-token snippets, per-query official records and pinned evaluator argv |
| CUA | `computer(...)` | cua-speed-run agent plane and exact two-file export; OSWorld reset/step/record/evaluate bridge with the verifier kept outside the agent |
| Multi-agent | four communication tools | ordinary `MiniAgent` instances, bounded async lifecycle, shared and per-agent budgets, stable resource identities, root-only result |

The web profiles implement all eight published BrowseComp client families through
the same wrapper: OpenAI, Anthropic, Gemini, GLM/Z.ai, GPT-OSS, Qwen, Search-R1,
and Tongyi. SWE includes mini-swe tool-call, text, named-backtick, and XML action
profiles plus a minimal SWE-agent bash profile. CUA maps all 18 cua-speed-run
templates: 13 MiniAgent profiles, two external nested-agent references, and three
fixtures.

Profiles are source-informed implementations, not claims that a normalized
wrapper reproduces a proprietary runtime or training method. Every profile has a
fidelity label and explicit gaps. Exact upstream loops remain external references.

## Install

```bash
python3 -m pip install mini-agent
# or, from a source checkout
python3 -m pip install -e .
# optional benchmark dependencies
python3 -m pip install -e '.[web,swebench,cua]'
```

The public CLI is intentionally small:

```text
mini-agent run
mini-agent eval
mini-agent grade
mini-agent doctor
mini-agent export
mini-agent profile
mini-agent catalog
mini-agent reference list|validate|run
```

Inspect a fully resolved prompt and policy:

```bash
mini-agent profile --application web --profile openai --model gpt-5.4
```

Run one local task:

```bash
mini-agent run --application swe --model openai/gpt-5.4 \
  --profile mini-swe-tool-call --workspace /path/to/repo \
  --task 'Fix the failing tests' --output runs/local-swe

mini-agent run --application web --model openai/gpt-5.4 \
  --profile default --corpus tests/fixtures/browsecomp_plus/corpus.jsonl \
  --task 'Find the answer and cite document IDs'

mini-agent run --application cua --model openai/gpt-5.4 \
  --profile gpt54 --env-url http://127.0.0.1:8000 \
  --task 'Complete the visible desktop task'
```

Add `--mode multi --max-agents 4` to a one-task run to add only
`spawn_agent`, `send_message`, `read_messages`, and `wait`. Web agents receive
separate wrappers over immutable retrieval data; SWE agents receive separate
copies; only the root CUA agent controls the desktop.

## Evaluations

`eval` performs generation and writes durable, resumable artifacts:

```text
run/
  manifest.json
  instances/<task-id>/result.json
  instances/<task-id>/trace.jsonl
  artifacts/
  official/
  summary.json
```

SWE example:

```bash
mini-agent eval --application swe --model openai/gpt-5.4 \
  --profile swebench --tasks /data/swebench.jsonl --output runs/swe \
  --max-workers 2
mini-agent grade --application swe --predictions runs/swe/predictions.jsonl \
  --dataset-name princeton-nlp/SWE-bench_Verified --run-id mini-agent-smoke
```

BrowseComp-Plus uses `--index-path` plus a resolved local `--tokenizer-path` for
canonical runs. `--corpus` is a deterministic fixture backend only. The official
grader requires both the pinned source checkout and a local judge-model snapshot:

```bash
mini-agent eval --application web --model openai/gpt-5.4 \
  --profile openai --tasks /data/browsecomp/queries.tsv \
  --index-path /data/browsecomp/lucene-index \
  --tokenizer-path /models/qwen-tokenizer --output runs/web
mini-agent grade --application web --checkout /src/BrowseComp-Plus \
  --input-dir runs/web/official --ground-truth /data/ground_truth.json \
  --eval-dir runs/web/grades --judge-model /models/Qwen2.5-72B-Instruct
```

Add `--mode multi --max-agents 4 --child-profile PROFILE` to SWE or web
generation to run the same batch contract with communication-enabled MiniAgents.
Child profiles are explicitly allowlisted and each task still produces one
root-owned official record.

CUA `eval` delegates provisioning, clock, seeds, hidden verification, and scoring
to the pinned cua-speed-run checkout. Export the desired single- or multi-agent
submission first. The mini-agent wheel is embedded and hash-checked so the
submission itself remains exactly two files. `init.py` installs the runtime and
its complete dependency wheel set from the embedded, hash-checked wheelhouse; it
does not use a package index or depend on preinstalled Python packages.

```bash
python3 -m build
python3 -m pip download --only-binary=:all: \
  --requirement requirements/cua-export.lock --dest dist/cua-wheelhouse
dependency_args=()
for dependency in dist/cua-wheelhouse/*.whl; do
  dependency_args+=(--dependency-wheel "$dependency")
done
mini-agent export --target cua-speed-run --profile gpt54 \
  --model gpt-5.4 --provider openai-responses \
  --wheel dist/mini_agent-0.3.0-py3-none-any.whl \
  "${dependency_args[@]}" \
  --mode multi --max-agents 4 --child-profile gpt54 \
  --output submissions/gpt54-multi
mini-agent eval --application cua --checkout /src/cua-speed-run \
  --cua-executable /src/cua-speed-run/.venv/bin/cua-speed-run \
  --submission submissions/gpt54-multi --benchmark /data/osworld-mini \
  --output runs/cua
```

Repeat `--dependency-wheel` for every file downloaded for the target Python and
platform (including conditional lock entries). Export verifies the closed wheel
set with offline pip resolution, embeds every artifact and digest in `init.py`,
and bootstrap uses `--no-index`. The CUA runner console script and its Python
interpreter must resolve inside the verified pinned checkout; execution imports
the pinned checkout source directly. Codex CLI and Claude Code keep their own
nested loops and therefore remain references, not MiniAgent profiles.

Before real runs, use `mini-agent doctor --application ...`. Missing credentials,
data, Docker/VM, Java, tokenizer snapshots, or hardware produce a machine-readable
`blocked` result; a skipped or blocked smoke is never presented as passing.

## Reproducibility and research boundaries

Every model and tool call uses `RunContext`'s shared ledger and trace. Batch
records are atomic, resume manifests are fingerprinted, and official evaluator
inputs are kept separate from internal traces. Benchmark answers, qrels, graders,
verifiers, and training procedures never enter the agent core.

The former Scaffold Lab tree and its 28 topology studies are archived under
[research/scaffoldlab](https://github.com/ljang0/mini-agent/tree/main/research/scaffoldlab) and preserved by tag
`scaffoldlab-v0.2-handoff`. It is not installed. The read-only catalog still
exposes 55 audited rows across 18 labs without promoting studies or gaps to exact
implementations:

```bash
mini-agent catalog --json
mini-agent catalog --frontiers --json
mini-agent reference list --application swe
```

Catalog and reference listing work from the normal `mini-agent` installation.
Explicit `reference validate` and `reference run` commands additionally require
the preserved Scaffold Lab evaluator from the tagged/source archive (or an
equivalent separate installation), plus the selected reference's pinned
checkout or distribution, credentials, and environment where applicable. The
archive and its loops are not bundled into the `mini-agent` wheel.

## Verification

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m compileall -q src tests
ruff check src tests
mypy src/mini_agent
python3 -m build
twine check dist/*
```

See [architecture](https://github.com/ljang0/mini-agent/blob/main/docs/architecture.md),
[handoff](https://github.com/ljang0/mini-agent/blob/main/HANDOFF.md),
[contributing](https://github.com/ljang0/mini-agent/blob/main/CONTRIBUTING.md), and
[security](https://github.com/ljang0/mini-agent/blob/main/SECURITY.md).
