# Command-line reference

Every command, the exact invocation for each benchmark, where files are
written, and what a result does and does not mean. The [README](../README.md)
covers installing and running one task; this is the full surface.

## Commands

The public surface is intentionally small:

```text
mini-agent profile
mini-agent run --environment swe|web
mini-agent eval --benchmark swebench|programbench|browsecomp|browsecomp-plus|osworld-v1|osworld-v2
mini-agent eval --benchmark aime|math500|olympiadbench|minervamath
mini-agent grade --benchmark swebench|programbench|browsecomp-plus
mini-agent doctor
mini-agent report --runs DIR [DIR ...] [--best-of]
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
instructions), `message-board` (peers sharing one append-only log instead of
mailboxes), and `recursive` — the free-form mesh `--multi-agent` selects.
`--role-model ROLE=MODEL` runs one role on a different model, and
`--per-agent-input-tokens` bounds each agent in its own right rather than
dividing one pool. Every result records what each agent spent and how much of it
went into coordination. See [docs/harnesses.md](harnesses.md).

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

# Competition mathematics. No container, no index, no VM, and the score is
# produced locally -- the one family that needs nothing but a model.
mini-agent eval \
  --benchmark aime \
  --dataset /data/reasoning/aime-2024.jsonl \
  --model openai/MODEL \
  --output /path/to/durable/mini-agent/runs/aime-canary

# `--grader-model` adds an equivalence judge for the answers normalization
# cannot decide. Without it those answers are scored zero, not dropped.
mini-agent eval \
  --benchmark math500 \
  --dataset /data/reasoning/math500.jsonl \
  --model openai/MODEL \
  --grader-model openai/GRADER_MODEL

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

See [benchmark fidelity](benchmarks.md) for exact pins, prerequisites, and
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

See [machine images](machine-images.md) for exact upstream pins,
acquisition commands, hashes from the validated node, and the optional adjacent
provenance sidecar contract.

## Comparing runs

Every artifact is per-run, but the question the runs answer is comparative.
`report` joins finished runs into one table:

```bash
mini-agent report --runs runs/aime-single runs/aime-fixed-team-3 --format table
```

It recomputes nothing. Each row is a projection of what a run already
committed, and only committed instances count — resume refuses to trust an
uncommitted one, so a table that averaged it in would report a score resume
would not. Two things are called out rather than smoothed over: a summary whose
`mean_score` disagrees with its own committed results, and a run whose
`max_concurrency` was below its team size, where the team serialized on one
semaphore and the latency columns are not comparable. Runs recorded before the
coordination block carried idle time and duplicate work show `-` in those
columns rather than `0`, because not measured is not the same finding as
measured as none.

`--best-of` treats the given runs as repeated samples of one configuration and
asks what the extra agents actually bought:

```bash
mini-agent report --runs runs/aime-single-{1,2,3} --best-of
```

**Observed best@k** is the fraction of tasks some run solved. **Independent
expectation** is what k statistically independent runs would have scored,
`1 - (1 - p)^k` at the per-run mean `p`. Agents that think alike solve and fail
the same tasks, so observed falls below expected; inverting the same formula
turns the gap into an **effective team size** — the number of genuinely
independent runs that would have produced what these k produced. A team of ten
worth 2.3 independent runs is the finding this exists to surface, and a mean
score cannot show it. Where the question has no answer — a per-run mean of 0 or
1, or a union that solved everything — it reports nothing rather than a
misleading number.

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
[validation](validation.md).

