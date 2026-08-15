# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/) with a 0.x compatibility caveat.

## [Unreleased]

### Added

- Reasoning benchmarks: `mini-agent eval --benchmark {aime,math500,
  olympiadbench,minervamath}`. They need no container, no retrieval index, and
  no virtual machine, which makes them the only family that produces a score on
  a host without Docker. Tasks load from a local JSONL export whose exact bytes
  are hashed into the manifest. Agents get one `bash` tool over a private,
  ungraded scratch directory — deliberately, because with no tool at all every
  topology collapses to "each agent thinks alone" and a harness comparison
  measures nothing.
- Grading for those benchmarks is deterministic normalized comparison, with
  `--grader-model` as an equivalence judge for the answers normalization cannot
  decide. Which grader decided each task is recorded. Without a judge, undecided
  answers are scored zero and flagged `undecided_scored_zero`, so the
  deterministic grader's undercount stays measurable; an unparseable judge reply
  leaves the task ungraded rather than charging grader flakiness to the model.
  No upstream evaluation code runs, so these scores are comparable across
  `mini-agent` runs and are not reproductions of published leaderboard figures.
- A `reasoning` domain profile. It shares the `bash` tool with `swe` but has its
  own prompt, because a mathematics agent is not told to reproduce an issue.
- `--harness message-board`: N peers with no mailboxes, coordinating through one
  shared append-only log. Two new actions, `post` (addressed to nobody) and
  `board` (returns what you have not read, `wait=true` blocks for one). Reading
  does not consume, so two peers see the same post and an agent that looks late
  still sees everything said before it. Board traffic counts toward the same
  coordination tallies as `send`/`inbox`, or a team that talks only through the
  board would report as having said nothing. `--max-board-bytes` bounds the log.
- `--role-model ROLE=PROVIDER/MODEL`, repeatable, runs one harness role on a
  different model — the manager-size-against-worker-size comparison. Each role
  records its own model in its own `AgentSpecV1`, so recorded fingerprints
  describe the agents that actually ran.
- `--per-agent-input-tokens` and `--per-agent-output-tokens` bound each agent in
  its own right rather than dividing one pool, so a team of ten at a million
  tokens each is allowed ten million. Both require a multi-agent harness.
- Idle time and duplicate work in the coordination block, schema
  `mini-agent-coordination-v2`. Each agent reports `lifespan_seconds`,
  `tool_seconds`, `idle_seconds` (lifespan minus time inside a model call and
  inside a tool call, clamped at zero), and `tool_calls_duplicated`; the totals
  add `idle_seconds`, `active_seconds`, `tool_seconds`, and
  `duplicate_tool_calls`. Tool execution is work, not waiting — counting it as
  idle would report a sixty-second `bash` command as an agent doing nothing.
  Both come from events the trace already recorded —
  `agent_spawned`/terminal pairs and `tool_call_started.arguments_sha256` — so
  old traces can be resummarized. An agent whose terminal event is outside the
  slice reports no lifespan rather than an invented one, and one agent repeating
  its own call is not duplicate work.
- `mini-agent report --runs DIR [DIR ...]` joins finished runs into one
  comparison table (`--format table|json`). It recomputes nothing: a row is a
  projection of committed evidence, uncommitted instances are excluded because
  resume will not trust them either, and two conditions are flagged rather than
  smoothed over — a summary whose `mean_score` disagrees with its own results,
  and a run whose `max_concurrency` was below its team size, where the team
  serialized and the latency columns are not comparable.
- `mini-agent report --best-of` compares observed best@k across repeated runs
  against the independent-agent expectation `1 - (1 - p)^k`, and inverts the
  gap into an effective team size — how many genuinely independent runs the k
  actual ones were worth. Correlated agents solve and fail the same tasks, which
  a mean score cannot show. Degenerate cases (a per-run mean of 0 or 1, a union
  that solved everything) report no team size rather than a misleading one.

- `--filter` (regex over task ids) and `--slice` (`start:stop[:step]`) select a
  subset of any benchmark's tasks, applied uniformly after loading. Adopted
  from mini-swe-agent's `filter_instances`, which is simpler and more
  consistent than the three mechanisms this had grown: `--limit`/`--all`
  everywhere, `--sample-seed` for BrowseComp only, and `--task-list` doing
  double duty as both a selection list and a benchmark-data pointer. Selecting
  nothing is an error rather than a run that vacuously passes.
- Selectable multi-agent harnesses. `--harness` chooses a named topology and
  `--team-size` sizes the ones that take a size: `single`, `fixed-team`
  (3/5/10 peers with a designated lead), `orchestrator` (a coordinator with no
  task tools, delegating to blocking subagents), `async-subagents` (a lead with
  long-lived subagents that idle between instructions), and `recursive` — the
  free-form mesh that predates this layer, which `--multi-agent` now aliases.
  Adding a topology is one module in `mini_agent.harnesses` plus one name in
  its registry; nothing in the scheduler, adapters, or CLI changes. See
  `docs/harnesses.md`.
- Two communication actions the four topologies need: `delegate`, which spawns
  one subagent and blocks until it answers, and `release`, which retires an
  idle subagent cleanly so its workspace can be adopted.
- A per-agent coordination block in every evaluation result: model calls, tool
  calls, bytes, tokens, messages sent and received, active seconds, and final
  status. `BudgetLedger.agent_ids()` makes the per-agent tallies reachable.
- `mini-agent eval --benchmark programbench --runtime apptainer` now works; the
  adapter has supported Apptainer since the backend landed, but the CLI refused
  it.

### Fixed

- A failed child no longer kills the run. `_deliver_result` marked its report
  `kind="error"`, which `MailboxMessage` rejected; the resulting `ValueError`
  escaped into terminal handling and became a whole-run failure, and the parent
  never received the error report.
- The ProgramBench manifest recorded `"runtime": "docker"` unconditionally,
  which was true only because the CLI refused every other value.

### Changed

- `BashEnvironment.isolated` takes `git_baseline=False` for a workspace that is
  scratch space rather than a submission, and `source=None` for one that starts
  empty. A fresh private root is what isolates agents from each other; the Git
  baseline exists only to diff against, so an environment nobody exports from
  no longer pays two Git subprocesses per agent for a patch nobody reads. Such
  an environment exports no state — `export_state` returns `None` rather than
  raising, matching what the scheduler already accepts from an environment with
  no state at all — and adoption still fails closed.
- Container runtimes are now benchmark-neutral infrastructure under
  `src/mini_agent/runtimes/`: `base.py` (the `SandboxRuntime` protocol plus the
  shared `ProcessResult` and argument validators), `local.py`, `docker.py`, and
  `apptainer.py`. They contain no benchmark imports, image names, or task
  identifiers; SWE-bench and ProgramBench pass all of that in as configuration.
- `DockerSWEEnvironment`, `ApptainerSWEEnvironment`, and the local
  `BashEnvironment` collapse into one `mini_agent.environments.bash
  .BashEnvironment` over any `SandboxRuntime`. `environments/swe.py` and
  `environments/swebench.py` are removed; import `BashEnvironment`,
  `SWEPatchState`, and `SWEArchiveState` from `mini_agent.environments.bash`,
  and `SWEbenchImageBinding`, `resolve_swebench_image_binding`,
  `swebench_image_name`, `swebench_doctor`, `docker_swe_environment`, and
  `apptainer_swe_environment` from `mini_agent.benchmarks.swebench`.
- Local direct and isolated bash runs now go through `BashEnvironment.local()`
  and `BashEnvironment.isolated()`; the raw constructor takes a runtime.
- SWE environment provenance is consistent across backends. The Apptainer
  environment now also reports `tools` and `patch_export`, and local runs also
  report `base_commit`. No previously recorded key changed name or value.
- Official SWE-bench grader-image verification runs in the grading process
  through its own Docker SDK instead of an embedded 133-line program executed by
  a separate `-I` interpreter. It still checks, before and after the grader
  subprocess, that every task's mutable `:latest` tag resolves to the exact image
  ID captured at generation, and still records the SDK version, module origin,
  and package root. Its `engine_contract` provenance value changes from
  `isolated-python-I:docker.from_env` to `in-process:docker.from_env`.
- `swebench_grader_source_identity` no longer asserts a separate 591-file /
  1,890,632-byte inventory. The pinned source digest already covers every path,
  size, and content hash; the counts remain grade provenance.

### Added

- ProgramBench adapter (`mini-agent eval --benchmark programbench`,
  `mini-agent grade --benchmark programbench`) pinned to
  `facebookresearch/ProgramBench` commit
  `963063c9271cc40fa179977356782ea4582e0b0c` / version `1.2.4`. Tasks load from
  the pinned checkout, the per-task `task_cleanroom_v6` image is bound by full
  image ID, and hidden `tests.json` is only ever hashed into provenance.
- Offline generation: the ProgramBench agent container runs with
  `--network none`, as upstream requires, and its whole `/workspace` tree is
  exported as `submission.tar.gz` into the official
  `<run>/<instance_id>/submission.tar.gz` layout. No score is assigned locally;
  `mini-agent grade` invokes the official `programbench eval` CLI.
- `DockerSWEEnvironment` options for that path — `network_disabled`, `workdir`,
  `require_git_baseline`, `benchmark_identity`, `max_archive_bytes` — plus
  `export_archive()` and archive-based state export/adoption (`SWEArchiveState`)
  for images without an inspectable Git baseline.

## [0.5.0] - 2026-08-12

First public release.

### Added

- Bounded provider retries with jittered backoff for 408/429/5xx and transport
  errors, honoring `Retry-After`; `--provider-retries`, `--provider-timeout`.
  Retry counts surface in trace events; usage is never double-charged.
- Bounded screenshot replay on the transcript-replay codecs
  (`--max-history-images`, default 4, `unlimited` opt-out), declared as a
  translation loss and recorded in provenance and manifests — makes long
  computer-use runs feasible on Chat Completions and Anthropic Messages.
- `mini-agent eval --progress`; run/eval print their output directory at start.
- Credential preflight: missing API-key environment variables fail before any
  output directory, container, or VM is created.
- Quickstart, `examples/` (offline library demo + JSONL corpus),
  `docs/library.md`, and an environment-variable reference.
- CLI-level end-to-end tests for every benchmark, including the osworld-v2
  worker path against a faked gated-data loader.
- `multi-mini-agent` companion distribution: a thin front door that installs
  `mini-agent` and runs the same CLI with multi-agent defaults.

### Changed

- Published on PyPI as `mini-agent-cmu` (the bare `mini-agent` name is
  blocked by similarity to an unrelated existing project); the import
  package `mini_agent` and the `mini-agent` console script are unchanged.
- `AgentSpecV1.bind()` is now the production path for every CLI run/eval agent
  construction, so manifest-recorded fingerprints are enforced, not
  descriptive.
- Translation reports carry each provider codec's declared losses instead of a
  hard-coded lossless claim.
- `run --environment swe` requires an explicit `--workspace` (the bash agent
  runs unsandboxed as the current user).
- Provider HTTP errors surface allowlisted `error.type`/`error.code` tokens
  only; free-form response text stays out of durable artifacts.
- Lint gate enforces line length (E501); grading, doctor, and the isolated
  grader probe moved out of the CLI into dedicated modules.

### Fixed

- SWE-bench grader source verification now asserts the pinned 591-file /
  1,890,632-byte inventory, not only the digest.
- The OSWorld per-step pause (0 s) is recorded in factory timing provenance.

## [0.4.0] - 2026-08-12

Internal milestone: chat-completions provider, browser state adoption, live
validation canaries, spec fingerprints, and the full-codebase audit that this
release builds on. Never published.

## [0.3.0 and earlier]

Internal rebuilds of the repository toward the minimal-loop design. Never
published.
