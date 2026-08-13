# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/) with a 0.x compatibility caveat.

## [Unreleased]

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
- `cua-speed-run` step-budget exhaustion now runs the hidden checker
  (`finish_reason: step_budget`), matching OSWorld semantics.
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
