# Validation evidence

This page records validation performed through 2026-08-12. The compact,
machine-readable companion is
[`validation-record.json`](validation-record.json). Image acquisition and
provenance details are in [`machine-images.md`](machine-images.md).

Evidence on this page has three deliberately separate scopes:

- **frozen-source gate**: repeatable without a paid model or benchmark service;
- **reference-pass canary**: real environment plumbing executed before the
  final source freeze, normally with a deterministic scripted responder; and
- **operator narrative**: an earlier manual run that was useful during
  development but is not a current release result.

A source-ready doctor result is not a machine boot, a machine boot is not a
task run, and a task plumbing canary is not benchmark-quality evidence. No
result below is presented as a release winner.

"The validation node" below is the single shared-cluster Linux host this
evidence was gathered on (local NVMe scratch, small durable NFS quota, no
Docker); none of its constraints apply to your machine.

Heavy assets and disposable VM state live on node-local NVMe under
`/tmp/mini-agent`; they must be reacquired after node reassignment. Small
provenance records may be retained under a user-owned durable data directory.

## Deterministic gates

The frozen source passed **360 tests on each of CPython 3.10.20 and 3.12.5**,
the two interpreters present on the audit node; the four-interpreter
3.10–3.13 matrix remains enforced in CI. The suite covers full CLI evaluation paths for every benchmark
(argparse through worker, spec binding, agent loop, and artifact
contracts, with faked model transports and machine planes); the agent
loop invariants; provider codecs and bounded transport; shared budgets and trace
recording; scheduling, messaging, stopping, invalid actions, and state
adoption; agent-spec binding enforcement; SWE isolation; bounded web and
remote I/O; computer episode lifecycle; exact source and image identities;
hidden evaluator boundaries; resume contracts; and official artifact schemas.

This snapshot is the full-codebase audit/cleanup pass over the previously
attested 27-file snapshot
(`9b0d0ceb6e640fb195e767523eeb39c2a46b2b1f0d2a999da8f58ac2d819665a`): agent
construction on every CLI run/eval path now goes through `AgentSpecV1.bind`,
translation reports carry the provider codecs' declared losses, the lint gate
enforces line length, dead code was removed, and the grading, doctor, and
grader-probe subsystems moved out of the CLI into their own modules.

The location-independent harness identity for this source snapshot covers 32
package source files and is
`a5506cfbaced8bfcb834c86a08794a1e03bbd5eaab670cbff3534f4e2ee08c66`.

The required commands pass:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m compileall -q src tests
python3 -m ruff check src tests
python3 -m mypy src/mini_agent
PYTHONPATH=src python3 -m mini_agent profile \
  --application web --profile default --model openai/test-model
```

The test command was run with the installed 3.10 and 3.12 interpreters rather
than the node's unqualified `python3`, which is Python 3.9 and below the
declared Python 3.10 minimum. The latest static pass used Ruff 0.15.20 with
`E501` enforced at 88 columns (per-file ignores cover only byte-exact upstream
prompt literals and embedded runtime-script literals); compilation, Mypy, and
the required profile command also passed. A local `python3 -m build` under
`umask 0022` passed `twine check --strict`, and the wheel installed cleanly
into a fresh CPython 3.12 venv with the console script, grading/doctor
modules, and the packaged grader-probe resource verified; artifact hashes are
in the validation record, and the checked artifacts were then removed.

## Reference-pass canaries

These canaries exercise real repository-owned environments without making a
paid model call. They were run during the reference audit, before the final
source freeze. The computer evaluations bind their exact pre-freeze harness
identity in saved manifests; the web artifacts bind their exact retrieval
inputs and trace bytes. Later source hardening — including the audit/cleanup pass —
is covered by the deterministic 360-test gate,
not by retroactively relabelling these as final-source E2E runs. Scores of zero
below are deliberate and say nothing about agent quality.

| Canary | Result | What was exercised |
|---|---|---|
| SWE-bench generation | pass | official `astropy__astropy-12907` image through Apptainer 1.4.0; `testbed` Conda activation; 3 model calls, 2 `bash` calls, an agent-created Git commit, and a 566-byte exported patch after that commit; image SHA `3c2e9eb1…`, patch SHA `4d1ded2e…`; generation only, not an official score |
| BrowseComp-Plus fixed retrieval, single | pass | exact 2.17 GB Lucene index, Anserini 1.1.1 and Qwen tokenizer; one search returned the canonical five 512-token snippets and official run schema; 2 model calls and 1 tool call; trace SHA `b517ef8d…`; no judge score |
| BrowseComp-Plus fixed retrieval, two agents | pass | the same real Lucene backend plus child spawn, message/read/wait and result production; 2 completed agents, 6 model calls and 4 tool calls; trace SHA `696cde45…`; no judge score |
| Chromium page reader | pass | real Chromium opened `https://example.com`, checked the successful response and final public URL, extracted bounded title/text, and cleaned up; transport smoke only |
| cua-speed-run, single | pass in 24.26 s | exact upstream checkout and submodule, real QEMU/KVM guest, 1600×900 observation, one deliberate wait action, hidden checker, artifact commit, and cleanup; 2 model calls and 1 tool call; score 0; manifest SHA `0f919dc5…`, harness `28c22d61…` |
| OSWorld v1, single | pass in 151.15 s | exact checkout, real official guest, Apptainer compatibility client, official reset/setup/action/evaluate lifecycle including the reference 60-second initial settle, fresh observation, and 20-second pre-evaluation settle, artifact commit, and cleanup; 2 model calls and 1 tool call; score 0; manifest SHA `eb1240eb…`, harness `69a9393c…` |
| cua-speed-run, two agents | pass in 25.23 s | two independent QEMU/KVM guests; child spawn, mailbox delivery, wait, explicit live-session adoption, post-adoption action, hidden checker on the adopted child session, and cleanup; 7 model calls and 5 tool calls; score 0; manifest SHA `3d422efa…`, harness `42039442…` |
| OSWorld v2 boot | machine/controller pass only | exact tagged checkout and v2 image booted through the pinned runtime; the controller reported Linux and returned a valid 1920×1080 screenshot; no task reset, action, evaluation, or score was attempted |

The successful two-agent trace contains exactly two `agent_spawned`, two
`agent_completed`, two `message_sent`, and one `agent_state_adopted` event. The
two message events are the child's explicit message to the root and its
automatically delivered terminal result.

OSWorld v2's task-class dataset is separately gated. The node received HTTP
401 when acquiring it, so the adapter correctly refuses to replace it with v1
or synthetic tasks. The v2 result proves only image boot and controller
connectivity.

## Machine-image evidence

The full acquisition commands and provenance contract are documented in
[`machine-images.md`](machine-images.md). The bytes validated on this node are:

| Input | Size | SHA-256 / source identity |
|---|---:|---|
| OSWorld v1 official archive | 12,273,896,463 bytes | `b795b6cd4c69b252c1b4f10150a347795555032501b60fd031751ed09b896712` |
| OSWorld v1 extracted qcow2 | 24,460,197,888 bytes | `6bf667a852b3c307f61d9f09c42559351f45e0607e428b4997becf534cf4d313` |
| OSWorld runtime SIF | 80,822,272 bytes | `8a9ee8e99bae986e3b7633419bce2043906c9892bd555f4765b7fa6b2adebccd`; source OCI manifest `sha256:0e6497a9295647cf05bf2b2af522fdd79bdeba2737595259cab310a3bcf6baa9` |
| OSWorld v2 official archive | 14,189,763,267 bytes | `eb737ae70b49849e24af407de6a518439a23de05a8497096a948334ce0a909aa` |
| OSWorld v2 extracted qcow2 | 27,402,633,216 bytes | `3d632f031459583cf936e0c4c5bb939122df0fec85aecb0d044ef2d3e5863335` |
| cua-speed-run node build | 6,701,187,072 bytes | `a9d802e021ac883fc4877bf334aaef48505502d23ac51b097e39ed7b3bb8d85f` |

Both extracted OSWorld images passed `qemu-img check`. The cua-speed-run image
was produced by the pinned upstream builder; because that builder consumes
mutable network inputs, its hash is an observed node-build identity rather than
a universal upstream release hash.

## Earlier operator narrative

The following observations predate the final source pass or require
credentials/infrastructure not presently available. They remain useful for
development history, but are **not current release scores**, are not included
as deterministic gate results, and should not be used for model comparisons.

| Boundary | Earlier observation | Current limitation |
|---|---|---|
| credentialed provider adapter | direct SWE, fixture web, and recursive SWE runs completed against an OpenAI-compatible deployment; the recursive run exercised child messaging and patch adoption | not rerun after final changes; no model credential is available for a fresh run; private endpoint, model, session-header, and credential details are intentionally not recorded |
| SWE-bench Verified, single | a paid-model patch for one official instance passed its `FAIL_TO_PASS` test inside the task image | the official 4.1.0 scoring harness was not run because this node has no Docker; this is operator narrative, not an official score |
| SWE-bench Verified, multi-agent mode | the same instance completed through the multi-agent harness and its exported patch passed the target test; the model did not spawn a child | no official Docker grading and no release score; live child communication is evidenced separately by the reference-pass cua-speed-run canary |
| BrowseComp-Plus, paid multi-agent mode | a root plus three children used the real Lucene backend and adopted a child's browser state, but the team hit its 3600-second wall clock | incomplete run; no official score and no claim of superiority |
| BrowseComp live web | no run | `SERPAPI_API_KEY` is not configured; the fixture web canary is protocol evidence only |

## Multi-agent surface

Multi-agent mode adds one model-facing tool, `agent`, with six actions:
`spawn`, `send`, `inbox`, `wait`, `stop`, and `adopt`. The benchmark runners
accept the same mode, while direct `run --environment computer` correctly
refuses it because that path has no isolated machine factory. State adoption is
domain-specific: SWE patches, browser reference sets with backend-identity
checks, and pool-owned live computer sessions.

Deterministic tests cover blocking inbox delivery, direct messaging, delivery ordering,
terminal-recipient rejection, descendant-only subtree stopping, capacity
release after cleanup, message byte/hash trace identity, invalid actions, and
ordered state-adoption history. A depth-12 recursion test proves there is no
hard-coded topology depth; active and total agent limits remain configurable
resource budgets. The reference two-machine canary supplies real
VM evidence for spawn/message/wait/adopt; it is still a topology and plumbing
test, not evidence of improved benchmark performance.

## Explicitly unvalidated boundaries

- No fresh paid-provider run was made for the final source pass.
- No SerpAPI credential is configured, so BrowseComp live search was not run.
- Docker is unavailable, so no official SWE-bench score was produced.
- No suitable GPU/vLLM Qwen3-32B judge was available, so no official
  BrowseComp-Plus score was produced.
- OSWorld v2 task data is gated and returned HTTP 401; no v2 task or score was
  produced.
- OSWorld v2 multi-phase tasks and `ASK_USER`/user-simulator tasks are rejected
  before machine or model startup because the minimal protocol does not yet
  implement those official lifecycles.
- The maintained in-process cua-speed-run adapter preserves official clock,
  grace, and checker semantics, but cannot safely reproduce the official
  executor's separate-process 90-second environment and 300-second verifier
  watchdogs. Operators requiring those watchdog guarantees must use the
  upstream executor topology.
- The default Gym-Anything QEMU runtime reference is mutable upstream; the run
  manifest records the effective reference, and comparable runs should pin a
  local SIF or digest-qualified OCI image.
- No full benchmark suite or statistically meaningful model comparison was
  run.
- Real environment canaries predate the final source freeze; the final source
  changes were validated by deterministic gates rather than a second full VM,
  index, or paid-provider pass.

## Credential hygiene

No credential value, private provider endpoint, private deployment model name,
session-header value, or transcript history is retained in this repository or
its validation record. Provider secrets are accepted through environment
variables, credential-shaped custom body/header names are rejected, remote
bodies are bounded, and trace redaction is covered by deterministic tests.
The earlier credentialed-provider observations above are self-reported operator
narrative and are not a reproducible current release artifact.
