# Architecture

## The invariant

`src/mini_agent/agent.py` is the only agent loop:

```text
task + optional initial observation
              |
              v
      model(messages, tools)
         /              \
  final text          tool calls
      |                   |
 environment.finish   environment.execute
      |                   |
    result <---------- observations
```

`MiniAgent` knows nothing about shell commands, search engines, screenshots,
benchmarks, providers, graders, delegation policies, or storage layouts. It
validates the provider-neutral contracts, preserves one linear history, executes
every declared tool call in order, and stops on final text or an explicit limit.

## Boundaries

The repository has five layers:

1. `types.py`, `agent.py`: provider-neutral messages, tools, usage, and loop.
2. `providers.py`, `models.py`: request/response translation and continuation
   state for OpenAI Responses, OpenAI-compatible Chat Completions
   (full-transcript replay, selected with `--protocol chat-completions`),
   Anthropic Messages, and an explicit-endpoint Meta adapter.
3. `environments/`: one narrow model-facing tool per domain.
4. `orchestrator.py`: optional recursive scheduling through one communication
   tool, without a second agent class.
5. `benchmarks/`, `cli.py`, `grading.py`, `doctor.py`, `storage.py`: task
   loading, hidden evaluation, manifests, artifacts, official-grader
   orchestration, environment doctors, and operator controls. Shared
   micro-modules keep single concerns in one place: `_http.py` (bounded
   response reads), `_hash.py` (stable file hashing),
   `benchmarks/checkout.py` (Git checkout inspection), and
   `_grader_probe.py` (the isolated grader-runtime probe source, a real
   module so it sits under the lint/type gate).

This distinction matters. A provider codec is an inference harness component. An
evaluation adapter owns task data and scoring. Neither is a training method, and
no benchmark result implies reproduction of a training recipe.

## Environment contracts

### SWE

`BashEnvironment` exposes only `bash(command)`. Each command gets a fresh shell;
filesystem state persists. Neither direct nor multi-agent local execution is a
security boundary: both run as the current user with host filesystem and network
access. Direct mode edits the selected workspace.

State-isolated local workers receive private copies, private `HOME` directories, a Git
baseline, bounded process output, process-group timeouts, binary patch export,
and transactional descendant-state adoption. Escaping symlinks are rejected, but
the copy and private `HOME` do not restrict access to the rest of the host.

SWE-bench workers use either an independent persistent rootless Docker container
without host credential mounts or an independent Apptainer fakeroot overlay.
The resulting patch is captured before cleanup. Docker timeouts destroy the
container so a remote child cannot remain running.

### Web

`BrowserEnvironment` exposes one `browser` function. General web runs enable two
actions:

- `search`: returns bounded session references and snippets.
- `open`: accepts only a reference returned by that browser session.

Canonical BrowseComp-Plus generation exposes only `search`, matching the pinned
upstream runner's default search-only capability. This capability set is part of
the environment provenance and evaluation manifest.

Backends are live SerpAPI plus HTTP/Playwright page reads, pinned
BrowseComp-Plus Lucene BM25, and a deterministic JSONL BM25 fixture. Fixed-corpus
runs record the entire index tree hash, tokenizer revision, and tokenizer JSON
hash. Live readers bound content size and reject URLs resolving to loopback,
private, link-local, multicast, reserved, or unspecified addresses.

These checks reduce accidental SSRF. They do not turn Playwright or arbitrary web
content into a sandbox; see the security policy.

### Computer use

`CUAEnvironment` exposes one batched `computer` tool. Actions use native pixel
coordinates from the latest PNG observation. Reset, machine provisioning,
expected answers, shell access, snapshots, and evaluation remain outside the
model plane.

The environment tracks episode termination, sends `done` at most once without
transport retries, rejects actions after termination, and makes live state
single-claim during adoption.
OSWorld and cua-speed-run allocate one independent machine/environment lease per
agent. Their hidden checkers run only after agent execution.

## Shared accounting and evidence

Every maintained path uses one `RunContext` containing:

- a concurrency-safe global `BudgetLedger`;
- optional per-agent budgets configured when each worker starts; and
- a streaming, secret-redacting `TraceRecorder`.

The ledger charges calls before execution and usage/output after execution. A
call that crosses a resource limit remains charged. Unknown prices or incomplete
provider usage stop a cost- or token-bounded run because the remaining budget
cannot be proven. Global and per-agent wall time are checked before model and tool
operations.

Traces store hashes and sizes by default. Direct runs may opt into content capture;
benchmark evaluations reject that option. Evaluation manifests hash prompts,
hidden task data, configuration, limits, and referenced files. Resume requires an
identical manifest and restores paid accounting plus elapsed timing. A task is
resumable only after its result hash is atomically committed and both the file
bytes and containing-directory rename have been synced; official prediction
collectors also verify the artifact hash recorded in that committed result.
Uncommitted work is rerun only when its trace proves that no model or tool
operation started; otherwise resume fails closed because exact prior spend
cannot be reconstructed.
The trace file and its directory entry are synced when created, and every
`model_call_started` and `tool_call_started` record is flushed and synced before
the provider or environment request is issued, so this crash-recovery decision
is not based only on userspace buffers.

## Agent specification boundary

`AgentSpecV1` is the stable provider-neutral configuration contract. Its
canonical fingerprint binds the domain, model identifier, resolved prompt,
profile name, step cap, full shared budget, domain tools, and communication
capabilities. `TranslationReport` makes dropped, approximated, or unsupported
source fields explicit and can fail closed when an exact declared-field mapping
is required. Even a loss-free report is scoped to declared fields only and does
not assert equivalent provider behavior, policy training, timing, compaction, or
benchmark fidelity. `AgentSpecV1.from_json` verifies exported fingerprints, and
`spec.bind(...)` validates explicit model/domain identities, the declared
tool/action surface, and the budget before constructing the ordinary `MiniAgent`.
Binding is the CLI's production path: `mini-agent run` and every benchmark
adapter construct their agents — orchestrator children included — through
`spec.bind(...)` against the same spec recorded in the manifest, so recorded
fingerprints are enforced, not merely descriptive. Translation reports carry
the provider codec's declared losses (for example, the OpenAI-protocol codecs
drop the tool-result error flag and relocate tool-result images), so a report
is `exact` only when the selected codec declares none. Provider transport
settings and benchmark assets are deliberately outside this portable contract.

The portable domain-tool surface is name-level (`bash`, `browser`, or
`computer`); only the universal `agent` tool also has a portable action enum.
Domain-specific action schemas can vary by benchmark (for example canonical
BrowseComp-Plus is search-only), so evaluation manifests bind that adapter
surface and every model-call trace hashes the complete actual tool definitions.
This keeps the interchange contract small without losing execution evidence.

## Minimal multi-agent layer

`CommunicationEnvironment` adds exactly one model-facing function, `agent`, with
six actions:

```text
spawn(task)
send(agent_id, message)
inbox(wait=false)
wait(agent_ids?)
stop(agent_id)
adopt(agent_id)
```

Agent IDs encode ancestry (`/root/1/2`), but there is no hard-coded depth. Only a
descendant can be waited on or adopted. Adoption is explicit and only works when
the domain can export a compatible state: SWE agents adopt patches, computer
agents adopt pool-owned live sessions, and web agents adopt the descendant's
discovered references (the right to `open` them against the same backend). A
child's textual answer is delivered through the mailbox; it does not silently
replace the root submission.
`wait` returns when the first requested running descendant stops, so callers can
inspect statuses and wait again without stalling on every branch.
`inbox(wait=true)` blocks within the existing tool timeout until a message arrives.
`stop` is limited to the caller's descendant subtree and awaits cancellation and
environment cleanup before reporting every affected status.

All workers share global accounting. Limits cap simultaneously active agents,
total agents ever created, message size, inbox size, model concurrency, calls,
tool use, bytes, time, and optionally tokens/cost. Resource identities must be
unique, start/cleanup failures are terminal evidence, and remaining descendants
are cancelled when the root stops.

The scheduler deliberately has no role graph, planner class, debate protocol,
selector, or topology-specific worker. Parallel child sampling is not best-of-N
until a selector exists, and recursive delegation is not claimed to reproduce
any published recursive-agent method's policy training.

## Benchmark boundary

Benchmark modules may:

- load and fingerprint agent-visible tasks;
- create isolated environment leases;
- run `MiniAgent` or `Orchestrator` with the shared context;
- export official prediction/run schemas; and
- invoke or adapt a hidden upstream evaluator after the agent finishes.

They may not leak answers, qrels, evaluator configuration, or verifier methods to
the model. Generation and official grading remain separate commands where the
upstream project has that split. Detailed fidelity and pins are in
[benchmarks.md](benchmarks.md).
