# Multi-agent harnesses

A harness is a named, selectable topology. The point is comparability: hold the
model, the benchmark, and the budget fixed, change only the harness, and the
difference is attributable to coordination rather than to anything else.

Select one with `--harness`. `--team-size` applies only to harnesses that take
one.

| Harness | Agents | Lead has task tools | Spawning | Team formed |
|---|---|---|---|---|
| `single` | 1 | yes | — | — |
| `recursive` | any | yes | non-blocking, any depth | by the model |
| `fixed-team` | 3, 5, or 10 | yes | none | up front |
| `orchestrator` | 1 + subagents | **no** | blocking | on demand |
| `async-subagents` | 1 + subagents | yes | non-blocking, long-lived | on demand |

## What each one is

**`single`** — one agent, no `agent` tool at all. The baseline.

**`recursive`** — the free-form mesh that predates this layer, kept so
`--multi-agent` and every manifest recorded before harnesses existed stay
reproducible. Every agent holds every action and may delegate to any depth. It
is deliberately *not* one of the four studied topologies; it is the control for
"what the harness used to be".

**`fixed-team`** — N peers share one task. All of them see the task text
verbatim, all hold identical tools, and all are running before the lead takes
its first turn. They exchange `send`/`inbox` messages; none of them can spawn.
The lead alone submits the answer.

**`orchestrator`** — one coordinator with **no domain tools whatsoever**: its
only action is `delegate`, which spawns a subagent, blocks, and returns that
subagent's answer as the tool result. All work happens in subagents, each of
which gets the benchmark's full tool set.

**`async-subagents`** — a lead that keeps its own task tools and spawns
long-lived subagents. Spawning returns immediately. A subagent sees only what
its lead told it — not the original task — reports its answer as a message, and
then idles until the lead sends new instructions. `release` retires an idle
subagent cleanly so the lead can adopt its workspace.

## Adding one

One new module in `src/mini_agent/harnesses/`, plus its name in the `_MODULES`
tuple in that package's `__init__.py`. Nothing in the scheduler, the benchmark
adapters, or the CLI changes. A harness declares `Role`s — which actions the
role holds, whether it sees the domain's tools, its prompt suffix, and whether
it idles after answering — and a `Harness` binding them to a lead.

The registry is an explicit list rather than a directory scan because the
selected harness name is recorded in every manifest; which topologies exist
must not depend on what happens to be on disk.

## Interpretations, stated because they are choices

The design these implement leaves some things unsaid. Where behaviour had to be
chosen, it was chosen once, visibly:

- **Peers idle instead of terminating.** A terminal agent cannot be messaged, so
  a team whose members finished would silently shrink to one. `fixed-team`
  peers therefore stay available after answering.
- **Blocking delegation is its own action.** `delegate` is not a flag on
  `spawn`, because two roles that declare the same capabilities must mean the
  same thing — otherwise the capability list recorded in a manifest describes an
  agent that never existed.
- **`release` is not in the source design.** It exists because `adopt` requires
  a completed agent and `stop` cancels one, and a cancelled agent exports no
  state. Without it an async lead could never take back a subagent's workspace.
- **`fixed-team` has no `adopt`.** The described capability list is Send and
  Wait-for-Message only. On SWE-style benchmarks that means peers can transfer
  findings as text but not files; the lead's own workspace is the submission.
  ProgramBench is the exception: `--agent-git-share` gives the team a shared
  bare repository to push and pull through, with no network.

## A consequence worth knowing before you run one

The submission is always the lead's own workspace. On file-producing benchmarks
that interacts with each topology differently, and `orchestrator` is the sharp
case: its coordinator has no task tools by construction, so its workspace is
never edited and it produces an **empty patch** unless it explicitly adopts a
subagent's state. That is the topology behaving as described, not a defect —
but it means `orchestrator` on SWE-bench measures delegation, not patch
quality, unless adoption is part of what you are testing. Verified directly: a
real two-container orchestrator run completes with a 0-byte patch while its
subagent's own container holds the edit.

## Capacity

Team size interacts with three separate limits, and getting it wrong is a
measurement hazard rather than a crash:

- `--max-active-agents` is admission control, not a queue. A team larger than it
  fails outright rather than degrading.
- `--max-total-agents` bounds agents ever created.
- `BudgetLimits.max_concurrency` is one run-wide semaphore over model calls. A
  ten-agent team under the default of four still produces correct output — it
  just serializes, silently, and every latency number from that run is
  meaningless.

Because of the third, capacity shortfalls are reported before a run starts
rather than repaired automatically: `max_concurrency` is part of the recorded
agent spec, so raising it silently would make recorded fingerprints a function
of team size.

Team size also does not mean the same thing on every benchmark. OSWorld leases a
full desktop VM per agent, so a ten-peer team is ten VMs.
