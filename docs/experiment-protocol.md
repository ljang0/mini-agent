# Controlled experiment protocol

Status: draft. No release candidate has been selected.

## Questions

The first experiment should isolate six mechanisms:

1. Does decomposition help beyond equal-cost single-agent inference?
2. Does persistent worker context outperform fresh blocking workers?
3. Does asynchronous scheduling remove enough barrier latency to justify extra
   coordination?
4. Does dynamic DAG replanning beat a fixed initial decomposition?
5. Does explicit candidate selection improve accuracy over independent fan-out?
6. Does external-context recursion help only on long-context tasks, and at what
   depth does it become wasteful?

## Candidate labels

Use these names in every result artifact:

- `single`
- `flat_parallel` (throughput workload only)
- `parallel_best_of_n`
- `blocking_orchestrator`
- `fixed_agent_team`
- `async_subagents`
- `macu_dynamic_dag`
- `recursive_delegation` (generic recursion control; not RAO or RAH)
- `rlm_repl` (restricted clean-room RLM algorithm implementation)
- `platoon_recursive_inference` (recursive inference only; not RAO training)
- `external_context_json_search` (non-RLM ablation)
- `anthropic_managed_agents` (hosted session boundary)
- `prime_agent`
- `grok_build`
- `xai_hosted_multi_agent` (separate four- and sixteen-agent variants)
- `openai_hosted_multi_agent`

Do not put `flat_parallel` on the same accuracy leaderboard as best-of-N: it runs
different tasks and has no final-answer selector.

## Task strata

A release experiment needs enough tasks in each stratum to expose topology effects:

| Stratum | Defining property | Expected winner hypothesis |
| --- | --- | --- |
| Serial | Every step consumes the prior result | Single or recursive; parallel overhead may hurt |
| Decomposable research | Independent evidence branches, final reconciliation | Async, fixed team, or MACU |
| Partial observability | Downstream work needs observations/files that cannot be rediscovered | MACU |
| Long external context | Relevant slices are sparse and context exceeds the normal window | `rlm_repl`/upstream RLM/Prime Agent; JSON-search ablation as control |
| Diverse solution search | Multiple plausible answers, objective verifier available | Best-of-N |
| Coding integration | Parallel components plus merge/test conflicts | Fixed team or async with isolated worktrees |
| Tool-heavy web | Independent searches with slow external calls | Async or flat parallel for batch throughput |

Tasks must be natural user requests with observable completion boundaries. Difficulty
must come from useful dependent work, not dead sites, verbosity, or arbitrary quotas.
For mutable web state, record an observation timestamp and label blocked sources as
access shortfalls rather than evidence of absence.

## Two budget regimes

One budget cannot answer both the quality and latency questions. Run both in the
eventual release protocol.

These are target regimes, not both current capabilities. `max_model_calls` counts
Scaffold Lab backend invocations. One hosted OpenAI/xAI response, Anthropic Managed
session, or Prime/Grok external session can contain many internal calls. Local
concurrency, depth, turn, and tool limits do not constrain such provider-owned trees.
Aggregate-compute comparisons are valid only when `usage_complete` and `cost_known`
are both true and the candidates use commensurate cost sources. Prime and Grok always
fail those completeness checks today and are therefore ineligible for equal-total-
compute claims. The hosted APIs should also be analyzed separately from in-process
harnesses because their schedulers and intermediate state are closed.

Scaffold Lab token/cost enforcement is post-response, so one outer call can overshoot
a cap. When a backend reports incomplete accounting, hard resource caps fail closed
before another call. Anthropic's `--managed-budget-cents` is a separate provider-side,
session-wide public-list-cost cap: it pauses threads between model requests, so the
crossing in-flight request can still leave the final cost slightly above the cap. The
adapter records `budget_reached` as an incomplete/error trial and does not automatically
raise the cap or resume the session. The
Prime/Grok example configs intentionally omit token and dollar caps because their one
outer session cannot be stopped at a verified complete inner-tree boundary. Per-agent
token/context limits are not implemented.

### A. Equal total compute

- Same pinned model snapshot and effort.
- Same total input + output token cap across the complete agent tree.
- Same cost basis, dollar cap, and tool-call charges.
- Same tool-call and tool-output-byte caps only where the complete tree is observable
  and those caps are actually enforced at that tree boundary.
- Same wall-time cap and environment snapshot.

This measures whether orchestration spends a fixed budget better.

### B. Equal per-agent compute

- Same per-agent context/output/step cap.
- Agent count is allowed to scale total tokens.
- Report total cost and tokens without normalization.

This measures the latency-quality frontier from scale-out, matching the spirit of the
Fable system-card comparison.

## Experimental controls

- Pin model IDs, provider API versions, prompts, tool schemas, browser/container/VM
  images, package locks, and source commits.
- For hosted configurations, export and hash the resolved agent/model, prompt, tool,
  roster, permissions, and environment definitions. An agent or environment ID alone
  is not a reproducible snapshot.
- Reset every browser, repository, and computer environment per trial. Record whether
  logical agents share state or receive isolated sessions/workspace copies.
- Keep run artifacts outside every model-visible source workspace; matrix preflight
  rejects overlap before writing the manifest. Use a clean frozen source as well,
  because unfiltered copy mode can still expose older evaluator data or artifacts
  already present there.
- Use copy mode for matrices. Direct SWE mode mutates one shared source and is accepted
  only for a single planned trial.
- Freeze a task contract and evaluator before observing candidate outputs.
- Randomize harness order with a recorded seed.
- Run at least three repeats for deterministic-looking tasks and more for high
  sampling variance; report numerator and denominator.
- Keep development and release holdouts separate. Any prompt or harness change
  invalidates prior measurements for that variant.
- Preserve every observable trace, error, model/tool usage record, exact config hash,
  transport mode, and accounting source; explicitly label provider-hidden events.
- Run all pending calls to completion or explicitly mark cancellations; do not hide
  failed or timed-out children.
- Run each Prime/Grok repeat as a separate invocation with a newly provisioned
  workspace and output directory. Combine artifacts only after every trial finishes.

## Metrics

Primary:

- task success against deterministic or frozen rubric evaluators;
- full-tree tokens and comparable recorded cost when accounting is complete; otherwise,
  explicitly named observed lower bounds. Distinguish provider-returned cost,
  public-list-price `list_cost`, and local token-price estimates from invoice charges;
- raw wall-clock latency;
- union of backend-active intervals (not a causal critical path);
- error, timeout, deadlock, and invalid-action rates.

Diagnostic:

- agents created, peak concurrency, recursion depth, DAG replans, tool calls, and
  tool-output bytes;
- duplicate work, conflicting reports, and unconsumed worker results;
- coordination-token share and context re-establishment cost;
- verifier false-selection rate for best-of-N;
- contamination/fabrication and honest-partial rates.

The current `backend_active_union_seconds` is the union of real backend-call
intervals. It is neither a causal critical path nor Anthropic's token-rate-normalized
metric.

## Release gate

A candidate becomes `RELEASE_READY` only after:

1. the release holdout is frozen and feasible;
2. internal QA and human trace review pass;
3. its task-success interval improves over `single` in at least one target stratum;
4. the improvement survives an equal-total-compute comparison;
5. cost, latency, and operational-failure tradeoffs are explicit;
6. every fidelity claim matches the source audit;
7. install, run, resume, and failure-recovery instructions work from a clean checkout.

The best result may be a router rather than one universal harness: use single-agent
execution for serial/small tasks and escalate only when a decomposition or context
diagnostic predicts benefit. That router must be evaluated as its own candidate and
charged for routing overhead.
