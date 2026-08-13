# Benchmark fidelity

`mini-agent` adapts each evaluation environment to one minimal agent protocol. It
does not claim that this protocol reproduces an upstream provider-specific model
harness. Pins are enforced at runtime where an upstream checkout is required.

| Benchmark | Pinned boundary | Maintained path | Deliberate difference |
|---|---|---|---|
| SWE-bench | tag `v4.1.0`, commit `726c5461e2ef52d83cf1ea2107870a8bb3328d57` | Official task fields, independent persistent Docker or Apptainer workspaces, binary patch artifact, exact prediction JSONL, official grader command | Generic one-`bash` loop and baseline prompt are not an official leaderboard model harness; upstream scoring remains Docker-only |
| ProgramBench | commit `963063c9271cc40fa179977356782ea4582e0b0c`, package version `1.2.4` | Pinned `task.yaml` task loading, per-task `task_cleanroom_v6` cleanroom image, `--network none` agent container, `<run>/<instance_id>/submission.tar.gz` layout, official `programbench eval` command | Generic one-`bash` loop and baseline prompt are not the upstream mini-swe-agent baseline; `tests.json` is never loaded into task data, only hashed |
| BrowseComp | `openai/simple-evals` commit `652c89d0ca9df547706735883097e9537d40dc47` | Official encrypted CSV decoding, live SerpAPI search/open, required independent private-answer grader | Search provider/tool protocol is a maintained baseline, not simple-evals' hosted model harness |
| BrowseComp-Plus | `texttron/BrowseComp-Plus` commit `046949032b0328319cc9a02663a759ec601d9402` | Fixed Lucene BM25, exactly top five, 512-token snippets, search-only capability, full index-tree hash, tokenizer provenance, official per-query run schema and pinned grader command | One `browser` action renames the upstream provider-specific search function |
| OSWorld v1 | `xlang-ai/OSWorld` commit `091f5ef1d5544bc74953c77875d5feb5bed30108` | Official task configs, reset/setup, 60-second reset settle plus fresh observation, independent `DesktopEnv`, pyautogui and literal `FAIL` steps, 20-second evaluation settle, hidden `evaluate()` | Batched native-pixel computer tool and baseline prompt are adapted |
| OSWorld v2 | tag `v2026.06.24`, commit `2b9b7b4eb73243d557bdbf2998fe18d8e18e19c6` | Standard single-phase lifecycle, including the 60-second reset settle and immediate final evaluation; GitLab excluded by default | Gated release data is unavailable on this node; multi-phase and user-simulator tasks fail closed instead of receiving an invalid single-loop score |
| cua-speed-run | commit `7230223cbc57df68331cad32889adf01f3601651`; gym-anything gitlink `70d9e51d2517049d995cc820a319a355c3c6e979` | Pinned generation, one backend instance per evaluation, preparation off-clock, independent leases, runtime-input revalidation, timeout/grace, hidden checker, 0–100 score | Runs in process against the environment plane, without the official process-isolated gateway watchdogs or two-file submission/executor protocol |

The minimal loop itself records the design reference to
`SWE-agent/mini-swe-agent` commit
`a83fcae82d2a08f0ee0c688f9d137b3566c097f8`; this is a design reference, not a
claim of line-for-line or prompt-for-prompt equivalence.

## SWE-bench

Input is JSONL containing at least `instance_id` and `problem_statement`, plus the
upstream image metadata used by SWE-bench. Docker mode requires a functioning
rootless Docker client. No host credentials or workspace are mounted. Apptainer
mode requires fakeroot overlay support and enough local scratch for one private
overlay per agent. Before the evaluation manifest is committed, every selected
Docker tag is resolved to a full image ID and every Apptainer SIF is materialized
and hashed. Docker starts by that image ID, never by the mutable tag; Apptainer
re-hashes the selected SIF before use. Those task-to-image bindings are part of
the manifest, so resume fails closed if image bytes change. Source OCI images are
cached under durable assets.

Each environment verifies a clean Git worktree, records the image's actual
40-character `HEAD`, and, when the dataset supplies `base_commit`, proves that it
is an ancestor of that image baseline. Official instance images may add a
committed environment-setup change on top of the task base commit. Shell calls
set `BASH_ENV=/root/.bashrc`, matching the pinned mini-swe-agent benchmark
configuration so the image's `testbed` Conda environment is active. Patch export
and multi-agent adoption always diff/reset against the captured image baseline,
even if an agent creates its own Git commits.

Generation produces a hash-bound `prediction.json` per instance and a sorted
`predictions.jsonl`. It does not run tests inside the model loop or assign a
benchmark score. `mini-agent grade --benchmark swebench` invokes
`swebench.harness.run_evaluation` from the exact installed 4.1.0 package. Grade
requires a local `.json` array or `.jsonl` dataset; a remote dataset name is
rejected because it cannot be bound to the generation manifest. The selected
rows' prompt and complete canonical data hashes must match that manifest before
any grader process starts. The v4.1.0 evaluator selects remote instance images by
the mutable `swebench/sweb.eval.x86_64.<instance>:latest` tag. Official grading
is therefore accepted only for a Docker-generated manifest whose image bindings
exactly cover its task list. Immediately before and after the grader subprocess,
each exact upstream tag must resolve through `docker.from_env()` to the full
Docker image ID captured before inference. Verification receives the exact
allowlisted environment used by upstream, including an explicit `DOCKER_HOST`,
so both calls address one engine. Those mappings and the Docker SDK version are
grade provenance. Be precise about what that verification is: it runs in the
grading process using this process's Docker SDK, not in a separate `-I`
interpreter as it did previously. It still proves the tag-to-image-ID mapping
against the recorded generation bindings, and it still binds the installed SDK's
version, module origin, and package root into grade provenance; it no longer
proves that a *distinct* interpreter with an independently resolved SDK observed
the same mapping. A grading process whose own `docker` import is already
compromised can no longer be caught by this check. The runtime argv stored in an evaluation artifact is never
executed and is retained only as inert generation provenance. Apptainer
generation remains useful for agent execution on this
node, but official grading of that run is rejected until OCI-to-SIF equivalence
is recorded; a SIF byte hash alone does not prove which mutable OCI manifest was
converted.

The `v4.1.0` release is not published as version 4.1.0 on PyPI. Install an exact
editable checkout of the pinned commit: a wheel built from the tag omits 500
tracked `swebench/resources` YAML files used by the evaluator and is rejected.
At grading time the adapter verifies the imported package tree against the
pinned source digest
`63d4d3d0543de66520fa44f12badddaa810f708a0d780954684c24c7ce075cc8` — a hash over
every relative path, size, and content hash in the tree — and rechecks it
immediately before and after the subprocess. The file count and total byte size
are still recorded as grade provenance, but they are no longer asserted
separately: the digest covers them, so a separate inventory assertion could only
ever fail alongside it. Upstream does not
publish a dependency lock for this tag, so installed dependency versions remain
runtime provenance rather than a claim of a fully locked environment.

Primary references are the pinned
[mini-swe-agent loop](https://github.com/SWE-agent/mini-swe-agent/blob/a83fcae82d2a08f0ee0c688f9d137b3566c097f8/src/minisweagent/agents/default.py),
[mini-swe-agent SWE-bench configuration](https://github.com/SWE-agent/mini-swe-agent/blob/a83fcae82d2a08f0ee0c688f9d137b3566c097f8/src/minisweagent/config/benchmarks/swebench.yaml),
and [SWE-bench evaluator](https://github.com/SWE-bench/SWE-bench/blob/726c5461e2ef52d83cf1ea2107870a8bb3328d57/swebench/harness/run_evaluation.py).

## ProgramBench

Input is a local checkout of `facebookresearch/ProgramBench` at commit
`963063c9271cc40fa179977356782ea4582e0b0c`, whose `pyproject.toml` must declare
version `1.2.4`. The checkout is verified before any task loads: exact `HEAD`,
no tracked modifications, and no untracked executable or source file that could
shadow evaluation. Tasks come from
`src/programbench/data/tasks/<instance_id>/task.yaml`, read through a strict
reader for the pinned metadata grammar; anything else fails closed.

Each task directory also holds `tests.json`, the evaluator's branch-to-test map.
Its bytes never enter task data. Only its SHA-256 and size are recorded as
hidden provenance, so the manifest binds which hidden tests existed without the
agent, the prompt, or the trace ever seeing a test name.

Inference uses the per-task Docker image
`programbench/<instance_id with '__' replaced by '_1776_'>:task_cleanroom_v6`.
As with SWE-bench, every selected tag is resolved to a full image ID before the
manifest is committed and the container starts by that ID. Upstream requires the
agent to have no internet during inference, so the agent container runs with
`--network none`; that is recorded in the environment provenance and in the
evaluation manifest as `agent_network: none`. The cleanroom images ship a
compiled reference program and documentation rather than a Git tree, so the
container is created without a Git baseline: patch export is unavailable and the
whole `/workspace` tree is the exported state. Multi-agent adoption therefore
replaces a descendant's workspace archive instead of applying a diff.

Generation writes one hash-bound `submission.tar.gz` per instance and collects
them into the exact upstream layout, `<run>/<instance_id>/submission.tar.gz`.
The agent's build contract is upstream's: the evaluator wipes `/workspace`,
unpacks the archive, deletes any shipped `./executable`, and runs `./compile.sh`
with the build network blocked, expecting the built program at `./executable`.
No score is assigned locally — `EvaluationOutcome.score` stays `None` and the
result records `scoring: official-programbench-eval-only`.

`mini-agent grade --benchmark programbench` requires the same pinned checkout
plus `programbench==1.2.4` in the current Python, verified through the isolated
grader-runtime probe. It re-derives every selected task from the checkout and
refuses to start unless the prompt and complete canonical data hashes match the
generation manifest. The collected submissions are snapshotted into the private
`0700` grade directory and the official CLI is invoked there with `--output`
pointing at a separate directory inside the grade output, so the snapshot the
manifest hashes is never mutated by the evaluator. The pinned package ships no
`__main__` module, so the isolated Python runs the entry point declared in the
upstream `pyproject.toml` (`programbench.cli.main:app`) directly. The official
evaluator pulls Docker images and downloads per-branch test blobs from
HuggingFace on demand; `HOME` is the private grade directory, so that cache
stays inside the grade evidence. Scores come only from those `.eval.json` files
and upstream's own `programbench info` aggregation.

Primary references are the pinned
[usage guide](https://github.com/facebookresearch/programbench/blob/963063c9271cc40fa179977356782ea4582e0b0c/docs/README.md),
[image naming](https://github.com/facebookresearch/programbench/blob/963063c9271cc40fa179977356782ea4582e0b0c/src/programbench/constants.py),
and [evaluator](https://github.com/facebookresearch/programbench/blob/963063c9271cc40fa179977356782ea4582e0b0c/src/programbench/eval/eval.py).

## BrowseComp

BrowseComp needs the official encrypted CSV, network egress, `SERPAPI_API_KEY`,
and a solver model. Scoring is inseparable from a model grader, so eval requires a
separate `--grader-model` configuration. Its endpoint, credential variable,
request body, generation limit, and prices are independent of the solver.

The pinned simple-evals code defines dataset decoding, prompt construction, and
grading; it does not define a browsing tool. The maintained live baseline uses
SerpAPI's Google organic results plus a bounded HTTP or Playwright page reader.
SerpAPI cache state, location, and the live web can change, so two runs with the
same manifest are not retrieval-reproducible.

The decryption, prompts, and `correct: yes|no` decision contract follow the
pinned simple-evals revision. That revision returns the entire regex match
(`correct: yes`) and then compares it with `yes`, so a matched grade satisfies
neither its `yes` nor its `no` comparison. This adapter deliberately returns the
capture group before comparison; that one-line defect correction is recorded
rather than described as byte-for-byte upstream execution.

Hidden answers and raw grader output are written under each task's `private/`
artifact directory and must not be published. Solver and grader calls share the
evaluation's global ledger and trace.

## BrowseComp-Plus

Canonical generation requires the official query TSV, downloaded Lucene index,
Java 21 or newer, the pinned Anserini 1.1.1 fat JAR, Pyjnius 1.6.1,
Hugging Face Hub 0.33.4, tokenizers 0.21.2, and a resolved tokenizer. The fixed
adapter rejects version drift from those direct lock entries. Upstream loads the
tokenizer through Transformers 4.53.2; this adapter loads the same pinned
`tokenizer.json` directly and preserves upstream encode/slice/decode behavior
without the model framework or NumPy. Like the pinned upstream project, the
fixed-web extra requires Python 3.10 or newer. Obtain the
queries through the pinned upstream checkout; its generated two-column TSV is
headerless, and the adapter rejects a header rather than silently evaluating it
as a bogus query. Obtain the pre-built index from
[`Tevatron/browsecomp-plus-indexes`](https://huggingface.co/datasets/Tevatron/browsecomp-plus-indexes),
at repository revision `b3f37f70c33829eb09d04784a54277a31871fd63`, then
pin the downloaded bytes with `--index-sha256`; the repository and revision are
bound in the evaluation manifest, the observed local tree hash is stored before
generation, and the complete index is re-hashed after all workers finish. Any
mutation aborts before official-run collection. Upstream names but does not pin
the `Qwen/Qwen3-0.6B` tokenizer revision, so this adapter requires a full
40-character `--snippet-tokenizer-revision` and records the downloaded JSON hash;
that makes each local run reproducible without claiming to recover an unstated
historical revision.

The upstream lock uses Pyserini 1.2.0, whose wheel contains
`anserini-1.1.1-fatjar.jar`. Extract that wheel without installing its unrelated
dense/server dependencies:

```bash
python -m pip download --no-deps --dest /tmp pyserini==1.2.0
python -m zipfile -e /tmp/pyserini-1.2.0-py3-none-any.whl /path/to/assets/pyserini
```

Pass the extracted JAR with `--anserini-jar`. The adapter requires SHA-256
`69270ba4d160826953347411ce5d7e205ce363766a9cc72ac9da3b945341af83` and
records it in provenance. Updating the JAR, JNI bridge, or tokenizer is a
benchmark-adapter change, not routine dependency drift.

The JSONL backend accepted by direct `run` is only a deterministic small-corpus
fixture. It is not BrowseComp-Plus evaluation evidence.

The pinned upstream OpenAI runner defaults to
`QUERY_TEMPLATE_NO_GET_DOCUMENT`, `--k 5`, and `--snippet-max-tokens 512`; its
`--get-document` flag is opt-in. Canonical `mini-agent eval --benchmark
browsecomp-plus` therefore exposes only `action=search` and rejects other top-k,
snippet, or tokenizer choices. It preserves all five token-bounded snippets instead of
applying the general browser's additional whole-observation character cap; the
shared tool-output byte ledger remains authoritative. Direct/general web runs may
still expose `open`. The action set is bound into environment provenance and the
evaluation manifest; the upstream template name is bound into each per-query
artifact and the manifest.

Official grading requires the exact clean checkout, ground truth, qrel evidence,
the upstream Python dependencies, and the selected local judge-model snapshot.
The model directory must be local, materialized, and symlink-free. The command
enforces the lock's direct grader versions (`numpy==1.26.4`, `tqdm==4.67.1`, and
`vllm==0.9.0.1`) and records the pinned `uv.lock` hash; this is a runtime identity
check, not a claim that another environment with merely compatible dependency
ranges is equivalent. `grade` constructs the pinned upstream evaluator command;
it does not normalize its scores. Untracked executable/source files, including
Python bytecode caches, are rejected in the checkout, and the grader process is
run with bytecode writes disabled.

For both official graders, `manifest.json` must be a valid
`mini-agent-eval-v2` manifest whose fingerprint, benchmark, task IDs, and visible
prompt hashes agree with the collected completed artifacts. Before execution,
the CLI copies the exact predictions/runs and local dataset or hidden answer data
under `<grade>/inputs/` and verifies that each copy retained the source content
hash. Those snapshot hashes are checked again after the grader exits.
BrowseComp-Plus's `--eval-dir`, when supplied, must remain below the grade
directory. The directory is `0700`; its files, captured grader streams, and
grader-produced artifacts are hardened to `0600`. `manifest.json` records source
and snapshot identities, runtime/package pins, upstream source/lock identities,
the judge-model tree hash where applicable, and exact argv. The pinned checkout,
grader script, lockfile, command, and full local judge-model tree are revalidated
immediately before and after BrowseComp-Plus grading; a change leaves the grade
uncommitted. Both Python graders run in isolated mode under an explicit,
benchmark-specific environment allowlist. Solver/provider, SerpAPI, cloud, and
computer-gateway credentials are not inherited; `HOME` is the private grade
directory, bytecode writes are disabled, and `PYTHONPATH`/`PYTHONHOME` are absent.
The manifest hashes every supplied environment value without publishing it.
`result.json` records the return code plus hashes of the
self-fingerprinted grade manifest, stdout, stderr, verified grader assets, and
official output files; `completed.json` commits the exact result bytes. Treat the
entire grade directory as benchmark-private because it contains hidden answers
and judge output.

## OSWorld v1 and v2

The checkout must be clean and at the exact commit. That identity is rechecked
before every desktop lease and after evaluation and cleanup, before a score is
committed. OSWorld also requires its
official task assets, VM/container images, host virtualization support, and the
dependencies documented by that release. A successful checkout inspection is not
a machine-launch canary. Docker evaluations require an explicit `--path-to-vm`
or an untracked `<checkout>/docker_vm_data/Ubuntu.qcow2`; the adapter hashes the
entire selected image instead of allowing the upstream manager to download a
mutable asset during evaluation.

For each agent, the adapter constructs an independent `DesktopEnv`, calls the
official reset with the hidden task configuration, passes only the instruction
and screenshots/actions to the agent, records redacted trajectory metadata and
private screenshots, then invokes the hidden evaluator after the agent stops.
The pinned generic runners wait 60 seconds after reset and obtain a fresh
`_get_obs()` before the first model call. The v1 runner additionally waits 20
seconds before `evaluate()`; the v2 runner does not. Each step passes
`sleep_after_execution=0` to `DesktopEnv.step` because the adapter re-observes
explicitly after every batch. All of these timings are outside model inference
and are recorded in factory provenance. Scores retain
OSWorld's 0–1 scale. As in the official runner, step-budget exhaustion still
evaluates the final desktop, while an unexpected agent/model exception does not
invoke the hidden evaluator or manufacture a zero-valued score.

The OSWorld computer schema exposes a `fail` action only for OSWorld. It is
encoded as the upstream literal `FAIL` and must be sent alone; the upstream
environment treats it as terminal, while the adapter itself keeps relying on
the environment's episode-done signal rather than terminating locally. This is
required for v1's 27 `infeasible` tasks: their official evaluator awards credit
only when `FAIL` is the final action. A normal final model response remains the
minimal-loop completion signal and does not manufacture an upstream action.

OSWorld v2 data is gated. `mini-agent` reports missing release data rather than
substituting v1 tasks. The official release also binds
`xlangai/osworld_v2_tasks@v2026.06.24`, its 108-file task hash manifest, and
`Task-Web/OSWorld-web@v2026.06.24`. A run that has not verified those gated task
hashes and the matching website deployment is not an officially comparable v2
run even if the public code and VM image are exact. The current node has no gated
task access and therefore makes only a machine/controller claim.

The v2 reference runner has special protocols for task classes with
`get_phases()` and for tasks with a user simulator. The former performs repeated
setup/evaluation/gating on one desktop; the latter turns an actionless model
response into an `ASK_USER` turn. The minimal single-loop protocol implements
neither implicitly. Such tasks are rejected before a desktop or model call so
they cannot produce a plausible but invalid score. GitLab is excluded recursively
by default because its setup has extra external-service requirements; use
`--include-gitlab` only with the official service infrastructure.

The v2 pins (tag and commit) live in the `xlang-ai/OSWorld-V2` repository, not
`xlang-ai/OSWorld`; clone accordingly.

On Docker-less KVM hosts, `--runtime apptainer` plus
`--osworld-apptainer-image` selects the repo-owned compatibility client. It
implements only the Docker SDK calls used by the pinned provider, fails closed
if that contract changes, and runs the official container entrypoint. Because
Apptainer cannot grant the container a private NET_ADMIN network here, it uses
QEMU user-mode `hostfwd`, disables host noVNC, and materializes the exact UEFI
bytes from the pinned SIF as private mode-`0600` files so fakeroot QEMU can open
them. It also uses an Apptainer PID namespace so entrypoint daemons terminate
with the launcher. These runtime deviations are explicit in the manifest; the
official entrypoint, guest, controller, task setup, actions, and evaluator are
unchanged. See [machine images](machine-images.md) for the immutable OCI/archive
pins. Size guest concurrency to host memory: each guest reserves its configured
RAM in full.

## cua-speed-run

The checkout must be clean and exact. The parent and gym-anything identities
are rechecked before every preparation/lease and again after checker and cleanup,
before a verdict is committed. `--benchmark-path` points to an upstream
benchmark specification, and `--backend` must resolve in that checkout. Run
`doctor --target computer` with the same checkout, benchmark, and backend before
inference.

Direct `--environment computer` accepts the official run-plane URL shape
`https://host/<run_token>` and parses the upstream `{"ok": true, "info": ...}`
step envelope. A bearer token remains available for non-upstream compatible
gateways, but it is not presented as cua-speed-run's native authentication
contract. URL-path secrets are excluded from environment provenance.

Computer doctor is deliberately non-provisioning: it requires an already
materialized `manifest.yaml`, checks the pinned source, imports, runtime commands,
KVM access, and referenced environment/task files, but never invokes the
upstream provisioning preflight or launches a machine. `source_ready` therefore
does not mean a VM/base image is present or bootable. Image provisioning and a
machine-launch canary are separate, explicit operator actions. Evaluation does
invoke the real upstream provisioning preflight before task execution and fails
before model calls if that boundary is not ready. Cache filenames observed at
preflight are explicitly labeled unverified candidates, not selected images.
After each lease is prepared, but before its first model call, the adapter hashes
the effective base image and any selected checkpoint exposed by the runner, and
records whether a container reference is content-addressed. It rechecks the same
effective inputs when the lease is handed to an agent, which matters when a
multi-agent prewarm leaves prepared guests waiting in a pool. A single upstream
backend object owns all leases in one evaluation, preserving cua-speed-run's
invariant that the selected concrete runner cannot change mid-evaluation.

The exact gym-anything gitlink, submodule HEAD/cleanliness, and imported module
origin are enforced. For gym-anything tasks, the agent instruction is loaded from
the exact selected environment task spec using upstream file precedence. The
environment spec, selected task sources, and every resolved mount tree are
hash-bound into task data and rechecked at execution. Python bytecode is rejected
rather than excluded because timestamp- or hash-based caches can shadow verified
source. Seeded tasks bind a canonical hash of both the instruction and hidden
expected state while retaining only the hash outside the environment plane.

Environment preparation, including all possible multi-agent leases, occurs
before the benchmark task clock. The clock covers agent execution. The
configured grace period is applied before every checker invocation — normal
completion, timeout, and step-budget exhaustion alike. Step-budget exhaustion
(`finish_reason: step_budget`) still runs the hidden checker, matching the
OSWorld semantics, because the machine may already show the requested state.
An agent process error receives the upstream-equivalent false/zero verdict
without running a checker. Verifier errors and cleanup failures are
infrastructure failures, not zero scores.

The official gateway also supervises environment operations at 90 seconds and
the verifier at five minutes from a separate control plane. The maintained
in-process adapter cannot safely abandon a blocked Python call and tear down the
same live adapter underneath it. It therefore does not claim those process-level
watchdogs. Use the official submission/executor topology when hard preemption of
a wedged environment or checker is part of the required evidence.

Multi-agent computer runs lease one machine per agent; a 4-agent team therefore
needs roughly four guests' worth of memory. On constrained allocations run
single-agent with `--max-workers 1`. The hidden evaluator scores the root's live
machine unless the root explicitly adopts a completed descendant. Provenance
records `root_environment` or `adopted_descendant_environment` and, for the
latter, the ordered adoption history; merely enabling multi-agent mode does not
claim adoption.

## Parallelism

`--max-workers N` runs N benchmark tasks concurrently through a worker pool;
each task keeps its own environment (overlay, browser session, or machine
lease) while sharing one budget ledger and trace. `--model-concurrency` bounds
in-flight provider calls across all agents of a run and defaults to 4 — raise
it when `--max-workers` or a multi-agent team would otherwise serialize on the
provider semaphore. Resume markers (`completed.json`) are written per instance,
so an interrupted parallel run resumes with `--resume` without repeating
finished tasks. Failed outcomes are committed too — `--resume` replays them
rather than retrying. Start a new output directory to retry a model-backed
failure: deleting an instance is intentionally insufficient when its trace shows
that a possibly billable model call began, because prior spend cannot be
reconstructed exactly. Container and VM benchmarks are memory-bound before they
are CPU-bound: size `--max-workers` to what the host can hold in RAM.

## Claim discipline

For any reported result, retain the evaluation manifest, summary, redacted trace,
exact package/checkout pins, model identifiers, endpoint protocol, prompt hash,
budgets, environment provenance, and official grader output. Report sample size
and failures. A one-task canary proves plumbing only. It does not establish model
quality or a release winner.
