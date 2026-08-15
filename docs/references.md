# Reference and claim ledger

This ledger records the primary implementation sources used for the maintained
adapters. A source pin identifies the contract inspected; it does not make the
generic mini-agent loop identical to an upstream model harness. The last
reference review was 2026-08-12.

## Universal loop and model protocols

| Surface | Primary source | What is preserved | Claim boundary |
|---|---|---|---|
| Minimal loop | [`SWE-agent/mini-swe-agent` `DefaultAgent` at `a83fcae…`](https://github.com/SWE-agent/mini-swe-agent/blob/a83fcae82d2a08f0ee0c688f9d137b3566c097f8/src/minisweagent/agents/default.py) | One linear history, one model query per step, declared actions, observations, and explicit limits | Design reference only. This repository uses typed provider-neutral messages, native function tools, asynchronous environments, a shared ledger, and a trace recorder. It does not copy mini-swe-agent prompts or provider wrappers. |
| OpenAI Responses | [Responses API reference](https://platform.openai.com/docs/api-reference/responses/create), [conversation state](https://developers.openai.com/api/docs/guides/conversation-state), and [latest-model guidance](https://developers.openai.com/api/docs/guides/latest-model) | Function calls and outputs, multimodal observations, usage, refusals, completion status, response-model identity, and `previous_response_id` continuation | The maintained codec is single-shot and stored-response based. It rejects streaming, `store=false`, and server-managed `conversation`; a downstream replay codec is required for Zero Data Retention. |
| OpenAI-compatible Chat Completions | [Chat Completions API reference](https://platform.openai.com/docs/api-reference/chat/create) | One choice, function calls/results, finish reasons, multimodal inputs, usage, response-model identity, and full transcript replay | Compatibility codec, not a claim that every third-party endpoint implements every OpenAI behavior. Sampling fields are left to the endpoint unless the operator supplies them. |
| Anthropic Messages | [Messages API](https://docs.anthropic.com/en/api/messages), [tool-use contract](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls), and [stop reasons](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons) | Ordered tool-use/tool-result blocks, images, cache usage, response-model identity, and the documented terminal stop reasons | Client tools are supported. `pause_turn` requires unsupported server-tool continuation and fails closed. |
| Meta deployment | Operator-selected endpoint implementing one of the two OpenAI wire contracts above | Explicit endpoint, requested model, protocol choice, response-model identity, and the selected OpenAI codec | There is no guessed Meta hostname and no claim of a canonical Meta inference harness. The adapter is an explicitly labelled compatibility path. |

Model and system cards describe a particular model snapshot's behavior and
safety properties; they are not transport or agent-loop specifications. The
portable spec therefore records the requested model identifier, while provider
responses record and hash the resolved model identifier. For a reproducible live
experiment, operators must also bind the exact resolved snapshot and cite the
corresponding provider card from the [OpenAI Deployment Safety
Hub](https://deploymentsafety.openai.com/), [Anthropic model-system-card
index](https://www.anthropic.com/system-cards), or selected [Meta Llama model
card](https://www.llama.com/get-started/), as applicable. A card for one model
cannot validate another model served behind a mutable alias. These indexes are
reference directories, not pins: the run record must name the exact card and
snapshot actually used.

## Evaluation domains

| Domain | Primary implementation source | Maintained fidelity boundary |
|---|---|---|
| SWE-bench | [`SWE-bench` evaluator at tag `v4.1.0`, commit `726c546…`](https://github.com/SWE-bench/SWE-bench/blob/726c5461e2ef52d83cf1ea2107870a8bb3328d57/swebench/harness/run_evaluation.py) and [mini-swe-agent SWE-bench config at `a83fcae…`](https://github.com/SWE-agent/mini-swe-agent/blob/a83fcae82d2a08f0ee0c688f9d137b3566c097f8/src/minisweagent/config/benchmarks/swebench.yaml) | Official task rows, container images, patch/prediction schema, and grader command are bound. The solver remains the generic one-`bash` baseline, not an official leaderboard inference harness. |
| ProgramBench | [`facebookresearch/ProgramBench` at `963063c…`, version `1.2.4`](https://github.com/facebookresearch/programbench/blob/963063c9271cc40fa179977356782ea4582e0b0c/docs/README.md), its [image naming](https://github.com/facebookresearch/programbench/blob/963063c9271cc40fa179977356782ea4582e0b0c/src/programbench/constants.py), [evaluator](https://github.com/facebookresearch/programbench/blob/963063c9271cc40fa179977356782ea4582e0b0c/src/programbench/eval/eval.py), and [paper](https://arxiv.org/abs/2605.03546) | Pinned task metadata, `task_cleanroom_v6` image naming, the mandatory no-internet inference container, the `<run>/<instance_id>/submission.tar.gz` submission layout, and the official `programbench eval` command are bound. Hidden `tests.json` is hashed, never loaded. The solver is the generic one-`bash` baseline, not the upstream mini-swe-agent baseline system. |
| BrowseComp | [`openai/simple-evals` `browsecomp_eval.py` at `652c89d…`](https://github.com/openai/simple-evals/blob/652c89d0ca9df547706735883097e9537d40dc47/browsecomp_eval.py) | Dataset decoding, question prompt, and independent private-answer grading are preserved. The live SerpAPI/browser baseline is local policy because simple-evals does not define a browsing tool. |
| Live web transport | [SerpAPI Google Search API](https://serpapi.com/search-api), [organic-results schema](https://serpapi.com/organic-results), [Playwright `page.goto`](https://playwright.dev/python/docs/api/class-page#page-goto), and [HTTPX async streaming](https://www.python-httpx.org/async/) | The adapter consumes successful Google organic results, streams bounded bodies, checks every HTTP redirect, and treats Playwright's returned response status separately because `goto` does not raise for ordinary HTTP error responses. DNS rebinding and browser subresources remain outside the in-process security boundary. |
| BrowseComp-Plus | [`texttron/BrowseComp-Plus` OpenAI client at `0469490…`](https://github.com/texttron/BrowseComp-Plus/blob/046949032b0328319cc9a02663a759ec601d9402/search_agent/openai_client.py), [BM25 searcher](https://github.com/texttron/BrowseComp-Plus/blob/046949032b0328319cc9a02663a759ec601d9402/searcher/searchers/bm25_searcher.py), and [grader](https://github.com/texttron/BrowseComp-Plus/blob/046949032b0328319cc9a02663a759ec601d9402/scripts_evaluation/evaluate_run.py) | Canonical generation is search-only, top five, with 512-token snippets and official run/grader artifacts. The generic tool is named `browser`; full-document `open` is deliberately absent. |
| OSWorld v1 | [`xlang-ai/OSWorld` `run_multienv.py` at `091f5ef…`](https://github.com/xlang-ai/OSWorld/blob/091f5ef1d5544bc74953c77875d5feb5bed30108/scripts/python/run_multienv.py) | Exact checkout, task loading, reset/setup, reference settle timing, pyautogui/`FAIL` steps, and hidden evaluation. The model-facing computer action format is adapted. |
| OSWorld v2 | [`xlang-ai/OSWorld-V2` release `v2026.06.24`, commit `2b9b7b4…`](https://github.com/xlang-ai/OSWorld-V2/blob/2b9b7b4eb73243d557bdbf2998fe18d8e18e19c6/scripts/python/run_multienv.py) | Standard single-phase tasks use the release lifecycle. Gated data, human-in-the-loop, user-simulator, and multi-phase tasks fail closed unless their full protocol is implemented. |
| AIME, MATH-500, OlympiadBench, MinervaMath | The published problem/answer datasets themselves ([`HuggingFaceH4/aime_2024`](https://huggingface.co/datasets/HuggingFaceH4/aime_2024), [`HuggingFaceH4/MATH-500`](https://huggingface.co/datasets/HuggingFaceH4/MATH-500), [`Hothan/OlympiadBench`](https://huggingface.co/datasets/Hothan/OlympiadBench), [`math-ai/minervamath`](https://huggingface.co/datasets/math-ai/minervamath)); the normalization follows the widely reused `hendrycks_math` answer-cleaning convention | **No upstream evaluation code is executed.** These datasets publish problems and answers, not a harness, so the prompt, answer extraction, normalization, and the optional equivalence judge are this project's own and are versioned with it. The exact task-file bytes are hashed into each manifest. A score is comparable across `mini-agent` runs and is not a reproduction of any published leaderboard figure. |

See [benchmark fidelity](benchmarks.md) for required assets, deliberate defect
corrections, grader boundaries, and runnable commands. See [validation](validation.md)
for what was actually executed on the current node. Neither a deterministic
unit test nor a scripted canary is reported as a benchmark win.
