"""BrowseComp and BrowseComp-Plus task boundaries."""

from __future__ import annotations

import base64
import binascii
import csv
import hashlib
import inspect
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterator, Mapping, TypeGuard

from ..environments.web import BrowserEnvironment
from ..models import Model
from ..runtime import RunContext
from ..types import (
    Message,
    strict_json_loads,
    _require_callable,
    _require_int,
    _require_mapping,
    _require_no_symlink,
    _require_positive_int,
    _require_str,
)
from .checkout import git as _git, reject_untracked_execution_files
from .._hash import (
    immutable_file_identity,
    immutable_tree_identity,
)
from ..storage import (
    atomic_bytes,
    atomic_json,
    read_committed_result,
)
from .base import (
    TeamOptions,
    run_benchmark_team,
    BenchmarkTask,
    EvaluationOutcome,
    task_agent_prefix,
)


BROWSECOMP_REVISION = "652c89d0ca9df547706735883097e9537d40dc47"
BROWSECOMP_PLUS_REVISION = "046949032b0328319cc9a02663a759ec601d9402"

BROWSECOMP_QUERY = """
{Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()

BROWSECOMP_PLUS_QUERY = """
You are a deep research agent. You need to answer the given question by interacting with a search engine, using the browser tool with action=search provided. Please perform reasoning and use the tool step by step, in an interleaved manner. You may use the search action multiple times.

Question: {Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer. For this explanation section only, you should cite your evidence documents inline by enclosing their docids in square brackets [] at the end of sentences. For example, [20].}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()  # noqa: E501 (byte-exact upstream literal)

BROWSECOMP_GRADER = r"""
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available.
""".strip()  # noqa: E501 (byte-exact upstream literal)


ModelFactory = Callable[[str], Model | Awaitable[Model]]
BrowserFactory = Callable[[str], BrowserEnvironment | Awaitable[BrowserEnvironment]]


def load_browsecomp(
    path: Path,
    *,
    limit: int | None = None,
    sample_seed: int = 0,
) -> tuple[BenchmarkTask, ...]:
    """Load the official encrypted CSV downloaded from simple-evals."""

    if limit is not None:
        _require_positive_int(limit, "BrowseComp limit")
    _require_int(sample_seed, "BrowseComp sample_seed")
    with path.expanduser().resolve().open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if (
            fields is None
            or len(fields) != len(set(fields))
            or not {"problem", "answer", "canary"}.issubset(fields)
        ):
            raise ValueError("BrowseComp CSV has invalid required columns")
        rows = list(enumerate(reader))
    if limit is not None:
        if limit > len(rows):
            raise ValueError("BrowseComp limit exceeds the dataset size")
        rows = random.Random(sample_seed).sample(rows, limit)
    tasks: list[BenchmarkTask] = []
    seen: set[str] = set()
    for source_index, row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"BrowseComp row {source_index + 1} is malformed")
        canary = row["canary"]
        problem = _decrypt(row["problem"], canary)
        answer = _decrypt(row["answer"], canary)
        if not problem.strip() or not answer.strip():
            raise ValueError(
                f"BrowseComp row {source_index + 1} decrypts to an empty field"
            )
        task_id = row.get("id") or f"browsecomp-{source_index:04d}"
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError(f"BrowseComp row {source_index + 1} has an invalid id")
        if task_id in seen:
            raise ValueError(f"duplicate BrowseComp task id {task_id!r}")
        seen.add(task_id)
        tasks.append(
            BenchmarkTask(
                task_id,
                BROWSECOMP_QUERY.format(Question=problem),
                {"question": problem, "answer": answer},
            )
        )
    if not tasks:
        raise ValueError("BrowseComp CSV contains no tasks")
    return tuple(tasks)


def load_browsecomp_plus(
    path: Path, *, limit: int | None = None
) -> tuple[BenchmarkTask, ...]:
    """Load BrowseComp-Plus query TSV or normalized JSONL without qrels."""

    source = path.expanduser().resolve()
    tasks: list[BenchmarkTask] = []
    seen: set[str] = set()
    if limit is not None:
        _require_positive_int(limit, "BrowseComp-Plus limit")
    if source.suffix.casefold() == ".jsonl":
        rows = []
        for number, line in _numbered_lines(source):
            try:
                rows.append(strict_json_loads(line))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"BrowseComp-Plus line {number} is invalid JSON"
                ) from exc
    else:
        with source.open(newline="", encoding="utf-8") as stream:
            rows = []
            for number, row in enumerate(csv.reader(stream, delimiter="\t"), 1):
                if not row or not any(value.strip() for value in row):
                    continue
                if len(row) != 2:
                    raise ValueError(
                        f"BrowseComp-Plus TSV row {number} must have two columns"
                    )
                if number == 1 and (
                    row[0].strip().removeprefix("\ufeff").casefold() == "query_id"
                    and row[1].strip().casefold() == "query"
                ):
                    raise ValueError(
                        "BrowseComp-Plus TSV must be headerless, matching the "
                        "pinned upstream generator"
                    )
                rows.append({"query_id": row[0].strip(), "query": row[1].strip()})
    for index, row in enumerate(rows):
        _require_mapping(row, "BrowseComp-Plus rows")
        query = row.get("query", row.get("question"))
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"BrowseComp-Plus row {index + 1} has no query")
        normalized_id = _query_identifier(row.get("query_id", row.get("id", index)))
        if normalized_id is None:
            raise ValueError(f"BrowseComp-Plus row {index + 1} has no query_id")
        if normalized_id in seen:
            raise ValueError(f"duplicate BrowseComp-Plus query_id {normalized_id!r}")
        seen.add(normalized_id)
        data: dict[str, Any] = {"benchmark": "browsecomp-plus"}
        answer = row.get("answer")
        if answer is not None:
            _require_str(
                answer, f"BrowseComp-Plus row {index + 1} answer", non_empty=False
            )
            data.update({"question": query, "answer": answer})
        tasks.append(
            BenchmarkTask(
                normalized_id,
                BROWSECOMP_PLUS_QUERY.format(Question=query),
                data,
            )
        )
        if limit is not None and len(tasks) == limit:
            break
    if not tasks:
        raise ValueError("BrowseComp-Plus task file contains no tasks")
    return tuple(tasks)


async def run_web_task(
    task: BenchmarkTask,
    context: RunContext,
    directory: Path,
    *,
    browser_factory: BrowserFactory,
    options: TeamOptions,
    model_name: str | None = None,
) -> EvaluationOutcome:
    _require_callable(browser_factory, "browser_factory")
    if task.data.get("benchmark") == "browsecomp-plus":
        _require_str(model_name, "BrowseComp-Plus generation model_name")

    async def environment_for(agent_id: str) -> BrowserEnvironment:
        browser = browser_factory(agent_id)
        resolved = await browser if inspect.isawaitable(browser) else browser
        if not isinstance(resolved, BrowserEnvironment):
            raise TypeError("browser_factory must return BrowserEnvironment")
        if task.data.get("benchmark") == "browsecomp-plus":
            if resolved.allow_open:
                raise ValueError("BrowseComp-Plus generation must be search-only")
            if (
                resolved.top_k != 5
                or resolved.snippet_tokens != 512
                or resolved.tokenizer is None
                or resolved.max_observation_chars is not None
            ):
                raise ValueError(
                    "BrowseComp-Plus generation requires top-5, 512-token "
                    "snippets without an additional observation cap"
                )
        return resolved

    team = await run_benchmark_team(
        task,
        context,
        environment_factory=environment_for,
        options=options,
    )
    result = team.require()
    metadata: Mapping[str, Any] = {
        **team.metadata(),
        "browsers": {
            agent_id: {
                "accounting": dict(base.accounting()),
                "provenance": dict(base.provenance()),
            }
            for agent_id, base in team.bases().items()
        },
    }
    prediction = {
        "task_id": task.task_id,
        "answer": result.answer,
        "steps": result.steps,
        "metadata": metadata,
    }
    atomic_json(directory / "prediction.json", prediction)
    artifact_metadata = {
        **metadata,
        "prediction_sha256": hashlib.sha256(
            (directory / "prediction.json").read_bytes()
        ).hexdigest(),
    }
    if task.data.get("benchmark") == "browsecomp-plus":
        accounting = _browser_accounting(metadata)
        official = {
            "metadata": {
                "model": model_name,
                "harness": "mini-agent",
                "source_revision": BROWSECOMP_PLUS_REVISION,
                "upstream_query_template": "QUERY_TEMPLATE_NO_GET_DOCUMENT",
                "tool_topology": "one browser tool with a search action",
            },
            "query_id": task.task_id,
            "tool_call_counts": {
                "search": accounting["tool_calls"].get("search", 0),
            },
            "status": "completed",
            "retrieved_docids": accounting["references"],
            "result": [
                {
                    "type": "output_text",
                    "tool_name": None,
                    "arguments": None,
                    "output": result.answer,
                }
            ],
        }
        atomic_json(directory / "browsecomp_plus_run.json", official)
        artifact_metadata["browsecomp_plus_run_sha256"] = hashlib.sha256(
            (directory / "browsecomp_plus_run.json").read_bytes()
        ).hexdigest()
    return EvaluationOutcome(
        task.task_id,
        "completed",
        answer=result.answer,
        metadata=artifact_metadata,
    )


async def grade_browsecomp(
    *,
    task: BenchmarkTask,
    response: str,
    grader: Model,
    context: RunContext,
) -> tuple[float, str]:
    answer = task.data.get("answer")
    question = task.data.get("question")
    if not isinstance(answer, str) or not isinstance(question, str):
        raise ValueError("BrowseComp task has no hidden grading fields")
    prompt = BROWSECOMP_GRADER.format(
        question=question, response=response, correct_answer=answer
    )
    result = await context.query(
        grader,
        [Message(role="user", content=prompt)],
        (),
        agent_id=task_agent_prefix(task.task_id) + "/grader",
        role="grader",
    )
    match = re.search(r"correct: (yes|no)", result.text)
    return (1.0 if match and match.group(1) == "yes" else 0.0, result.text)


def _derive_key(password: str, length: int) -> bytes:
    digest = hashlib.sha256(password.encode()).digest()
    return digest * (length // len(digest)) + digest[: length % len(digest)]


def _decrypt(ciphertext: str, password: str) -> str:
    if not isinstance(ciphertext, str) or not isinstance(password, str):
        raise ValueError("BrowseComp encrypted fields must be strings")
    try:
        encrypted = base64.b64decode(ciphertext, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("BrowseComp field is not valid base64") from exc
    key = _derive_key(password, len(encrypted))
    try:
        return bytes(left ^ right for left, right in zip(encrypted, key)).decode()
    except UnicodeDecodeError as exc:
        raise ValueError("BrowseComp field did not decrypt to UTF-8") from exc


def _numbered_lines(path: Path) -> Iterator[tuple[int, str]]:
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            yield number, line


def _query_identifier(value: Any) -> str | None:
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        return None
    return str(value) or None


def _reject_missing(
    identifiers: set[str], known: Mapping[str, Any] | set[str], message: str
) -> None:
    missing = sorted(identifiers.difference(known))
    if missing:
        raise ValueError(f"{message}: " + ", ".join(missing))


def _valid_counts(value: Any, *, named: bool) -> bool:
    """Return whether ``value`` maps tool names to non-negative counts.

    ``named`` additionally rejects the empty tool name.
    """

    return isinstance(value, Mapping) and all(
        isinstance(name, str)
        and (name != "" or not named)
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count >= 0
        for name, count in value.items()
    )


def _valid_references(value: Any) -> TypeGuard[list[str]]:
    return isinstance(value, list) and all(
        isinstance(item, str) and item for item in value
    )


def _browser_accounting(metadata: Mapping[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    references: set[str] = set()
    browsers = metadata.get("browsers", {})
    values = (
        [
            item.get("accounting", {})
            for item in browsers.values()
            if isinstance(item, Mapping)
        ]
        if isinstance(browsers, Mapping)
        else []
    )
    if not values:
        raise ValueError("web generation metadata has no browser accounting")
    for value in values:
        _require_mapping(value, "web browser accounting")
        raw_counts = value.get("tool_calls", {})
        _require_mapping(raw_counts, "web tool-call accounting")
        if not _valid_counts(raw_counts, named=False):
            raise ValueError("web tool-call accounting is invalid")
        for name, count in raw_counts.items():
            counts[name] = counts.get(name, 0) + count
        raw_references = value.get("references", [])
        if not _valid_references(raw_references):
            raise ValueError("web reference accounting is invalid")
        references.update(raw_references)
    return {"tool_calls": counts, "references": sorted(references)}


def collect_browsecomp_plus_runs(output: Path, destination: Path) -> int:
    """Collect per-task artifacts into the directory consumed by upstream."""

    expanded_root = output.expanduser()
    root = expanded_root.resolve()
    instances = root / "instances"
    if (
        expanded_root.is_symlink()
        or expanded_root.absolute() != root
        or not root.is_dir()
        or instances.is_symlink()
        or not instances.is_dir()
        or instances.resolve() != instances
    ):
        raise ValueError(
            "BrowseComp-Plus evaluation instances must be a non-symlink directory"
        )
    target = _browsecomp_plus_collection_target(root, destination)
    target.mkdir(mode=0o700, exist_ok=True)
    target.chmod(0o700)
    records: list[tuple[str, bytes]] = []
    query_ids: set[str] = set()
    for path in sorted(root.glob("instances/*/browsecomp_plus_run.json")):
        parent = path.parent
        if (
            parent.parent != instances
            or parent.is_symlink()
            or not parent.is_dir()
            or parent.resolve().parent != instances
            or path.is_symlink()
            or not path.is_file()
        ):
            raise ValueError(
                "BrowseComp-Plus run must be an owned regular instance artifact"
            )
        raw = path.read_bytes()
        value = strict_json_loads(raw)
        query_id = _validate_browsecomp_plus_run(value, path)
        try:
            result_value = read_committed_result(path.parent, query_id)
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"BrowseComp-Plus run has no committed result: {path}"
            ) from exc
        if result_value.get("status") != "completed":
            raise ValueError(f"BrowseComp-Plus result is not completed: {path}")
        metadata = result_value.get("metadata")
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("browsecomp_plus_run_sha256")
            != hashlib.sha256(raw).hexdigest()
        ):
            raise ValueError(f"BrowseComp-Plus run hash does not match: {path}")
        if query_id in query_ids:
            raise ValueError(f"duplicate BrowseComp-Plus query_id {query_id!r}")
        query_ids.add(query_id)
        records.append((query_id, raw))
    if not records:
        raise ValueError("evaluation contains no BrowseComp-Plus run artifacts")
    existing: dict[str, Path] = {}
    for path in target.glob("*.json"):
        if path.is_symlink():
            raise ValueError(f"collected run artifact must not be a symlink: {path}")
        try:
            value = strict_json_loads(path.read_bytes())
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"invalid existing collected run artifact: {path}"
            ) from exc
        query_id = _validate_browsecomp_plus_run(value, path)
        expected = hashlib.sha256(query_id.encode("utf-8")).hexdigest() + ".json"
        if path.name != expected:
            raise ValueError(
                f"unexpected JSON file in BrowseComp-Plus collection: {path}"
            )
        existing[path.name] = path

    index: dict[str, str] = {}
    for query_id, raw in records:
        _browsecomp_plus_collection_target(root, destination, require_exists=True)
        name = hashlib.sha256(query_id.encode("utf-8")).hexdigest() + ".json"
        atomic_bytes(target / name, raw)
        index[query_id] = name
    for name, path in existing.items():
        if name not in index.values():
            _browsecomp_plus_collection_target(root, destination, require_exists=True)
            if path.parent != target or path.is_symlink() or not path.is_file():
                raise ValueError(
                    "collected run cleanup target must be an owned regular file"
                )
            path.unlink()
    index_content = "".join(
        f"{query_id}\t{name}\n" for query_id, name in sorted(index.items())
    ).encode("utf-8")
    atomic_bytes(target / "_index.tsv", index_content)
    return len(records)


def _browsecomp_plus_collection_target(
    root: Path, destination: Path, *, require_exists: bool = False
) -> Path:
    expanded = destination.expanduser()
    target = _require_no_symlink(expanded, "BrowseComp-Plus collection").resolve()
    if target.parent != root:
        raise ValueError(
            "BrowseComp-Plus collection must be a direct child of the evaluation"
        )
    if target.exists() and (target.is_symlink() or not target.is_dir()):
        raise ValueError("BrowseComp-Plus collection must be a non-symlink directory")
    if require_exists and not target.is_dir():
        raise ValueError("BrowseComp-Plus collection directory disappeared")
    return target


def _validate_browsecomp_plus_run(value: Any, path: Path) -> str:
    if not isinstance(value, Mapping):
        raise ValueError(f"invalid BrowseComp-Plus run artifact: {path}")
    query_id = value.get("query_id")
    counts = value.get("tool_call_counts")
    references = value.get("retrieved_docids")
    result = value.get("result")
    if (
        not isinstance(query_id, str)
        or not query_id
        or value.get("status") != "completed"
        or not _valid_counts(counts, named=True)
        or not _valid_references(references)
        or len(references) != len(set(references))
        or not isinstance(result, list)
        or not result
        or not all(isinstance(item, Mapping) for item in result)
        or result[-1].get("type") != "output_text"
        or not isinstance(result[-1].get("output"), str)
    ):
        raise ValueError(f"invalid BrowseComp-Plus run artifact: {path}")
    return query_id


def official_browsecomp_plus_grader_argv(
    *,
    checkout: Path,
    input_dir: Path,
    ground_truth: Path,
    eval_dir: Path,
    qrel_evidence: Path,
    python_executable: str = sys.executable,
    model: str = "Qwen/Qwen3-32B",
    tensor_parallel_size: int = 1,
) -> tuple[str, ...]:
    root = _require_no_symlink(
        checkout.expanduser(), "BrowseComp-Plus checkout"
    ).resolve()
    script = root / "scripts_evaluation" / "evaluate_run.py"
    if not root.is_dir() or not script.is_file() or script.is_symlink():
        raise ValueError("BrowseComp-Plus checkout is missing the official grader")
    revision = _git(root, "rev-parse", "HEAD")
    if revision != BROWSECOMP_PLUS_REVISION:
        raise ValueError(
            f"BrowseComp-Plus checkout must be {BROWSECOMP_PLUS_REVISION}, "
            f"found {revision}"
        )
    if _git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("BrowseComp-Plus checkout has tracked modifications")
    reject_untracked_execution_files(root, label="BrowseComp-Plus", run_git=_git)
    words = (python_executable, model)
    if (
        not isinstance(tensor_parallel_size, int)
        or isinstance(tensor_parallel_size, bool)
        or tensor_parallel_size < 1
        or any(not isinstance(w, str) or not w or "\x00" in w for w in words)
        or model.startswith("-")
    ):
        raise ValueError("invalid BrowseComp-Plus grader configuration")
    resolved_input = input_dir.expanduser().resolve()
    resolved_truth = ground_truth.expanduser().resolve()
    resolved_eval = eval_dir.expanduser().resolve()
    resolved_qrels = qrel_evidence.expanduser().resolve()
    if not resolved_input.is_dir() or not resolved_eval.parent.is_dir() or not all(
        item.is_file() for item in (resolved_truth, resolved_qrels)
    ):
        raise ValueError("BrowseComp-Plus grader inputs are missing")
    return (
        python_executable,
        "-I",
        str(script),
        "--input_dir", str(resolved_input),
        "--ground_truth", str(resolved_truth),
        "--eval_dir", str(resolved_eval),
        "--qrel_evidence", str(resolved_qrels),
        "--model", model,
        "--tensor_parallel_size", str(tensor_parallel_size),
    )


def inspect_browsecomp_plus_grade_inputs(
    *, input_dir: Path, ground_truth: Path, qrel_evidence: Path
) -> Mapping[str, Any]:
    """Validate official grader inputs and bind runs to their visible queries."""

    runs_root = input_dir.expanduser().resolve()
    run_ids: set[str] = set()
    for path in sorted(runs_root.glob("*.json")):
        value = strict_json_loads(path.read_text(encoding="utf-8"))
        query_id = _validate_browsecomp_plus_run(value, path)
        if query_id in run_ids:
            raise ValueError(f"duplicate BrowseComp-Plus grader query {query_id!r}")
        run_ids.add(query_id)
    if not run_ids:
        raise ValueError("BrowseComp-Plus grader input contains no runs")

    truth_path = ground_truth.expanduser().resolve()
    questions: dict[str, str] = {}
    for number, line in _numbered_lines(truth_path):
        value = strict_json_loads(line)
        _require_mapping(value, f"ground truth line {number}")
        query = value.get("query")
        normalized = _query_identifier(value.get("query_id"))
        if normalized is None or not all(
            isinstance(item, str) and item.strip()
            for item in (query, value.get("answer"))
        ):
            raise ValueError(f"ground truth line {number} is malformed")
        if normalized in questions:
            raise ValueError(f"duplicate ground-truth query_id {normalized!r}")
        questions[normalized] = query
    _reject_missing(
        run_ids, questions, "BrowseComp-Plus runs are missing ground truth"
    )

    qrel_path = qrel_evidence.expanduser().resolve()
    qrel_ids: set[str] = set()
    for number, line in _numbered_lines(qrel_path):
        fields = line.split()
        if len(fields) != 4 or not fields[0] or not fields[2]:
            raise ValueError(f"qrel evidence line {number} is malformed")
        try:
            int(fields[3])
        except ValueError as exc:
            raise ValueError(f"qrel evidence line {number} is malformed") from exc
        qrel_ids.add(fields[0])
    _reject_missing(qrel_ids, questions, "qrel evidence references unknown queries")
    _reject_missing(run_ids, qrel_ids, "BrowseComp-Plus runs have no qrel evidence")

    return {
        "runs": immutable_tree_identity(runs_root, label="BrowseComp-Plus runs"),
        "ground_truth": immutable_file_identity(
            truth_path, label="BrowseComp-Plus ground truth"
        ),
        "qrel_evidence": immutable_file_identity(
            qrel_path, label="BrowseComp-Plus qrel evidence"
        ),
        "run_count": len(run_ids),
        "ground_truth_count": len(questions),
        "qrel_query_count": len(qrel_ids),
        "query_prompt_sha256": {
            query_id: hashlib.sha256(
                BROWSECOMP_PLUS_QUERY.format(Question=questions[query_id]).encode(
                    "utf-8"
                )
            ).hexdigest()
            for query_id in sorted(run_ids)
        },
    }


__all__ = [
    "BROWSECOMP_GRADER",
    "BROWSECOMP_PLUS_REVISION",
    "BROWSECOMP_PLUS_QUERY",
    "BROWSECOMP_QUERY",
    "BROWSECOMP_REVISION",
    "collect_browsecomp_plus_runs",
    "inspect_browsecomp_plus_grade_inputs",
    "grade_browsecomp",
    "load_browsecomp",
    "load_browsecomp_plus",
    "official_browsecomp_plus_grader_argv",
    "run_web_task",
]
