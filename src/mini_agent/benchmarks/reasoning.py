"""Competition-mathematics reasoning benchmarks.

AIME, MATH-500, OlympiadBench, and MinervaMath share one shape: a short
problem statement, one final answer, and no environment the task itself
requires. They are the only benchmarks here that need no container, no
retrieval index, and no virtual machine, so they are also the only ones whose
score can be produced without Docker.

Two things are deliberately *not* claimed. The prompt is a maintained baseline,
not any published leaderboard harness. And the default grader is deterministic
normalized comparison, not an equivalence judge: it can only undercount, never
overcount, and `--grader-model` adds an explicit second opinion for the answers
normalization cannot decide. Which grader ran is recorded per task.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .._hash import immutable_file_identity
from ..execution import RunContext
from ..models import Model
from ..types import Message, _require_mapping, _require_positive_int, _require_str
from .base import (
    BenchmarkTask,
    EvaluationOutcome,
    TeamOptions,
    numbered_json_rows,
    run_benchmark_team,
    write_prediction,
)


# The scratchpad is a deliberate choice, recorded in docs/benchmarks.md: with
# no tool at all every topology collapses to "each agent thinks alone", and the
# harness comparison this project exists to make would measure nothing.
REASONING_TASK_PROMPT = """
Solve the following problem.

{Problem}

You have a private scratch directory and a `bash` tool. Use it for arithmetic,
enumeration, or checking your work; nothing you write there is submitted or
graded. Your final message is the only thing scored.

End your final message with the exact line:

Answer: $ANSWER

where $ANSWER is the final answer alone, with no units, no working, and no
surrounding text.
""".strip()


REASONING_GRADER = r"""
Judge whether two answers to a mathematics problem are the same value.

[problem]: {problem}
[reference]: {reference}
[candidate]: {candidate}

They are the same if they differ only in form -- an unsimplified fraction, a
different but equal radical form, a decimal against its exact value, LaTeX
formatting, or ordering within an unordered set. They are different if they
name different values, or if the candidate states no answer at all.

Reply with exactly one line:

equivalent: yes
or
equivalent: no
""".strip()


@dataclass(frozen=True)
class ReasoningDataset:
    """One dataset's identity and its field names in the normalized JSONL.

    Field names are tuples because the published exports of these datasets
    disagree with each other about capitalization and naming; accepting the
    known spellings is honest, while silently accepting any field would let a
    malformed row grade as an empty answer.
    """

    name: str
    upstream: str
    problem_fields: tuple[str, ...]
    answer_fields: tuple[str, ...]
    id_fields: tuple[str, ...]
    integer_answer: bool = False


DATASETS: Mapping[str, ReasoningDataset] = {
    dataset.name: dataset
    for dataset in (
        ReasoningDataset(
            name="aime",
            upstream="HuggingFaceH4/aime_2024",
            problem_fields=("problem", "Problem", "question"),
            answer_fields=("answer", "Answer", "solution"),
            id_fields=("id", "ID", "problem_id"),
            integer_answer=True,
        ),
        ReasoningDataset(
            name="math500",
            upstream="HuggingFaceH4/MATH-500",
            problem_fields=("problem", "question"),
            answer_fields=("answer", "Answer"),
            id_fields=("unique_id", "id"),
        ),
        ReasoningDataset(
            name="olympiadbench",
            upstream="Hothan/OlympiadBench",
            problem_fields=("question", "problem"),
            answer_fields=("final_answer", "answer"),
            id_fields=("id", "index"),
        ),
        ReasoningDataset(
            name="minervamath",
            upstream="math-ai/minervamath",
            problem_fields=("problem", "question"),
            answer_fields=("answer", "solution"),
            id_fields=("id", "index"),
        ),
    )
}


def dataset_names() -> tuple[str, ...]:
    return tuple(sorted(DATASETS))


def load_reasoning(
    path: Path,
    *,
    benchmark: str,
    limit: int | None = None,
) -> tuple[BenchmarkTask, ...]:
    """Load a normalized JSONL export of one reasoning dataset.

    The export is local and hashed rather than downloaded at run time, for the
    same reason every other adapter here pins its input: a task set that can
    change underneath a manifest cannot be resumed or compared.
    """

    dataset = _dataset(benchmark)
    if limit is not None:
        _require_positive_int(limit, f"{dataset.name} limit")
    source = path.expanduser().resolve()
    tasks: list[BenchmarkTask] = []
    seen: set[str] = set()
    for number, row in numbered_json_rows(source, label=dataset.name):
        _require_mapping(row, f"{dataset.name} rows")
        problem = _field(row, dataset.problem_fields)
        if problem is None:
            raise ValueError(f"{dataset.name} line {number} has no problem statement")
        answer = _answer_field(row, dataset)
        if answer is None:
            raise ValueError(f"{dataset.name} line {number} has no answer")
        if dataset.integer_answer and _normalized_integer(answer) is None:
            raise ValueError(
                f"{dataset.name} line {number} answer is not an integer: {answer!r}"
            )
        task_id = _task_id(row, dataset, number)
        if task_id in seen:
            raise ValueError(f"duplicate {dataset.name} task id {task_id!r}")
        seen.add(task_id)
        tasks.append(
            BenchmarkTask(
                task_id,
                REASONING_TASK_PROMPT.format(Problem=problem),
                {"benchmark": dataset.name, "problem": problem, "answer": answer},
            )
        )
        if limit is not None and len(tasks) == limit:
            break
    if not tasks:
        raise ValueError(f"{dataset.name} task file contains no tasks")
    return tuple(tasks)


def dataset_identity(path: Path, *, benchmark: str) -> Mapping[str, Any]:
    """Bind the exact task-file bytes and the upstream export they came from."""

    dataset = _dataset(benchmark)
    return {
        "benchmark": dataset.name,
        "upstream_dataset": dataset.upstream,
        "export": immutable_file_identity(path, label=f"{dataset.name} tasks"),
    }


async def run_reasoning_task(
    task: BenchmarkTask,
    context: RunContext,
    directory: Path,
    *,
    scratch_root: Path,
    options: TeamOptions,
) -> EvaluationOutcome:
    """Run one problem and record the answer. No score is assigned here."""

    # Imported here rather than at module scope: the CLI reads `dataset_names`
    # from this module to build its parser, and the bash environment pulls in
    # the whole runtime stack behind it.
    from ..environments.bash import BashEnvironment

    if not isinstance(scratch_root, Path):
        raise ValueError("reasoning scratch_root must be a Path")

    async def environment_for(agent_id: str) -> BashEnvironment:
        # A fresh private root per call is what keeps a team's members from
        # sharing working state through the filesystem; what they share has to
        # travel through the harness's own messages, which is the thing being
        # measured. No Git baseline: the scratchpad is never submitted.
        del agent_id
        return await BashEnvironment.isolated(
            scratch_root=scratch_root, git_baseline=False
        )

    team = await run_benchmark_team(
        task,
        context,
        environment_factory=environment_for,
        options=options,
    )
    result = team.require()
    metadata: Mapping[str, Any] = {
        **team.metadata(),
        "benchmark": task.data.get("benchmark"),
        "extracted_answer": extract_answer(result.answer),
        "scratchpads": {
            agent_id: dict(base.provenance())
            for agent_id, base in team.bases().items()
        },
    }
    return EvaluationOutcome(
        task.task_id,
        "completed",
        answer=result.answer,
        metadata={
            **metadata,
            "prediction_sha256": write_prediction(
                directory,
                {
                    "task_id": task.task_id,
                    "answer": result.answer,
                    "extracted_answer": metadata["extracted_answer"],
                    "steps": result.steps,
                    "metadata": metadata,
                },
            ),
        },
    )


def unjudged_score(
    score: float | None, reason: str
) -> tuple[float, Mapping[str, Any]]:
    """Settle a grade when no equivalence judge is configured.

    Undecided becomes zero rather than being dropped: a deterministic grader
    can only undercount, and excluding those tasks would quietly remove the
    answers most likely to be wrong from the mean. The flag is what keeps that
    undercount measurable.
    """

    if score is not None:
        return score, {"grader": reason}
    return 0.0, {"grader": reason, "undecided_scored_zero": True}


def grade_reasoning(task: BenchmarkTask, response: str) -> tuple[float | None, str]:
    """Score one response deterministically, or decline to.

    Returns ``(None, reason)`` when normalization cannot decide, so an
    equivalence judge can be asked instead of a disagreement being recorded as
    a wrong answer.
    """

    reference = str(task.data.get("answer", ""))
    candidate = extract_answer(response)
    if candidate is None:
        return 0.0, "no-answer-line"
    dataset = _dataset(str(task.data.get("benchmark", "")))
    if dataset.integer_answer:
        # AIME answers are integers 0-999, so exact comparison is the whole of
        # the grading contract and a judge could only introduce error.
        expected = _normalized_integer(reference)
        observed = _normalized_integer(candidate)
        if observed is None:
            return 0.0, "not-an-integer"
        return (1.0 if observed == expected else 0.0), "integer-exact"
    if _normalize(candidate) == _normalize(reference):
        return 1.0, "normalized-exact"
    return None, "normalization-undecided"


async def judge_reasoning(
    *,
    task: BenchmarkTask,
    response: str,
    grader: Model,
    context: RunContext,
    agent_id: str,
) -> tuple[float | None, str]:
    """Ask a grader model whether two answers name the same value.

    An unparseable grade returns ``None`` rather than zero. A task nobody
    graded is not a task the model got wrong, and ``mean_score`` averages only
    numeric scores, so leaving it ungraded keeps grader flakiness out of the
    result instead of charging it to the model.
    """

    candidate = extract_answer(response)
    prompt = REASONING_GRADER.format(
        problem=str(task.data.get("problem", "")),
        reference=str(task.data.get("answer", "")),
        candidate="(no answer stated)" if candidate is None else candidate,
    )
    result = await context.query(
        grader,
        [Message(role="user", content=prompt)],
        (),
        agent_id=agent_id,
        role="grader",
    )
    match = re.search(r"equivalent:\s*(yes|no)", result.text, re.IGNORECASE)
    if match is None:
        return None, result.text
    return (1.0 if match.group(1).casefold() == "yes" else 0.0), result.text


def extract_answer(response: str) -> str | None:
    """Pull the final answer out of a response, preferring the stated form.

    The prompt asks for an ``Answer:`` line, so that is authoritative when it
    is present. A ``\\boxed{...}`` is the near-universal fallback these
    datasets' own solutions use.
    """

    _require_str(response, "reasoning response", non_empty=False)
    stated = None
    for line in response.splitlines():
        match = re.match(r"\s*(?:\*{0,2})answer(?:\*{0,2})\s*:\s*(.+?)\s*$", line, re.I)
        if match is not None:
            stated = match.group(1)
    if stated is not None and stated.strip():
        return _strip_wrappers(stated.strip())
    boxed = _last_boxed(response)
    if boxed is not None:
        return _strip_wrappers(boxed)
    return None


def _last_boxed(text: str) -> str | None:
    """Return the contents of the last ``\\boxed{...}``, brace-balanced."""

    marker = text.rfind("\\boxed{")
    if marker < 0:
        marker = text.rfind("\\fbox{")
        if marker < 0:
            return None
    start = text.index("{", marker) + 1
    depth = 1
    for index in range(start, len(text)):
        character = text[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start:index]
    return None


def _strip_wrappers(value: str) -> str:
    stripped = value.strip().strip("$").strip()
    boxed = _last_boxed(stripped)
    if boxed is not None:
        return boxed.strip()
    return stripped


_TEXT_WRAPPERS = ("\\text", "\\mathrm", "\\textbf", "\\mbox")


def _normalize(value: str) -> str:
    """Normalize a mathematical answer for comparison.

    This follows the widely reused `hendrycks_math` normalization: drop
    presentation-only LaTeX, unify fraction and radical spellings, and remove
    units and separators that carry no value.
    """

    text = value.strip().strip("$").strip()
    for wrapper in _TEXT_WRAPPERS:
        text = re.sub(re.escape(wrapper) + r"\s*\{([^{}]*)\}", r"\1", text)
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    text = text.replace("dfrac", "frac").replace("tfrac", "frac")
    text = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"\1/\2", text)
    text = re.sub(r"\\frac(\d)(\d)", r"\1/\2", text)
    text = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", text)
    text = re.sub(r"\\sqrt(\w)", r"sqrt(\1)", text)
    text = text.replace("^{\\circ}", "").replace("^\\circ", "")
    text = text.replace("\\%", "").replace("%", "")
    text = text.replace("\\$", "").replace("$", "")
    text = re.sub(r"\\(?:cdot|times)", "*", text)
    text = text.replace(" ", "").replace("\\", "")
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    text = text.rstrip(".")
    if re.fullmatch(r"-?\d+\.0+", text):
        text = text.split(".")[0]
    return text.casefold()


def _normalized_integer(value: str) -> int | None:
    text = _normalize(value)
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    return None


def _dataset(benchmark: str) -> ReasoningDataset:
    try:
        return DATASETS[benchmark]
    except KeyError:
        raise ValueError(
            f"unknown reasoning benchmark {benchmark!r}; "
            f"choose from {', '.join(dataset_names())}"
        ) from None


def _field(row: Mapping[str, Any], names: Sequence[str]) -> str | None:
    for name in names:
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _answer_field(row: Mapping[str, Any], dataset: ReasoningDataset) -> str | None:
    """Read the answer, accepting OlympiadBench's single-element list form."""

    for name in dataset.answer_fields:
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        if isinstance(value, list) and len(value) == 1:
            only = value[0]
            if isinstance(only, str) and only.strip():
                return only
    return None


def _task_id(row: Mapping[str, Any], dataset: ReasoningDataset, number: int) -> str:
    for name in dataset.id_fields:
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, int) and not isinstance(value, bool):
            return f"{dataset.name}-{value}"
    return f"{dataset.name}-{number:04d}"


__all__ = [
    "DATASETS",
    "REASONING_GRADER",
    "REASONING_TASK_PROMPT",
    "ReasoningDataset",
    "dataset_identity",
    "dataset_names",
    "extract_answer",
    "grade_reasoning",
    "judge_reasoning",
    "load_reasoning",
    "run_reasoning_task",
    "unjudged_score",
]
