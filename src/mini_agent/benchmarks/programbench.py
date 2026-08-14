"""ProgramBench generation and official-grader contracts.

Upstream is `facebookresearch/ProgramBench` ("Can Language Models Rebuild
Programs From Scratch?"). The agent works offline inside the per-task
cleanroom image and its submission is the workspace tree. Scoring is never
computed here: only the official `programbench eval` CLI produces a score.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

from ..environments.bash import (
    DEFAULT_MAX_ARCHIVE_BYTES,
    BashEnvironment,
    SWEArchiveState,
)
from ..models import Model
from ..runtime import RunContext
from ..specs import AgentSpecV1
from ..types import (
    BudgetLimits,
    _require_bool,
    _require_callable,
    _require_mapping,
    _require_no_symlink,
    _require_positive_int,
    _require_str,
    strict_json_loads,
)
from .._hash import (
    immutable_file_identity,
    immutable_tree_identity,
)
from ..storage import (
    atomic_bytes,
    read_committed_result,
)
from .base import (
    BenchmarkTask,
    owned_instance_artifacts,
    EvaluationOutcome,
    run_benchmark_team,
)
from .checkout import git as _git, reject_untracked_execution_files
from .swebench import (
    SWEbenchImageBinding,
    apptainer_swe_environment,
    docker_swe_environment,
)


PROGRAMBENCH_REVISION = "963063c9271cc40fa179977356782ea4582e0b0c"
PROGRAMBENCH_VERSION = "1.2.4"
PROGRAMBENCH_IMAGE_TAG = "task_cleanroom_v6"
PROGRAMBENCH_DOCKER_ORG = "programbench"
PROGRAMBENCH_WORKDIR = "/workspace"
PROGRAMBENCH_SUBMISSION_NAME = "submission.tar.gz"

# Every requirement below is an upstream contract: the evaluator wipes the
# workspace, unpacks the submission archive, deletes any shipped ./executable,
# and runs ./compile.sh with the build network blocked.
PROGRAMBENCH_TASK_PROMPT = """
Rebuild a program from scratch inside {workdir}.

{workdir} already contains the compiled reference program and its
documentation. Write a complete {language} codebase in {workdir} whose build
reproduces that program's observable behaviour.

Rules enforced by the evaluator:

- The submission is the {workdir} tree exactly as you leave it.
- {workdir}/compile.sh must build your sources and leave the built program at
  {workdir}/executable. It runs with no network access.
- Any prebuilt executable you ship is deleted before the build, so the binary
  must come from your own sources.

This container has no internet access. Task id: {instance_id}.
""".strip()

PROGRAMBENCH_SHARED_GIT = "/srv/team.git"

ModelFactory = Callable[[str], Model | Awaitable[Model]]


def _shared_git_repository(
    task_id: str, scratch_root: Path | None, enabled: bool
) -> dict[str, Path]:
    """Create the bare repository a team pushes and pulls through.

    It lives outside the workspace on purpose: `export_archive` tars the
    workspace, so a repository inside it would be submitted, and adopting a
    descendant's workspace deletes everything in it, which would destroy every
    teammate's history.
    """

    if not enabled:
        return {}
    if scratch_root is None:
        raise ValueError("--agent-git-share requires a scratch root")
    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
    path = _require_no_symlink(scratch_root / "share" / digest, "shared git root")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not (path / "HEAD").exists():
        _git(path, "init", "--bare", "--initial-branch=main")
        # Concurrent pushes must never race a background repack.
        _git(path, "config", "gc.auto", "0")
    return {PROGRAMBENCH_SHARED_GIT: path}
_INSTANCE_ID = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}")
_IMAGE_TAG = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}")
_PROJECT_VERSION = re.compile(r'(?m)^version\s*=\s*"([^"\s]+)"\s*$')
_TASK_YAML_SCALAR = re.compile(r"([a-z_]+): ([A-Za-z0-9._/+-]+)")
_TASK_YAML_KEY = re.compile(r"([a-z_]+):")
_TASK_YAML_ITEM = re.compile(r"- ([A-Za-z0-9._/+-]+)")
# The upstream entry point declared in the pinned pyproject.toml. The package
# ships no __main__ module, so the isolated grader Python imports it directly.
_PROGRAMBENCH_CLI_ENTRY = "from programbench.cli.main import app; app()"


def _require_instance_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _INSTANCE_ID.fullmatch(value):
        raise ValueError(f"{label} must be one lowercase path-safe component")
    return value


def programbench_image_name(instance_id: Any) -> str:
    """Return the pinned per-task inference image reference."""

    resolved = _require_instance_id(instance_id, "ProgramBench instance_id")
    repository = resolved.replace("__", "_1776_")
    return f"{PROGRAMBENCH_DOCKER_ORG}/{repository}:{PROGRAMBENCH_IMAGE_TAG}"


def _pyproject_version(path: Path) -> str:
    text = "\n" + path.read_text(encoding="utf-8")
    _, marker, tail = text.partition("\n[project]\n")
    if not marker:
        raise ValueError("ProgramBench pyproject.toml has no [project] table")
    match = _PROJECT_VERSION.search(re.split(r"(?m)^\[", tail, maxsplit=1)[0])
    if match is None:
        raise ValueError("ProgramBench pyproject.toml declares no version")
    return match.group(1)


def _read_task_yaml(path: Path) -> Mapping[str, Any]:
    """Read the pinned task.yaml grammar; anything else fails closed."""

    value: dict[str, Any] = {}
    current: list[str] | None = None
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        scalar = _TASK_YAML_SCALAR.fullmatch(line)
        key = _TASK_YAML_KEY.fullmatch(line)
        item = _TASK_YAML_ITEM.fullmatch(line)
        if scalar is not None:
            current = None
            name, entry = scalar.group(1), scalar.group(2)
        elif key is not None:
            current = []
            name, entry = key.group(1), current
        elif item is not None and current is not None:
            current.append(item.group(1))
            continue
        else:
            raise ValueError(
                f"unsupported ProgramBench task metadata: {path} line {number}"
            )
        if name in value:
            raise ValueError(f"duplicate ProgramBench task field {name!r}: {path}")
        value[name] = entry
    if not value:
        raise ValueError(f"empty ProgramBench task metadata: {path}")
    return value


def _task_field(metadata: Mapping[str, Any], name: str, path: Path) -> str:
    entry = metadata.get(name)
    if not isinstance(entry, str) or not entry:
        raise ValueError(f"ProgramBench task metadata has no {name}: {path}")
    return entry


def inspect_programbench_checkout(
    checkout: Path, *, run_git: Callable[..., str] | None = None
) -> Mapping[str, Any]:
    """Bind the pinned upstream checkout that supplies task data and scoring."""

    inspect_git = _git if run_git is None else run_git
    root = _require_no_symlink(
        checkout.expanduser(), "ProgramBench checkout"
    ).resolve()
    tasks = root / "src" / "programbench" / "data" / "tasks"
    project = root / "pyproject.toml"
    if (
        not root.is_dir()
        or tasks.is_symlink()
        or not tasks.is_dir()
        or project.is_symlink()
        or not project.is_file()
    ):
        raise ValueError("ProgramBench checkout is missing its pinned task data")
    revision = inspect_git(root, "rev-parse", "HEAD")
    if revision != PROGRAMBENCH_REVISION:
        raise ValueError(
            f"ProgramBench checkout must be {PROGRAMBENCH_REVISION}, "
            f"found {revision}"
        )
    if inspect_git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("ProgramBench checkout has tracked modifications")
    reject_untracked_execution_files(
        root, label="ProgramBench", run_git=inspect_git
    )
    version = _pyproject_version(project)
    if version != PROGRAMBENCH_VERSION:
        raise ValueError(
            f"ProgramBench checkout must declare version {PROGRAMBENCH_VERSION}, "
            f"found {version}"
        )
    return {
        "project": "ProgramBench",
        "revision": revision,
        "version": version,
        "checkout": str(root),
        "tasks_dir": str(tasks),
        "image_tag": PROGRAMBENCH_IMAGE_TAG,
        "pyproject": immutable_file_identity(
            project, label="ProgramBench pyproject"
        ),
    }


def load_programbench(
    checkout: Path,
    *,
    limit: int | None = None,
    task_ids: Sequence[str] | None = None,
) -> tuple[BenchmarkTask, ...]:
    """Load agent-visible tasks; `tests.json` only ever becomes a hidden hash."""

    if limit is not None:
        _require_positive_int(limit, "ProgramBench limit")
    identity = inspect_programbench_checkout(checkout)
    tasks_root = Path(str(identity["tasks_dir"]))
    selected: frozenset[str] | None = None
    if task_ids is not None:
        if isinstance(task_ids, (str, bytes)):
            raise ValueError("ProgramBench task_ids must be a sequence of ids")
        requested = [
            _require_instance_id(value, "ProgramBench task id") for value in task_ids
        ]
        if not requested or len(requested) != len(set(requested)):
            raise ValueError("ProgramBench task_ids must be unique and non-empty")
        selected = frozenset(requested)
    tasks: list[BenchmarkTask] = []
    seen: set[str] = set()
    for directory in sorted(tasks_root.iterdir()):
        if directory.is_symlink() or not directory.is_dir():
            continue
        task_yaml = directory / "task.yaml"
        if task_yaml.is_symlink() or not task_yaml.is_file():
            continue
        instance_id = _require_instance_id(
            directory.name, "ProgramBench instance_id"
        )
        if selected is not None and instance_id not in selected:
            continue
        hidden_tests = directory / "tests.json"
        if hidden_tests.is_symlink() or not hidden_tests.is_file():
            raise ValueError(
                f"ProgramBench task {instance_id!r} has no hidden tests.json"
            )
        metadata = _read_task_yaml(task_yaml)
        language = _task_field(metadata, "language", task_yaml)
        clean_hashes = metadata.get("eval_clean_hashes") or ()
        if not isinstance(clean_hashes, (list, tuple)):
            raise ValueError(
                f"ProgramBench eval_clean_hashes must be a list: {task_yaml}"
            )
        # tests.json is evaluator-only material. Its bytes never enter task
        # data; only a hash of them is recorded as hidden provenance.
        tests_identity = immutable_file_identity(
            hidden_tests, label="ProgramBench hidden tests"
        )
        yaml_identity = immutable_file_identity(
            task_yaml, label="ProgramBench task metadata"
        )
        seen.add(instance_id)
        tasks.append(
            BenchmarkTask(
                instance_id,
                PROGRAMBENCH_TASK_PROMPT.format(
                    instance_id=instance_id,
                    language=language,
                    workdir=PROGRAMBENCH_WORKDIR,
                ),
                {
                    "benchmark": "programbench",
                    "instance_id": instance_id,
                    "image_name": programbench_image_name(instance_id),
                    "language": language,
                    "difficulty": metadata.get("difficulty"),
                    "repository": metadata.get("repository"),
                    "upstream_commit": metadata.get("commit"),
                    "eval_clean_hash_count": len(clean_hashes),
                    "task_yaml_sha256": yaml_identity["sha256"],
                    "hidden_tests_sha256": tests_identity["sha256"],
                    "hidden_tests_bytes": tests_identity["size_bytes"],
                },
            )
        )
        if selected is None and limit is not None and len(tasks) == limit:
            break
    if selected is not None:
        missing = sorted(selected.difference(seen))
        if missing:
            raise ValueError(
                "ProgramBench task ids are not in the checkout: " + ", ".join(missing)
            )
        if limit is not None:
            tasks = tasks[:limit]
    if not tasks:
        raise ValueError("ProgramBench checkout contains no tasks")
    return tuple(tasks)


async def run_programbench_task(
    task: BenchmarkTask,
    context: RunContext,
    directory: Path,
    *,
    model_factory: ModelFactory,
    system_prompt: str,
    max_steps: int,
    runtime: str = "docker",
    container_runtime: Sequence[str] = ("docker",),
    apptainer_executable: str = "apptainer",
    apptainer_image_cache: Path | None = None,
    scratch_root: Path | None = None,
    overlay_size_mib: int = 16 * 1024,
    image_binding: SWEbenchImageBinding | None = None,
    max_archive_bytes: int = DEFAULT_MAX_ARCHIVE_BYTES,
    multi_agent: bool = False,
    git_share: bool = False,
    harness: str = "single",
    team_size: int | None = None,
    max_active_agents: int = 4,
    max_total_agents: int = 16,
    per_agent_limits: BudgetLimits | None = None,
    agent_spec: AgentSpecV1 | None = None,
) -> EvaluationOutcome:
    """Generate one offline submission archive; never assign a local score."""

    _require_callable(model_factory, "model_factory")
    _require_str(system_prompt, "system_prompt", non_empty=False)
    _require_positive_int(max_steps, "max_steps")
    _require_bool(multi_agent, "multi_agent")
    instance_id = _require_instance_id(
        task.task_id, "ProgramBench task instance_id"
    )
    if task.data.get("image_name") != programbench_image_name(instance_id):
        raise ValueError("ProgramBench task image does not match its instance id")

    if runtime not in {"docker", "apptainer"}:
        raise ValueError("ProgramBench runtime must be docker or apptainer")
    # The cleanroom contract is identical on either backend: the agent's
    # container has no network, and the submission is the whole exported
    # workspace rather than a diff.  The image does ship a .git tree, but the
    # adapter never requires or reads a Git baseline, so a rewritten history
    # cannot change what gets submitted.
    identity = {
        "benchmark": "programbench",
        "benchmark_revision": PROGRAMBENCH_REVISION,
        "benchmark_version": PROGRAMBENCH_VERSION,
        "benchmark_image_tag": PROGRAMBENCH_IMAGE_TAG,
    }

    shared = _shared_git_repository(task.task_id, scratch_root, git_share)

    async def environment_for(agent_id: str) -> BashEnvironment:
        del agent_id
        if runtime == "apptainer":
            return await apptainer_swe_environment(
                task.data,
                image_binding=image_binding,
                executable=apptainer_executable,
                scratch_root=scratch_root,
                image_cache=apptainer_image_cache,
                overlay_size_mib=overlay_size_mib,
                workdir=PROGRAMBENCH_WORKDIR,
                network_disabled=True,
                require_git_baseline=False,
                max_archive_bytes=max_archive_bytes,
                benchmark_identity=identity,
                shared_binds=shared,
            )
        if shared:
            raise ValueError(
                "--agent-git-share needs --runtime apptainer; the Docker "
                "runtime has no bind support"
            )
        return await docker_swe_environment(
            task.data,
            image_binding=image_binding,
            runtime=container_runtime,
            workdir=PROGRAMBENCH_WORKDIR,
            network_disabled=True,
            require_git_baseline=False,
            max_archive_bytes=max_archive_bytes,
            benchmark_identity=identity,
        )

    team = await run_benchmark_team(
        task,
        context,
        environment_factory=environment_for,
        model_factory=model_factory,
        system_prompt=system_prompt,
        max_steps=max_steps,
        agent_spec=agent_spec,
        harness=harness,
        team_size=team_size,
        multi_agent=multi_agent,
        max_active_agents=max_active_agents,
        max_total_agents=max_total_agents,
        per_agent_limits=per_agent_limits,
    )
    if not isinstance(team.state, SWEArchiveState):
        raise RuntimeError("root ProgramBench agent produced no workspace archive")
    archive = team.state.archive

    atomic_bytes(directory / PROGRAMBENCH_SUBMISSION_NAME, archive)
    return EvaluationOutcome(
        task.task_id,
        "completed",
        answer=team.require().answer,
        metadata={
            **team.metadata(),
            "environments": {
                agent_id: dict(base.provenance())
                for agent_id, base in team.bases().items()
            },
            "instance_id": instance_id,
            "submission_artifact": PROGRAMBENCH_SUBMISSION_NAME,
            "submission_bytes": len(archive),
            "submission_sha256": hashlib.sha256(archive).hexdigest(),
            "scoring": "official-programbench-eval-only",
        },
    )


def collect_submissions(output: Path, destination: Path) -> int:
    """Build the exact `<run>/<instance_id>/submission.tar.gz` layout."""

    root, _, artifacts = owned_instance_artifacts(
        output, PROGRAMBENCH_SUBMISSION_NAME, label="ProgramBench"
    )
    target = _submission_collection_target(root, destination)
    records: list[tuple[str, bytes]] = []
    instance_ids: set[str] = set()
    for path in artifacts:
        parent = path.parent
        content = path.read_bytes()
        try:
            declared = strict_json_loads(
                (parent / "result.json").read_text(encoding="utf-8")
            )
            _require_mapping(declared, "result artifact")
            instance_id = _require_instance_id(
                declared.get("task_id"), "ProgramBench result task_id"
            )
            result = read_committed_result(parent, instance_id)
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"ProgramBench submission has no committed result: {path}"
            ) from exc
        if result.get("status") != "completed":
            raise ValueError(f"ProgramBench result is not completed: {path}")
        metadata = result.get("metadata")
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("submission_sha256")
            != hashlib.sha256(content).hexdigest()
        ):
            raise ValueError(f"ProgramBench submission hash does not match: {path}")
        if instance_id in instance_ids:
            raise ValueError(f"duplicate ProgramBench instance_id {instance_id!r}")
        instance_ids.add(instance_id)
        records.append((instance_id, content))
    if not records:
        raise ValueError("evaluation contains no ProgramBench submissions")
    target.mkdir(mode=0o700, exist_ok=True)
    target.chmod(0o700)
    existing: dict[str, Path] = {}
    for candidate in sorted(target.iterdir()):
        if (
            candidate.is_symlink()
            or not candidate.is_dir()
            or sorted(item.name for item in candidate.iterdir())
            not in ([], [PROGRAMBENCH_SUBMISSION_NAME])
        ):
            raise ValueError(
                f"unexpected entry in the ProgramBench collection: {candidate}"
            )
        existing[candidate.name] = candidate
    for instance_id, content in records:
        _submission_collection_target(root, destination, require_exists=True)
        atomic_bytes(target / instance_id / PROGRAMBENCH_SUBMISSION_NAME, content)
    for name, candidate in existing.items():
        if name not in instance_ids:
            _submission_collection_target(root, destination, require_exists=True)
            shutil.rmtree(candidate)
    return len(records)


def _submission_collection_target(
    root: Path, destination: Path, *, require_exists: bool = False
) -> Path:
    target = _require_no_symlink(
        destination.expanduser(), "ProgramBench collection"
    ).resolve()
    if target.parent != root:
        raise ValueError(
            "ProgramBench collection must be a direct child of the evaluation"
        )
    if target.exists() and (target.is_symlink() or not target.is_dir()):
        raise ValueError("ProgramBench collection must be a non-symlink directory")
    if require_exists and not target.is_dir():
        raise ValueError("ProgramBench collection directory disappeared")
    return target


def inspect_programbench_grade_inputs(
    *, run_directory: Path, checkout: Path
) -> Mapping[str, Any]:
    """Validate the official run layout and bind it to visible task prompts."""

    runs_root = _require_no_symlink(
        run_directory.expanduser(), "ProgramBench run directory"
    ).resolve()
    if not runs_root.is_dir():
        raise ValueError("ProgramBench grading requires a run directory")
    submissions: dict[str, Mapping[str, Any]] = {}
    for candidate in sorted(runs_root.iterdir()):
        if candidate.is_symlink() or not candidate.is_dir():
            raise ValueError(
                f"ProgramBench run directory has an unexpected entry: {candidate}"
            )
        if sorted(item.name for item in candidate.iterdir()) != [
            PROGRAMBENCH_SUBMISSION_NAME
        ]:
            raise ValueError(
                "ProgramBench instance directory must hold only "
                f"{PROGRAMBENCH_SUBMISSION_NAME}: {candidate}"
            )
        instance_id = _require_instance_id(
            candidate.name, "ProgramBench run instance id"
        )
        submissions[instance_id] = immutable_file_identity(
            candidate / PROGRAMBENCH_SUBMISSION_NAME,
            label="ProgramBench submission",
        )
    if not submissions:
        raise ValueError("ProgramBench grader input contains no submissions")
    identity = inspect_programbench_checkout(checkout)
    tasks = {task.task_id: task for task in load_programbench(checkout)}
    missing = sorted(set(submissions).difference(tasks))
    if missing:
        raise ValueError(
            "ProgramBench submissions are missing from the checkout: "
            + ", ".join(missing)
        )
    ordered = sorted(submissions)
    return {
        "runs": immutable_tree_identity(
            runs_root, label="ProgramBench submissions"
        ),
        "checkout": identity,
        "prediction_count": len(submissions),
        "task_count": len(tasks),
        "submission_sha256": {
            instance_id: submissions[instance_id]["sha256"]
            for instance_id in ordered
        },
        "task_prompt_sha256": {
            instance_id: hashlib.sha256(
                tasks[instance_id].prompt.encode("utf-8")
            ).hexdigest()
            for instance_id in ordered
        },
        "task_data_sha256": {
            instance_id: hashlib.sha256(
                json.dumps(
                    tasks[instance_id].data,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            for instance_id in ordered
        },
    }


def official_programbench_grader_argv(
    *,
    run_directory: Path,
    output: Path,
    python_executable: str = sys.executable,
    workers: int = 1,
    image_tag: str = PROGRAMBENCH_IMAGE_TAG,
) -> tuple[str, ...]:
    """Build the official `programbench eval` command for one run directory."""

    if (
        not isinstance(workers, int)
        or isinstance(workers, bool)
        or workers < 1
        or not isinstance(python_executable, str)
        or not python_executable
        or "\x00" in python_executable
        or not isinstance(image_tag, str)
        or not _IMAGE_TAG.fullmatch(image_tag)
    ):
        raise ValueError("invalid ProgramBench grader configuration")
    resolved_run = run_directory.expanduser().resolve()
    resolved_output = output.expanduser().resolve()
    if not resolved_run.is_dir() or not resolved_output.parent.is_dir():
        raise ValueError("ProgramBench grader inputs are missing")
    if (
        resolved_output == resolved_run
        or resolved_output.is_relative_to(resolved_run)
        or resolved_run.is_relative_to(resolved_output)
    ):
        raise ValueError(
            "ProgramBench eval output must not overlap the submission snapshot"
        )
    return (
        python_executable,
        "-I",
        "-c",
        _PROGRAMBENCH_CLI_ENTRY,
        "eval",
        str(resolved_run),
        "--output",
        str(resolved_output),
        "--workers",
        str(workers),
        "--image-tag",
        image_tag,
    )


__all__ = [
    "PROGRAMBENCH_DOCKER_ORG",
    "PROGRAMBENCH_IMAGE_TAG",
    "PROGRAMBENCH_REVISION",
    "PROGRAMBENCH_SUBMISSION_NAME",
    "PROGRAMBENCH_TASK_PROMPT",
    "PROGRAMBENCH_VERSION",
    "PROGRAMBENCH_WORKDIR",
    "collect_submissions",
    "inspect_programbench_checkout",
    "inspect_programbench_grade_inputs",
    "load_programbench",
    "official_programbench_grader_argv",
    "programbench_image_name",
    "run_programbench_task",
]
