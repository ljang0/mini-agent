"""SWE-bench generation and official grader contracts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

from ..environments.swe import SWEPatchState
from ..environments.swebench import (
    ApptainerSWEEnvironment,
    DockerSWEEnvironment,
    SWEBENCH_REVISION,
    SWEbenchImageBinding,
    resolve_swebench_image_binding,
)
from ..models import Model
from ..orchestrator import Orchestrator
from ..runtime import RunContext
from ..specs import AgentSpecV1
from ..types import BudgetLimits, strict_json_loads
from .base import (
    task_agent_builder,
    BenchmarkTask,
    EvaluationOutcome,
    atomic_bytes,
    atomic_json,
    immutable_file_identity,
    read_committed_result,
    raise_after_cleanup,
    task_agent_root,
)


SWEBENCH_VERSION = "4.1.0"
SWEBENCH_SOURCE_SHA256 = (
    "63d4d3d0543de66520fa44f12badddaa810f708a0d780954684c24c7ce075cc8"
)
SWEBENCH_SOURCE_FILE_COUNT = 591
SWEBENCH_SOURCE_SIZE_BYTES = 1_890_632
ModelFactory = Callable[[str], Model | Awaitable[Model]]
_SAFE_GRADER_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,255}")
_DOCKER_PROBE_MAX_BYTES = 4 * 1024 * 1024
_DOCKER_PROBE_SOURCE = r"""
import importlib.metadata as metadata
import importlib.util
import json
import os
import re
import sys

SCHEMA = "mini-agent-isolated-swebench-images-v1"
REQUEST = sys.argv[1]
OUTPUT = sys.argv[2]


class ProbeError(Exception):
    def __init__(self, code, image=""):
        self.code = code
        self.image = image


def emit(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if len(encoded) > 3 * 1024 * 1024:
        raise SystemExit(70)
    descriptor = os.open(OUTPUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def docker_identity():
    try:
        version = metadata.version("docker")
    except metadata.PackageNotFoundError:
        raise ProbeError("sdk")
    spec = importlib.util.find_spec("docker")
    if spec is None or not isinstance(spec.origin, str) or not spec.origin:
        raise ProbeError("sdk")
    locations = list(spec.submodule_search_locations or ())
    if len(locations) != 1 or not isinstance(locations[0], str):
        raise ProbeError("sdk")
    origin = os.path.abspath(spec.origin)
    package_root = os.path.abspath(locations[0])
    if (
        os.path.basename(origin) != "__init__.py"
        or os.path.islink(origin)
        or os.path.islink(package_root)
        or os.path.realpath(origin) != origin
        or os.path.realpath(package_root) != package_root
        or os.path.dirname(origin) != package_root
        or not os.path.isfile(origin)
        or not os.path.isdir(package_root)
    ):
        raise ProbeError("sdk")
    return version, origin, package_root


client = None
failure = None
result = None
try:
    with open(REQUEST, "rb") as stream:
        request = json.load(stream)
    version, origin, package_root = docker_identity()
    import docker

    module_file = getattr(docker, "__file__", None)
    module_version = getattr(docker, "__version__", None)
    if (
        not isinstance(module_file, str)
        or os.path.realpath(module_file) != origin
        or module_version != version
    ):
        raise ProbeError("sdk")
    try:
        client = docker.from_env(
            environment=request["environment"], timeout=request["timeout_seconds"]
        )
    except BaseException:
        raise ProbeError("connect")
    observed_images = []
    for item in request["images"]:
        image_name = item["grader_image"]
        try:
            image = client.images.get(image_name)
        except BaseException:
            raise ProbeError("unavailable", image_name)
        observed = getattr(image, "id", None)
        if not isinstance(observed, str):
            raise ProbeError("invalid_id", image_name)
        observed = observed.strip().casefold()
        if re.fullmatch(r"sha256:[0-9a-f]{64}", observed) is None:
            raise ProbeError("invalid_id", image_name)
        if observed != item["expected_image_id"]:
            raise ProbeError("changed", image_name)
        observed_images.append(
            {"grader_image": image_name, "image_id": observed}
        )
    result = {
        "schema": SCHEMA,
        "ok": True,
        "python": {
            "executable": os.path.abspath(sys.executable),
            "prefix": os.path.abspath(sys.prefix),
            "base_prefix": os.path.abspath(sys.base_prefix),
        },
        "docker_sdk": {
            "version": version,
            "origin": origin,
            "package_root": package_root,
        },
        "images": observed_images,
    }
except ProbeError as error:
    failure = {"code": error.code, "image": error.image}
except BaseException:
    failure = {"code": "runtime", "image": ""}
if client is not None:
    try:
        client.close()
    except BaseException:
        if failure is None:
            failure = {"code": "cleanup", "image": ""}
if failure is not None:
    result = {
        "schema": SCHEMA,
        "ok": False,
        "error": failure["code"],
        "image": failure["image"],
    }
emit(result)
"""


def _require_grader_component(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_GRADER_COMPONENT.fullmatch(value):
        raise ValueError(f"{label} must be one path-safe component")
    return value


def swebench_grader_image_name(instance_id: Any) -> str:
    """Return the exact default v4.1.0 remote image tag for one task."""

    resolved = _require_grader_component(instance_id, "SWE-bench instance_id")
    key = f"sweb.eval.x86_64.{resolved.casefold()}:latest"
    return f"swebench/{key}".replace("__", "_1776_")


def verify_swebench_grader_images(
    generation_manifest: Mapping[str, Any],
    *,
    python_executable: str,
    grader_environment: Mapping[str, str],
    timeout_seconds: float = 60.0,
) -> Mapping[str, Any]:
    """Inspect tags through the exact Docker SDK/daemon used by the grader."""

    if not isinstance(generation_manifest, Mapping):
        raise ValueError("SWE-bench generation manifest must be an object")
    config = generation_manifest.get("config")
    adapter = config.get("adapter") if isinstance(config, Mapping) else None
    if not isinstance(adapter, Mapping) or adapter.get("runtime") != "docker":
        raise ValueError(
            "official SWE-bench grading requires Docker generation image bindings"
        )
    recorded_runtime = adapter.get("container_runtime")
    if (
        not isinstance(recorded_runtime, list)
        or not recorded_runtime
        or not all(
            isinstance(item, str) and item and "\x00" not in item
            for item in recorded_runtime
        )
    ):
        raise ValueError("SWE-bench generation container runtime is invalid")
    if not isinstance(grader_environment, Mapping) or not all(
        isinstance(name, str)
        and name
        and "\x00" not in name
        and isinstance(value, str)
        and "\x00" not in value
        for name, value in grader_environment.items()
    ):
        raise ValueError("SWE-bench grader environment must contain strings")
    environment = dict(grader_environment)
    if not environment.get("DOCKER_HOST"):
        raise ValueError(
            "SWE-bench grading requires an explicit DOCKER_HOST so image "
            "verification and the upstream grader address the same engine"
        )
    if environment.get("DOCKER_TLS_VERIFY") is not None and not environment.get(
        "DOCKER_CERT_PATH"
    ):
        raise ValueError(
            "DOCKER_TLS_VERIFY requires an explicit DOCKER_CERT_PATH for exact "
            "SWE-bench grader-engine binding"
        )
    bindings = adapter.get("image_bindings")
    tasks = generation_manifest.get("tasks")
    if not isinstance(bindings, Mapping) or not isinstance(tasks, list) or not tasks:
        raise ValueError("SWE-bench generation manifest has no Docker image bindings")
    task_ids: list[str] = []
    for task in tasks:
        if not isinstance(task, Mapping):
            raise ValueError("SWE-bench generation task identity is invalid")
        task_ids.append(
            _require_grader_component(task.get("id"), "SWE-bench generation task id")
        )
    if len(task_ids) != len(set(task_ids)) or set(bindings) != set(task_ids):
        raise ValueError(
            "SWE-bench Docker image bindings must exactly cover generation tasks"
        )
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or timeout_seconds <= 0
    ):
        raise ValueError("SWE-bench image verification timeout must be positive")

    expected_images: list[tuple[str, str, str]] = []
    for instance_id in sorted(task_ids):
        binding = bindings[instance_id]
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"runtime", "requested", "identity"}
            or binding.get("runtime") != "docker"
            or not isinstance(binding.get("requested"), str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(binding.get("identity")))
        ):
            raise ValueError(
                f"SWE-bench task {instance_id!r} has an invalid Docker image binding"
            )
        expected_images.append(
            (
                instance_id,
                swebench_grader_image_name(instance_id),
                str(binding["identity"]),
            )
        )

    if (
        not isinstance(python_executable, str)
        or not python_executable
        or "\x00" in python_executable
    ):
        raise ValueError("SWE-bench grader Python executable is invalid")
    executable_value = shutil.which(python_executable)
    try:
        current_python = executable_value is not None and os.path.samefile(
            executable_value, sys.executable
        )
    except OSError:
        current_python = False
    if not current_python:
        raise ValueError(
            "SWE-bench image verification requires the current grader Python"
        )
    assert executable_value is not None
    # Preserve the virtual-environment spelling: resolving this symlink would
    # execute the base interpreter with a different installed package set.
    executable = Path(executable_value).absolute()

    request_value = {
        "schema": "mini-agent-isolated-swebench-images-request-v1",
        "environment": environment,
        "timeout_seconds": float(timeout_seconds),
        "images": [
            {"grader_image": grader_image, "expected_image_id": expected}
            for _, grader_image, expected in expected_images
        ],
    }
    request_encoded = json.dumps(
        request_value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if len(request_encoded) > _DOCKER_PROBE_MAX_BYTES:
        raise ValueError("SWE-bench image verification request is too large")
    with tempfile.TemporaryDirectory(prefix="mini-agent-swebench-images-") as temporary:
        probe_root = Path(temporary)
        probe_root.chmod(0o700)
        request_path = probe_root / "request.json"
        output_path = probe_root / "identity.json"
        descriptor = os.open(
            request_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(request_encoded)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(descriptor)
        try:
            completed = subprocess.run(
                (
                    str(executable),
                    "-I",
                    "-c",
                    _DOCKER_PROBE_SOURCE,
                    str(request_path),
                    str(output_path),
                ),
                cwd=probe_root,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=environment,
                timeout=float(timeout_seconds),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(
                "isolated SWE-bench grader image verification failed"
            ) from exc
        if completed.returncode != 0:
            raise RuntimeError("isolated SWE-bench grader image verification failed")
        value = _read_docker_probe_output(output_path)

    if value.get("schema") != "mini-agent-isolated-swebench-images-v1":
        raise RuntimeError("isolated SWE-bench grader image identity is invalid")
    if value.get("ok") is not True:
        code = value.get("error")
        image = value.get("image")
        if not isinstance(image, str) or image not in {
            item[1] for item in expected_images
        }:
            image = ""
        if code == "connect":
            raise RuntimeError(
                "could not connect to the official SWE-bench grader Docker engine"
            )
        if code == "unavailable" and image:
            raise RuntimeError(
                f"official SWE-bench grader image {image!r} is unavailable"
            )
        if code == "invalid_id" and image:
            raise RuntimeError(
                f"official SWE-bench grader image {image!r} returned an invalid "
                "image ID"
            )
        if code == "changed" and image:
            raise RuntimeError(
                f"official SWE-bench grader image {image!r} changed identity"
            )
        if code == "cleanup":
            raise RuntimeError("SWE-bench grader Docker client cleanup failed")
        raise RuntimeError("isolated SWE-bench grader image identity is invalid")
    if set(value) != {"schema", "ok", "python", "docker_sdk", "images"}:
        raise RuntimeError("isolated SWE-bench grader image identity is invalid")
    python_identity = value.get("python")
    if (
        not isinstance(python_identity, Mapping)
        or set(python_identity) != {"executable", "prefix", "base_prefix"}
        or python_identity.get("executable") != str(executable)
        or python_identity.get("prefix") != str(Path(sys.prefix).absolute())
        or python_identity.get("base_prefix")
        != str(Path(sys.base_prefix).absolute())
    ):
        raise RuntimeError("isolated SWE-bench grader Python identity is invalid")
    docker_sdk = value.get("docker_sdk")
    observed_images = value.get("images")
    if not isinstance(docker_sdk, Mapping) or set(docker_sdk) != {
        "version",
        "origin",
        "package_root",
    }:
        raise RuntimeError("isolated SWE-bench Docker SDK identity is invalid")
    docker_version = docker_sdk.get("version")
    docker_origin = docker_sdk.get("origin")
    docker_root = docker_sdk.get("package_root")
    if (
        not isinstance(docker_version, str)
        or not docker_version
        or not isinstance(docker_origin, str)
        or not isinstance(docker_root, str)
    ):
        raise RuntimeError("isolated SWE-bench Docker SDK identity is invalid")
    origin_path = Path(docker_origin)
    root_path = Path(docker_root)
    try:
        valid_sdk_paths = (
            origin_path.is_absolute()
            and root_path.is_absolute()
            and origin_path.resolve(strict=True) == origin_path
            and root_path.resolve(strict=True) == root_path
            and origin_path.parent == root_path
            and origin_path.name == "__init__.py"
            and origin_path.is_file()
            and root_path.is_dir()
        )
    except OSError:
        valid_sdk_paths = False
    if not valid_sdk_paths:
        raise RuntimeError("isolated SWE-bench Docker SDK identity is invalid")
    if not isinstance(observed_images, list) or len(observed_images) != len(
        expected_images
    ):
        raise RuntimeError("isolated SWE-bench grader image identity is invalid")
    verified: list[Mapping[str, str]] = []
    for observed_value, expected_value in zip(observed_images, expected_images):
        instance_id, grader_image, expected = expected_value
        if (
            not isinstance(observed_value, Mapping)
            or set(observed_value) != {"grader_image", "image_id"}
            or observed_value.get("grader_image") != grader_image
            or observed_value.get("image_id") != expected
        ):
            raise RuntimeError("isolated SWE-bench grader image identity is invalid")
        binding = bindings[instance_id]
        assert isinstance(binding, Mapping)
        verified.append(
            {
                "instance_id": instance_id,
                "grader_image": grader_image,
                "generation_requested": str(binding["requested"]),
                "image_id": expected,
            }
        )

    environment_encoded = json.dumps(
        environment, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        "engine_contract": "isolated-python-I:docker.from_env",
        "docker_sdk_version": docker_version,
        "docker_sdk": {
            "version": docker_version,
            "origin": str(origin_path),
            "package_root": str(root_path),
        },
        "environment_sha256": hashlib.sha256(environment_encoded).hexdigest(),
        "generation_container_runtime": list(recorded_runtime),
        "images": verified,
    }


def _read_docker_probe_output(path: Path) -> Mapping[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(
            "isolated SWE-bench grader image verification produced no identity"
        ) from exc
    try:
        status = os.fstat(descriptor)
        if (
            not stat.S_ISREG(status.st_mode)
            or status.st_nlink != 1
            or status.st_uid != os.geteuid()
            or stat.S_IMODE(status.st_mode) & 0o077
            or status.st_size < 1
            or status.st_size > _DOCKER_PROBE_MAX_BYTES
        ):
            raise RuntimeError(
                "isolated SWE-bench grader image identity is invalid"
            )
        chunks: list[bytes] = []
        remaining = _DOCKER_PROBE_MAX_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 65536))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        encoded = b"".join(chunks)
    finally:
        os.close(descriptor)
    if len(encoded) != status.st_size:
        raise RuntimeError("isolated SWE-bench grader image identity changed")
    try:
        value = strict_json_loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(
            "isolated SWE-bench grader image identity is invalid"
        ) from exc
    if not isinstance(value, Mapping):
        raise RuntimeError("isolated SWE-bench grader image identity is invalid")
    return value


def load_swebench(path: Path, *, limit: int | None = None) -> tuple[BenchmarkTask, ...]:
    if limit is not None and (
        not isinstance(limit, int) or isinstance(limit, bool) or limit < 1
    ):
        raise ValueError("SWE-bench limit must be positive")
    source = path.expanduser().resolve()
    rows: list[tuple[int, Any]] = []
    if source.suffix.casefold() == ".json":
        try:
            value = strict_json_loads(source.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError("SWE-bench JSON dataset is invalid") from exc
        if not isinstance(value, list):
            raise ValueError("SWE-bench .json dataset must contain an array")
        rows = list(enumerate(value, 1))
    else:
        for number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                item = strict_json_loads(line)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"SWE-bench line {number} is invalid JSON") from exc
            rows.append((number, item))

    tasks: list[BenchmarkTask] = []
    seen: set[str] = set()
    for number, item in rows:
        if not isinstance(item, Mapping):
            raise ValueError(f"SWE-bench line {number} must be an object")
        instance_id = item.get("instance_id")
        problem = item.get("problem_statement")
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError(f"SWE-bench line {number} is missing required fields")
        instance_id = _require_grader_component(
            instance_id, f"SWE-bench line {number} instance_id"
        )
        if instance_id in seen:
            raise ValueError(f"duplicate SWE-bench instance_id {instance_id!r}")
        seen.add(instance_id)
        tasks.append(BenchmarkTask(instance_id, problem, dict(item)))
        if limit is not None and len(tasks) == limit:
            break
    if not tasks:
        raise ValueError("SWE-bench task file contains no instances")
    return tuple(tasks)


async def prepare_swebench_image_bindings(
    tasks: Sequence[BenchmarkTask],
    *,
    runtime: str,
    container_runtime: Sequence[str] = ("docker",),
    apptainer_executable: str = "apptainer",
    apptainer_image_cache: Path | None = None,
) -> Mapping[str, SWEbenchImageBinding]:
    """Resolve every selected task image before a run manifest is committed."""

    if isinstance(tasks, (str, bytes)) or not tasks:
        raise ValueError("SWE-bench image preflight requires tasks")
    bindings: dict[str, SWEbenchImageBinding] = {}
    by_requested: dict[str, SWEbenchImageBinding] = {}
    for task in tasks:
        if not isinstance(task, BenchmarkTask):
            raise ValueError("SWE-bench image preflight requires benchmark tasks")
        if task.task_id in bindings:
            raise ValueError(f"duplicate SWE-bench task id {task.task_id!r}")
        requested = task.data.get("image_name") or task.data.get("docker_image")
        cache_key = (
            requested if isinstance(requested, str) else "instance:" + task.task_id
        )
        binding = by_requested.get(cache_key)
        if binding is None:
            binding = await resolve_swebench_image_binding(
                task.data,
                runtime=runtime,
                container_runtime=container_runtime,
                apptainer_executable=apptainer_executable,
                apptainer_image_cache=apptainer_image_cache,
            )
            by_requested[cache_key] = binding
        bindings[task.task_id] = binding
    return bindings


async def run_swebench_task(
    task: BenchmarkTask,
    context: RunContext,
    directory: Path,
    *,
    model_factory: ModelFactory,
    system_prompt: str,
    max_steps: int,
    runtime: str,
    model_name: str = "mini-agent",
    scratch_root: Path | None = None,
    apptainer_image_cache: Path | None = None,
    image_binding: SWEbenchImageBinding | None = None,
    overlay_size_mib: int = 16 * 1024,
    container_runtime: Sequence[str] = ("docker",),
    apptainer_executable: str = "apptainer",
    multi_agent: bool = False,
    max_active_agents: int = 4,
    max_total_agents: int = 16,
    per_agent_limits: BudgetLimits | None = None,
    agent_spec: AgentSpecV1 | None = None,
) -> EvaluationOutcome:
    if not callable(model_factory):
        raise ValueError("model_factory must be callable")
    if not isinstance(system_prompt, str):
        raise ValueError("system_prompt must be a string")
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be non-empty")
    _require_grader_component(task.task_id, "SWE-bench task instance_id")
    _require_grader_component(
        model_name.replace("/", "__"),
        "SWE-bench model_name_or_path",
    )
    if not isinstance(multi_agent, bool):
        raise ValueError("multi_agent must be boolean")
    if runtime not in {"docker", "apptainer"}:
        raise ValueError("SWE-bench runtime must be docker or apptainer")

    async def environment_for(agent_id: str) -> Any:
        del agent_id
        if runtime == "docker":
            return await DockerSWEEnvironment.create(
                task.data,
                image_binding=image_binding,
                runtime=container_runtime,
            )
        return await ApptainerSWEEnvironment.create(
            task.data,
            image_binding=image_binding,
            executable=apptainer_executable,
            scratch_root=scratch_root,
            image_cache=apptainer_image_cache,
            overlay_size_mib=overlay_size_mib,
        )

    agent_for = task_agent_builder(
        model_factory=model_factory,
        system_prompt=system_prompt,
        max_steps=max_steps,
        agent_spec=agent_spec,
    )

    if multi_agent:
        root_id = task_agent_root(task.task_id)
        orchestrator = Orchestrator(
            agent_builder=agent_for,
            environment_factory=environment_for,
            context=context,
            max_active_agents=max_active_agents,
            max_total_agents=max_total_agents,
            per_agent_limits=per_agent_limits,
            root_id=root_id,
        )
        result = await orchestrator.run(task.prompt)
        root_state = orchestrator.records[root_id].state
        if not isinstance(root_state, SWEPatchState):
            raise RuntimeError("root SWE agent produced no patch state")
        patch = root_state.patch
        metadata: Mapping[str, Any] = {
            "mode": "multi",
            "runtime": runtime,
            "agents": {
                agent_id: record.status
                for agent_id, record in orchestrator.records.items()
            },
            "environments": {
                agent_id: dict(record.environment.base.provenance())
                for agent_id, record in orchestrator.records.items()
                if record.environment is not None
            },
        }
    else:
        root_id = task_agent_root(task.task_id)
        environment = await environment_for(root_id)
        operation_error: BaseException | None = None
        result = None
        patch = b""
        metadata = {}
        try:
            agent = await agent_for(root_id, environment, context)
            result = await agent.run(task.prompt)
            patch = await environment.export_patch()
            metadata = {
                "mode": "single",
                "runtime": runtime,
                "environment": dict(environment.provenance()),
            }
        except BaseException as exc:
            operation_error = exc
        cleanup_error: BaseException | None = None
        try:
            await environment.close()
        except BaseException as exc:
            cleanup_error = exc
        raise_after_cleanup("SWE-bench generation", operation_error, cleanup_error)
    assert result is not None
    atomic_bytes(directory / "patch.diff", patch)
    prediction = {
        "instance_id": task.task_id,
        "model_patch": patch.decode("utf-8", errors="strict"),
        "model_name_or_path": model_name,
    }
    atomic_json(directory / "prediction.json", prediction)
    artifact_metadata = {
        **metadata,
        "patch_bytes": len(patch),
        "patch_sha256": hashlib.sha256(patch).hexdigest(),
        "prediction_sha256": hashlib.sha256(
            (directory / "prediction.json").read_bytes()
        ).hexdigest(),
    }
    return EvaluationOutcome(
        task.task_id,
        "completed",
        answer=result.answer,
        metadata=artifact_metadata,
    )


def official_grader_argv(
    *,
    predictions: Path,
    dataset_name: str,
    run_id: str,
    max_workers: int = 1,
    python_executable: str = sys.executable,
) -> tuple[str, ...]:
    if (
        not isinstance(dataset_name, str)
        or not dataset_name
        or dataset_name.startswith("-")
        or "\x00" in dataset_name
        or not isinstance(max_workers, int)
        or isinstance(max_workers, bool)
        or max_workers < 1
        or not isinstance(python_executable, str)
        or not python_executable
        or "\x00" in python_executable
    ):
        raise ValueError("invalid SWE-bench grader configuration")
    _require_grader_component(run_id, "SWE-bench run_id")
    return (
        python_executable,
        "-I",
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name,
        "--predictions_path",
        str(predictions.expanduser().resolve()),
        "--max_workers",
        str(max_workers),
        "--run_id",
        run_id,
    )


def swebench_grader_source_identity(source_root: Path) -> Mapping[str, Any]:
    """Hash the explicit package root resolved by the isolated grader Python."""

    if not isinstance(source_root, Path):
        raise TypeError("SWE-bench package source root must be a path")
    root = source_root.expanduser()
    entry = root / "__init__.py"
    try:
        canonical = root.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("official SWE-bench package source tree is invalid") from exc
    if (
        not root.is_absolute()
        or canonical != root
        or root.is_symlink()
        or not root.is_dir()
        or entry.is_symlink()
        or not entry.is_file()
    ):
        raise RuntimeError("official SWE-bench package source tree is invalid")
    candidates: list[Path] = []
    for directory, directory_names, names in os.walk(root, followlinks=False):
        parent = Path(directory)
        retained: list[str] = []
        for name in sorted(directory_names):
            path = parent / name
            if path.is_symlink() or not path.is_dir():
                raise RuntimeError(
                    "official SWE-bench package contains an unsafe directory"
                )
            if name != "__pycache__":
                retained.append(name)
        directory_names[:] = retained
        for name in sorted(names):
            path = parent / name
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(
                    "official SWE-bench package contains an unsafe source"
                )
            candidates.append(path)
    files: list[Mapping[str, Any]] = []
    for path in sorted(candidates, key=lambda item: item.relative_to(root).as_posix()):
        identity = immutable_file_identity(path, label="SWE-bench package source")
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": identity["size_bytes"],
                "sha256": identity["sha256"],
            }
        )
    encoded = json.dumps(
        files, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    source_sha256 = hashlib.sha256(encoded).hexdigest()
    size_bytes = sum(int(item["size_bytes"]) for item in files)
    if (
        source_sha256 != SWEBENCH_SOURCE_SHA256
        or len(files) != SWEBENCH_SOURCE_FILE_COUNT
        or size_bytes != SWEBENCH_SOURCE_SIZE_BYTES
    ):
        raise RuntimeError(
            "installed SWE-bench package does not match pinned v4.1.0 source"
        )
    return {
        "project": "SWE-bench",
        "version": SWEBENCH_VERSION,
        "revision": SWEBENCH_REVISION,
        "source_root": str(root),
        "source_file_count": len(files),
        "source_size_bytes": size_bytes,
        "source_sha256": source_sha256,
    }


def collect_predictions(output: Path, destination: Path) -> int:
    """Build the exact SWE-bench JSONL prediction input from task artifacts."""

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
            "SWE-bench evaluation and instances must be non-symlink directories"
        )
    target = _prediction_collection_target(root, destination)
    records: list[dict[str, str]] = []
    instance_ids: set[str] = set()
    for path in sorted(instances.glob("*/prediction.json")):
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
                "SWE-bench prediction must be an owned regular instance artifact"
            )
        value = strict_json_loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError(f"invalid SWE-bench prediction: {path}")
        instance_id = value.get("instance_id")
        model_patch = value.get("model_patch")
        model_name = value.get("model_name_or_path")
        if not isinstance(model_patch, str) or not isinstance(model_name, str):
            raise ValueError(f"incomplete SWE-bench prediction: {path}")
        instance_id = _require_grader_component(
            instance_id, "SWE-bench prediction instance_id"
        )
        _require_grader_component(
            model_name.replace("/", "__"),
            "SWE-bench prediction model_name_or_path",
        )
        if instance_id in instance_ids:
            raise ValueError(f"duplicate SWE-bench instance_id {instance_id!r}")
        try:
            result = read_committed_result(path.parent, instance_id)
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"SWE-bench prediction has no committed result: {path}"
            ) from exc
        if result.get("status") != "completed":
            raise ValueError(f"SWE-bench prediction result is not completed: {path}")
        metadata = result.get("metadata")
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("prediction_sha256")
            != hashlib.sha256(path.read_bytes()).hexdigest()
        ):
            raise ValueError(f"SWE-bench prediction hash does not match: {path}")
        instance_ids.add(instance_id)
        records.append(
            {
                "instance_id": instance_id,
                "model_patch": model_patch,
                "model_name_or_path": model_name,
            }
        )
    if not records:
        raise ValueError("evaluation contains no SWE-bench predictions")
    records.sort(key=lambda value: str(value["instance_id"]))
    content = b"".join(
        (json.dumps(dict(value), sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        )
        for value in records
    )
    _prediction_collection_target(root, destination)
    atomic_bytes(target, content)
    return len(records)


def _prediction_collection_target(root: Path, destination: Path) -> Path:
    expanded = destination.expanduser()
    if expanded.is_symlink():
        raise ValueError("SWE-bench predictions destination must not be a symlink")
    if expanded.parent.resolve() != root:
        raise ValueError(
            "SWE-bench predictions must be a direct child of the evaluation"
        )
    target = root / expanded.name
    if target.exists() and (target.is_symlink() or not target.is_file()):
        raise ValueError("SWE-bench predictions destination must be a regular file")
    return target


def inspect_swebench_grade_inputs(
    *, predictions: Path, dataset: Path
) -> Mapping[str, Any]:
    """Validate local official-grader inputs and bind visible task prompts."""

    prediction_path = predictions.expanduser().resolve()
    dataset_path = dataset.expanduser().resolve()
    if dataset_path.suffix.casefold() not in {".json", ".jsonl"}:
        raise ValueError("SWE-bench grading requires a local .json or .jsonl dataset")
    tasks = load_swebench(dataset_path)
    task_prompts = {
        task.task_id: hashlib.sha256(task.prompt.encode("utf-8")).hexdigest()
        for task in tasks
    }
    task_data = {
        task.task_id: hashlib.sha256(
            json.dumps(
                task.data,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        for task in tasks
    }
    prediction_ids: set[str] = set()
    for number, line in enumerate(
        prediction_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = strict_json_loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"SWE-bench prediction line {number} must be an object")
        instance_id = value.get("instance_id")
        instance_id = _require_grader_component(
            instance_id, f"SWE-bench prediction line {number} instance_id"
        )
        if instance_id in prediction_ids:
            raise ValueError(f"duplicate SWE-bench prediction {instance_id!r}")
        prediction_ids.add(instance_id)
    if not prediction_ids:
        raise ValueError("SWE-bench grader input contains no predictions")
    missing = sorted(prediction_ids.difference(task_prompts))
    if missing:
        raise ValueError(
            "SWE-bench predictions are missing from the local dataset: "
            + ", ".join(missing)
        )
    return {
        "predictions": immutable_file_identity(
            prediction_path, label="SWE-bench predictions"
        ),
        "dataset": immutable_file_identity(dataset_path, label="SWE-bench dataset"),
        "prediction_count": len(prediction_ids),
        "dataset_count": len(task_prompts),
        "task_prompt_sha256": {
            instance_id: task_prompts[instance_id]
            for instance_id in sorted(prediction_ids)
        },
        "task_data_sha256": {
            instance_id: task_data[instance_id]
            for instance_id in sorted(prediction_ids)
        },
    }


__all__ = [
    "SWEBENCH_VERSION",
    "SWEBENCH_REVISION",
    "SWEBENCH_SOURCE_SHA256",
    "collect_predictions",
    "inspect_swebench_grade_inputs",
    "load_swebench",
    "official_grader_argv",
    "prepare_swebench_image_bindings",
    "run_swebench_task",
    "swebench_grader_source_identity",
    "swebench_grader_image_name",
    "verify_swebench_grader_images",
]
