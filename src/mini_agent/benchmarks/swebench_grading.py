"""Official SWE-bench grading: image verification, argv, and inputs.

Split out of the generation adapter because the two have no runtime overlap:
generation never touches a grader, and grading never runs an agent. Keeping
them together made the adapter look four times the size of the loop it
actually contains.
"""

from __future__ import annotations

import importlib.metadata as metadata
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

from .._hash import canonical_bytes, immutable_file_identity
from ..storage import atomic_bytes, read_committed_result
from ..types import (
    _require_finite_number,
    _require_mapping,
    _require_no_symlink,
    strict_json_loads,
)
from .base import owned_instance_artifacts
from .swebench import (
    SWEBENCH_REVISION,
    SWEBENCH_SOURCE_SHA256,
    SWEBENCH_VERSION,
    _IMAGE_ID,
    _INVALID_SDK,
    _INVALID_SOURCE,
    _UNSAFE_ENTRY,
    _plain_string,
    _require_grader_component,
    _sha256,
    load_swebench,
)


def swebench_grader_image_name(instance_id: Any) -> str:
    """Return the exact default v4.1.0 remote image tag for one task."""

    resolved = _require_grader_component(instance_id, "SWE-bench instance_id")
    key = f"sweb.eval.x86_64.{resolved.casefold()}:latest"
    return f"swebench/{key}".replace("__", "_1776_")


def _grader_image_expectations(
    generation_manifest: Mapping[str, Any],
) -> tuple[list[str], list[tuple[str, str, str, str]]]:
    """Return the recorded runtime and every task's exact expected image ID."""

    _require_mapping(generation_manifest, "SWE-bench generation manifest")
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
        or not all(_plain_string(item) for item in recorded_runtime)
    ):
        raise ValueError("SWE-bench generation container runtime is invalid")
    bindings = adapter.get("image_bindings")
    tasks = generation_manifest.get("tasks")
    if not isinstance(bindings, Mapping) or not isinstance(tasks, list) or not tasks:
        raise ValueError("SWE-bench generation manifest has no Docker image bindings")
    task_ids: list[str] = []
    for task in tasks:
        _require_mapping(task, "SWE-bench generation task")
        task_ids.append(
            _require_grader_component(task.get("id"), "SWE-bench generation task id")
        )
    if len(task_ids) != len(set(task_ids)) or set(bindings) != set(task_ids):
        raise ValueError(
            "SWE-bench Docker image bindings must exactly cover generation tasks"
        )
    expected: list[tuple[str, str, str, str]] = []
    for instance_id in sorted(task_ids):
        binding = bindings[instance_id]
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"runtime", "requested", "identity"}
            or binding.get("runtime") != "docker"
            or not isinstance(binding.get("requested"), str)
            or not _IMAGE_ID.fullmatch(str(binding.get("identity")))
        ):
            raise ValueError(
                f"SWE-bench task {instance_id!r} has an invalid Docker image binding"
            )
        expected.append(
            (
                instance_id,
                swebench_grader_image_name(instance_id),
                str(binding["requested"]),
                str(binding["identity"]),
            )
        )
    return list(recorded_runtime), expected


def _grader_environment(grader_environment: Mapping[str, str]) -> dict[str, str]:
    """Require an explicit, single-engine Docker environment for grading."""

    if not isinstance(grader_environment, Mapping) or not all(
        _plain_string(name) and isinstance(value, str) and "\x00" not in value
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
    return environment


def _require_current_grader_python(python_executable: Any) -> None:
    """Refuse to verify images on behalf of a different interpreter."""

    if not _plain_string(python_executable):
        raise ValueError("SWE-bench grader Python executable is invalid")
    resolved = shutil.which(python_executable)
    try:
        current = resolved is not None and os.path.samefile(resolved, sys.executable)
    except OSError:
        current = False
    if not current:
        raise ValueError(
            "SWE-bench image verification requires the current grader Python"
        )


def _docker_sdk_identity() -> tuple[str, str, str]:
    """Return the installed Docker SDK version and its exact package paths."""

    try:
        version = metadata.version("docker")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(_INVALID_SDK) from exc
    spec = importlib.util.find_spec("docker")
    locations = list(getattr(spec, "submodule_search_locations", None) or ())
    if spec is None or not _plain_string(spec.origin) or len(locations) != 1:
        raise RuntimeError(_INVALID_SDK)
    origin = os.path.abspath(str(spec.origin))
    package_root = os.path.abspath(str(locations[0]))
    if (
        not version
        or os.path.basename(origin) != "__init__.py"
        or os.path.islink(origin)
        or os.path.islink(package_root)
        or os.path.realpath(origin) != origin
        or os.path.realpath(package_root) != package_root
        or os.path.dirname(origin) != package_root
        or not os.path.isfile(origin)
        or not os.path.isdir(package_root)
    ):
        raise RuntimeError(_INVALID_SDK)
    return version, origin, package_root


def verify_swebench_grader_images(
    generation_manifest: Mapping[str, Any],
    *,
    python_executable: str,
    grader_environment: Mapping[str, str],
    timeout_seconds: float = 60.0,
) -> Mapping[str, Any]:
    """Prove every mutable ``:latest`` grader tag still resolves to generation bytes.

    Called before and after the upstream grader subprocess, so a tag that is
    re-pointed mid-run is caught even though grading itself is opaque.
    """

    recorded_runtime, expected = _grader_image_expectations(generation_manifest)
    environment = _grader_environment(grader_environment)
    _require_current_grader_python(python_executable)
    _require_finite_number(
        timeout_seconds, "SWE-bench image verification timeout", exclusive_minimum=0
    )
    version, origin, package_root = _docker_sdk_identity()
    import docker  # type: ignore[import-untyped]

    if (
        os.path.realpath(str(getattr(docker, "__file__", ""))) != origin
        or getattr(docker, "__version__", None) != version
    ):
        raise RuntimeError(_INVALID_SDK)
    try:
        client = docker.from_env(
            environment=environment, timeout=float(timeout_seconds)
        )
    except BaseException as exc:
        raise RuntimeError(
            "could not connect to the official SWE-bench grader Docker engine"
        ) from exc
    verified: list[Mapping[str, str]] = []
    try:
        for instance_id, grader_image, requested, image_id in expected:
            try:
                observed = client.images.get(grader_image).id
            except BaseException as exc:
                raise RuntimeError(
                    f"official SWE-bench grader image {grader_image!r} is unavailable"
                ) from exc
            if not isinstance(observed, str) or not _IMAGE_ID.fullmatch(
                observed.strip().casefold()
            ):
                raise RuntimeError(
                    f"official SWE-bench grader image {grader_image!r} "
                    "returned an invalid image ID"
                )
            if observed.strip().casefold() != image_id:
                raise RuntimeError(
                    f"official SWE-bench grader image {grader_image!r} "
                    "changed identity"
                )
            verified.append(
                {
                    "instance_id": instance_id,
                    "grader_image": grader_image,
                    "generation_requested": requested,
                    "image_id": image_id,
                }
            )
    finally:
        client.close()
    return {
        "engine_contract": "in-process:docker.from_env",
        "docker_sdk_version": version,
        "docker_sdk": {
            "version": version,
            "origin": origin,
            "package_root": package_root,
        },
        "environment_sha256": _sha256(canonical_bytes(environment)),
        "generation_container_runtime": recorded_runtime,
        "images": verified,
    }


def official_grader_argv(
    *,
    predictions: Path,
    dataset_name: str,
    run_id: str,
    max_workers: int = 1,
    python_executable: str = sys.executable,
) -> tuple[str, ...]:
    if (
        not _plain_string(dataset_name)
        or dataset_name.startswith("-")
        or not isinstance(max_workers, int)
        or isinstance(max_workers, bool)
        or max_workers < 1
        or not _plain_string(python_executable)
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
        raise RuntimeError(_INVALID_SOURCE) from exc
    if (
        not root.is_absolute()
        or canonical != root
        or root.is_symlink()
        or not root.is_dir()
        or entry.is_symlink()
        or not entry.is_file()
    ):
        raise RuntimeError(_INVALID_SOURCE)
    candidates: list[Path] = []
    for directory, directory_names, names in os.walk(root, followlinks=False):
        parent = Path(directory)
        retained: list[str] = []
        for name in sorted(directory_names):
            path = parent / name
            if path.is_symlink() or not path.is_dir():
                raise RuntimeError(_UNSAFE_ENTRY.format(kind="directory"))
            if name != "__pycache__":
                retained.append(name)
        directory_names[:] = retained
        for name in sorted(names):
            path = parent / name
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(_UNSAFE_ENTRY.format(kind="source"))
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
    # The digest covers every path, size, and content hash, so a separate file
    # count and total-size assertion could only ever fail with it.
    source_sha256 = _sha256(canonical_bytes(files))
    if source_sha256 != SWEBENCH_SOURCE_SHA256:
        raise RuntimeError(
            "installed SWE-bench package does not match pinned v4.1.0 source"
        )
    return {
        "project": "SWE-bench",
        "version": SWEBENCH_VERSION,
        "revision": SWEBENCH_REVISION,
        "source_root": str(root),
        "source_file_count": len(files),
        "source_size_bytes": sum(int(item["size_bytes"]) for item in files),
        "source_sha256": source_sha256,
    }


def collect_predictions(output: Path, destination: Path) -> int:
    """Build the exact SWE-bench JSONL prediction input from task artifacts."""

    root, _, artifacts = owned_instance_artifacts(
        output, "prediction.json", label="SWE-bench"
    )
    target = _prediction_collection_target(root, destination)
    records: list[dict[str, str]] = []
    instance_ids: set[str] = set()
    for path in artifacts:
        value = strict_json_loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError(f"invalid SWE-bench prediction: {path}")
        model_patch = value.get("model_patch")
        model_name = value.get("model_name_or_path")
        if not isinstance(model_patch, str) or not isinstance(model_name, str):
            raise ValueError(f"incomplete SWE-bench prediction: {path}")
        instance_id = _require_grader_component(
            value.get("instance_id"), "SWE-bench prediction instance_id"
        )
        _require_grader_component(
            model_name.replace("/", "__"), "SWE-bench prediction model_name_or_path"
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
        metadata_value = result.get("metadata")
        digest = _sha256(path.read_bytes())
        if (
            not isinstance(metadata_value, Mapping)
            or metadata_value.get("prediction_sha256") != digest
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
    content = "".join(
        json.dumps(dict(value), sort_keys=True, allow_nan=False) + "\n"
        for value in records
    ).encode("utf-8")
    _prediction_collection_target(root, destination)
    atomic_bytes(target, content)
    return len(records)


def _prediction_collection_target(root: Path, destination: Path) -> Path:
    expanded = _require_no_symlink(
        destination.expanduser(), "SWE-bench predictions destination"
    )
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
    task_prompts: dict[str, str] = {}
    task_data: dict[str, str] = {}
    for task in tasks:
        task_prompts[task.task_id] = _sha256(task.prompt.encode("utf-8"))
        task_data[task.task_id] = _sha256(canonical_bytes(task.data))
    prediction_ids: set[str] = set()
    lines = prediction_path.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        value = strict_json_loads(line)
        _require_mapping(value, f"SWE-bench prediction line {number}")
        instance_id = _require_grader_component(
            value.get("instance_id"),
            f"SWE-bench prediction line {number} instance_id",
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
    selected = sorted(prediction_ids)
    return {
        "predictions": immutable_file_identity(
            prediction_path, label="SWE-bench predictions"
        ),
        "dataset": immutable_file_identity(dataset_path, label="SWE-bench dataset"),
        "prediction_count": len(prediction_ids),
        "dataset_count": len(task_prompts),
        "task_prompt_sha256": {key: task_prompts[key] for key in selected},
        "task_data_sha256": {key: task_data[key] for key in selected},
    }
