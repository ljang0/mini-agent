"""Official-grader orchestration: isolated runtimes, snapshots, inventories.

Extracted from the CLI boundary; every helper here is security-relevant to
`mini-agent grade` and is exercised directly by the test suite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Mapping

from ._hash import (
    canonical_bytes,
    harness_identity,
    immutable_file_identity,
    immutable_tree_identity,
)
from .storage import atomic_json
from .types import _require_finite_number, _require_mapping, strict_json_loads

_FILE_IDENTITY_FIELDS = ("size_bytes", "sha256")
_TREE_IDENTITY_FIELDS = ("file_count", "size_bytes", "sha256")


def _required_path(value: Path | None, option: str) -> Path:
    if value is None:
        raise ValueError(f"{option} is required")
    expanded = value.expanduser()
    if expanded.is_symlink():
        raise ValueError(f"{option} must not be a symlink: {expanded}")
    resolved = expanded.resolve()
    if not resolved.exists():
        raise ValueError(f"{option} does not exist: {resolved}")
    return resolved


def _grade(args: argparse.Namespace) -> int:
    evaluation_argument = args.evaluation.expanduser()
    evaluation = evaluation_argument.resolve()
    if evaluation_argument.is_symlink() or not evaluation.is_dir():
        raise ValueError("--evaluation must be a non-symlink directory")
    generation_manifest_path = evaluation / "manifest.json"
    if generation_manifest_path.is_symlink() or not generation_manifest_path.is_file():
        raise ValueError("evaluation manifest must be a non-symlink file")
    generation_manifest = _require_mapping(
        strict_json_loads(generation_manifest_path.read_text(encoding="utf-8")),
        "evaluation manifest",
    )
    _verify_evaluation_manifest(generation_manifest, args.benchmark)
    grade_output = _grade_output_path(evaluation, args.output, args.benchmark)
    if grade_output.exists() and (
        not grade_output.is_dir() or any(grade_output.iterdir())
    ):
        raise ValueError(f"grade output is not empty: {grade_output}")
    grader_environment = _official_grader_environment(args.benchmark, grade_output)
    isolated_runtime = _grader_runtime_identity(
        args.python_executable,
        args.benchmark,
        grader_environment=grader_environment,
    )
    runtime = {
        **isolated_runtime,
        "environment": {
            "policy": "allowlist-v1",
            "value_sha256": {
                name: hashlib.sha256(value.encode("utf-8")).hexdigest()
                for name, value in sorted(grader_environment.items())
            },
        },
    }
    verify_grader_assets: Callable[[], Mapping[str, Any]] | None = None

    def observed_grader_runtime(project: str) -> Mapping[str, Any]:
        observed = _grader_runtime_identity(
            args.python_executable,
            args.benchmark,
            grader_environment=grader_environment,
        )
        if observed != isolated_runtime:
            raise RuntimeError(f"{project} grader runtime changed during grading")
        return observed

    if args.benchmark == "swebench":
        from .benchmarks.swebench import (
            collect_predictions,
            inspect_swebench_grade_inputs,
            official_grader_argv,
            swebench_grader_source_identity,
            verify_swebench_grader_images,
        )

        if args.dataset is None:
            raise ValueError(
                "SWE-bench grading requires --dataset with a local .json/.jsonl file"
            )
        dataset = _required_path(args.dataset, "--dataset")
        predictions = evaluation / "predictions.jsonl"
        _reject_grade_output_overlap(
            grade_output, evaluation / "instances", "evaluation instances"
        )
        collect_predictions(evaluation, predictions)
        source_inputs = inspect_swebench_grade_inputs(
            predictions=predictions, dataset=dataset
        )
        _verify_grade_prompt_binding(generation_manifest, source_inputs)
        _create_private_grade_output(grade_output)
        input_root = grade_output / "inputs" / "swebench"
        snapshot_predictions = input_root / "predictions.jsonl"
        snapshot_dataset = input_root / ("dataset" + dataset.suffix.casefold())
        _snapshot_grade_input(
            predictions,
            snapshot_predictions,
            source_inputs["predictions"],
            label="SWE-bench predictions",
        )
        _snapshot_grade_input(
            dataset,
            snapshot_dataset,
            source_inputs["dataset"],
            label="SWE-bench dataset",
        )
        inputs = {
            **inspect_swebench_grade_inputs(
                predictions=snapshot_predictions, dataset=snapshot_dataset
            ),
            "sources": source_inputs,
        }
        _verify_grade_prompt_binding(generation_manifest, inputs)
        argv = official_grader_argv(
            predictions=snapshot_predictions,
            dataset_name=str(snapshot_dataset),
            run_id=args.run_id,
            max_workers=args.max_workers,
            python_executable=str(isolated_runtime["python_executable"]),
        )
        swebench_root = _isolated_module_root(isolated_runtime, "swebench")
        grader_source = swebench_grader_source_identity(swebench_root)
        grader_images = verify_swebench_grader_images(
            generation_manifest,
            python_executable=str(isolated_runtime["python_executable"]),
            grader_environment=grader_environment,
        )
        grader = {
            "project": "SWE-bench",
            "version": runtime["packages"]["swebench"],
            "source": grader_source,
            "images": grader_images,
        }

        def verify_swebench_grader_assets() -> Mapping[str, Any]:
            observed_runtime = observed_grader_runtime("SWE-bench")
            observed_source = swebench_grader_source_identity(
                _isolated_module_root(observed_runtime, "swebench")
            )
            if observed_source != grader_source:
                raise RuntimeError("SWE-bench grader source changed during grading")
            observed_images = verify_swebench_grader_images(
                generation_manifest,
                python_executable=str(observed_runtime["python_executable"]),
                grader_environment=grader_environment,
            )
            if observed_images != grader_images:
                raise RuntimeError("SWE-bench grader images changed during grading")
            return {
                "runtime": observed_runtime,
                "source": observed_source,
                "images": observed_images,
            }

        verify_grader_assets = verify_swebench_grader_assets
    elif args.benchmark == "programbench":
        from .benchmarks.programbench import (
            PROGRAMBENCH_IMAGE_TAG,
            PROGRAMBENCH_REVISION,
            PROGRAMBENCH_VERSION,
            collect_submissions,
            inspect_programbench_checkout,
            inspect_programbench_grade_inputs,
            official_programbench_grader_argv,
        )

        checkout = _required_path(args.checkout, "--checkout")
        run_directory = evaluation / "official_run"
        for protected, label in (
            (evaluation / "instances", "evaluation instances"),
            (run_directory, "collected ProgramBench submissions"),
            (checkout, "ProgramBench checkout"),
        ):
            _reject_grade_output_overlap(grade_output, protected, label)
        collect_submissions(evaluation, run_directory)
        source_inputs = inspect_programbench_grade_inputs(
            run_directory=run_directory, checkout=checkout
        )
        _verify_grade_prompt_binding(generation_manifest, source_inputs)
        _create_private_grade_output(grade_output)
        snapshot_run = grade_output / "inputs" / "programbench" / "run"
        _snapshot_grade_input(
            run_directory,
            snapshot_run,
            source_inputs["runs"],
            label="ProgramBench submissions",
            tree=True,
        )
        inputs = {
            **inspect_programbench_grade_inputs(
                run_directory=snapshot_run, checkout=checkout
            ),
            "sources": source_inputs,
        }
        _verify_grade_prompt_binding(generation_manifest, inputs)
        eval_dir = _grade_eval_directory(grade_output, args.eval_dir)
        eval_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        argv = official_programbench_grader_argv(
            run_directory=snapshot_run,
            output=eval_dir,
            python_executable=str(isolated_runtime["python_executable"]),
            workers=args.max_workers,
        )
        grader_checkout = inspect_programbench_checkout(checkout)
        grader = {
            "project": "ProgramBench",
            "revision": PROGRAMBENCH_REVISION,
            "version": runtime["packages"]["programbench"],
            "image_tag": PROGRAMBENCH_IMAGE_TAG,
            "checkout": grader_checkout,
            "package_root": str(
                _isolated_module_root(isolated_runtime, "programbench")
            ),
        }
        if grader_checkout["version"] != PROGRAMBENCH_VERSION:
            raise ValueError(
                f"official ProgramBench grading requires {PROGRAMBENCH_VERSION}"
            )

        def verify_programbench_grader_assets() -> Mapping[str, Any]:
            observed_runtime = observed_grader_runtime("ProgramBench")
            observed_checkout = inspect_programbench_checkout(checkout)
            if observed_checkout != grader_checkout:
                raise RuntimeError(
                    "ProgramBench grader checkout changed during grading"
                )
            revalidated_argv = official_programbench_grader_argv(
                run_directory=snapshot_run,
                output=eval_dir,
                python_executable=str(observed_runtime["python_executable"]),
                workers=args.max_workers,
            )
            if revalidated_argv != argv:
                raise RuntimeError("ProgramBench grader command changed during grading")
            return {"runtime": observed_runtime, "checkout": observed_checkout}

        verify_grader_assets = verify_programbench_grader_assets
    else:
        from .benchmarks.web import (
            collect_browsecomp_plus_runs,
            inspect_browsecomp_plus_grade_inputs,
            official_browsecomp_plus_grader_argv,
        )

        if args.judge_model is None:
            raise ValueError(
                "BrowseComp-Plus grading requires --judge-model as a local "
                "immutable model snapshot directory"
            )
        judge_model_argument = Path(args.judge_model).expanduser()
        judge_model = judge_model_argument.resolve()
        if judge_model_argument.is_symlink() or not judge_model.is_dir():
            raise ValueError("--judge-model must be a local non-symlink directory")
        checkout = _required_path(args.checkout, "--checkout")
        ground_truth = _required_path(args.ground_truth, "--ground-truth")
        qrel = _required_path(args.qrel_evidence, "--qrel-evidence")
        input_dir = evaluation / "official_runs"
        for protected, label in (
            (evaluation / "instances", "evaluation instances"),
            (input_dir, "collected BrowseComp-Plus runs"),
            (checkout, "BrowseComp-Plus checkout"),
            (judge_model, "BrowseComp-Plus judge model"),
        ):
            _reject_grade_output_overlap(grade_output, protected, label)
        collect_browsecomp_plus_runs(evaluation, input_dir)
        source_inputs = inspect_browsecomp_plus_grade_inputs(
            input_dir=input_dir,
            ground_truth=ground_truth,
            qrel_evidence=qrel,
        )
        _verify_grade_prompt_binding(generation_manifest, source_inputs)
        _create_private_grade_output(grade_output)
        input_root = grade_output / "inputs" / "browsecomp-plus"
        snapshot_runs = input_root / "runs"
        snapshot_truth = input_root / ("ground_truth" + ground_truth.suffix.casefold())
        snapshot_qrel = input_root / ("qrel_evidence" + qrel.suffix.casefold())
        _snapshot_grade_input(
            input_dir,
            snapshot_runs,
            source_inputs["runs"],
            label="BrowseComp-Plus runs",
            tree=True,
        )
        _snapshot_grade_input(
            ground_truth,
            snapshot_truth,
            source_inputs["ground_truth"],
            label="BrowseComp-Plus ground truth",
        )
        _snapshot_grade_input(
            qrel,
            snapshot_qrel,
            source_inputs["qrel_evidence"],
            label="BrowseComp-Plus qrel evidence",
        )
        inputs = {
            **inspect_browsecomp_plus_grade_inputs(
                input_dir=snapshot_runs,
                ground_truth=snapshot_truth,
                qrel_evidence=snapshot_qrel,
            ),
            "sources": source_inputs,
        }
        _verify_grade_prompt_binding(generation_manifest, inputs)
        eval_dir = _grade_eval_directory(grade_output, args.eval_dir)
        eval_dir.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        argv = official_browsecomp_plus_grader_argv(
            checkout=checkout,
            input_dir=snapshot_runs,
            ground_truth=snapshot_truth,
            eval_dir=eval_dir,
            qrel_evidence=snapshot_qrel,
            python_executable=str(isolated_runtime["python_executable"]),
            model=str(judge_model),
            tensor_parallel_size=args.tensor_parallel_size,
        )
        grader = {
            "project": "BrowseComp-Plus",
            "revision": "046949032b0328319cc9a02663a759ec601d9402",
            "grader_script": immutable_file_identity(
                checkout / "scripts_evaluation" / "evaluate_run.py",
                label="BrowseComp-Plus grader script",
            ),
            "dependency_lock": immutable_file_identity(
                checkout / "uv.lock", label="BrowseComp-Plus dependency lock"
            ),
            "judge_model": immutable_tree_identity(
                judge_model, label="BrowseComp-Plus judge model"
            ),
        }

        def verify_browsecomp_plus_grader_assets() -> Mapping[str, Any]:
            observed_runtime = observed_grader_runtime("BrowseComp-Plus")
            revalidated_argv = official_browsecomp_plus_grader_argv(
                checkout=checkout,
                input_dir=snapshot_runs,
                ground_truth=snapshot_truth,
                eval_dir=eval_dir,
                qrel_evidence=snapshot_qrel,
                python_executable=str(observed_runtime["python_executable"]),
                model=str(judge_model),
                tensor_parallel_size=args.tensor_parallel_size,
            )
            if revalidated_argv != argv:
                raise RuntimeError(
                    "BrowseComp-Plus grader command changed during grading"
                )
            return {
                **_verify_browsecomp_plus_grader_assets(grader),
                "runtime": observed_runtime,
            }

        verify_grader_assets = verify_browsecomp_plus_grader_assets
    manifest_value = {
        "schema": "mini-agent-grade-v1",
        "benchmark": args.benchmark,
        "harness": harness_identity(),
        "evaluation_manifest": immutable_file_identity(
            generation_manifest_path, label="evaluation manifest"
        ),
        "evaluation_fingerprint": generation_manifest.get("fingerprint"),
        "inputs": inputs,
        "grader": grader,
        "runtime": runtime,
        "argv": list(argv),
    }
    manifest_encoded = canonical_bytes(manifest_value)
    manifest = {
        **manifest_value,
        "fingerprint": hashlib.sha256(manifest_encoded).hexdigest(),
    }
    atomic_json(grade_output / "manifest.json", manifest)
    manifest_identity = immutable_file_identity(
        grade_output / "manifest.json", label="grade manifest"
    )
    stdout_path = grade_output / "stdout.log"
    stderr_path = grade_output / "stderr.log"
    if verify_grader_assets is not None:
        verify_grader_assets()
    with (
        stdout_path.open("xb") as stdout,
        stderr_path.open("xb") as stderr,
    ):
        stdout_path.chmod(0o600)
        stderr_path.chmod(0o600)
        completed = subprocess.run(
            argv,
            cwd=grade_output,
            check=False,
            stdout=stdout,
            stderr=stderr,
            env=grader_environment,
        )
    verified_grader = (
        verify_grader_assets() if verify_grader_assets is not None else None
    )
    verified_inputs = _verify_grade_snapshot_identities(inputs)
    observed_manifest_identity = immutable_file_identity(
        grade_output / "manifest.json", label="grade manifest"
    )
    if any(
        observed_manifest_identity.get(field) != manifest_identity.get(field)
        for field in ("size_bytes", "sha256")
    ):
        raise RuntimeError("grade manifest changed during grader execution")
    artifacts = _grade_artifact_inventory(grade_output)
    result = {
        "schema": "mini-agent-grade-result-v1",
        "benchmark": args.benchmark,
        "returncode": int(completed.returncode),
        "grade_manifest": observed_manifest_identity,
        "stdout": immutable_file_identity(stdout_path, label="grader stdout"),
        "stderr": immutable_file_identity(stderr_path, label="grader stderr"),
        "verified_input_snapshots": verified_inputs,
        "artifacts": artifacts,
    }
    if verified_grader is not None:
        result["verified_grader_assets"] = verified_grader
    atomic_json(grade_output / "result.json", result)
    result_bytes = (grade_output / "result.json").read_bytes()
    atomic_json(
        grade_output / "completed.json",
        {
            "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
            "returncode": int(completed.returncode),
        },
    )
    print(json.dumps({"output": str(grade_output), **result}, indent=2, sort_keys=True))
    return int(completed.returncode)


_GRADER_RUNTIME_PROBE_MAX_BYTES = 64 * 1024
_GRADER_RUNTIME_PROBE_TIMEOUT_SECONDS = 30.0
_GRADER_RUNTIME_PROBE_SOURCE = (
    resources.files("mini_agent").joinpath("_grader_probe.py").read_text("utf-8")
)


def _current_python_executable(python_executable: str) -> Path:
    if (
        not isinstance(python_executable, str)
        or not python_executable
        or "\x00" in python_executable
    ):
        raise ValueError("--python-executable is invalid")
    executable = shutil.which(python_executable)
    try:
        matches_current = executable is not None and os.path.samefile(
            executable, sys.executable
        )
    except OSError:
        matches_current = False
    if not matches_current:
        raise ValueError(
            "--python-executable must be the current mini-agent Python so installed "
            "grader dependencies can be verified"
        )
    assert executable is not None
    # A virtual-environment interpreter is normally a symlink to the base
    # executable.  Its lexical path selects the venv and is therefore part of
    # the runtime identity; resolving it would silently discard the venv.
    return Path(executable).absolute()


def _read_isolated_probe(path: Path, *, label: str) -> Mapping[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(f"{label} did not produce a private identity") from exc
    try:
        status = os.fstat(descriptor)
        if (
            not stat.S_ISREG(status.st_mode)
            or status.st_nlink != 1
            or status.st_uid != os.geteuid()
            or stat.S_IMODE(status.st_mode) & 0o077
            or status.st_size < 1
            or status.st_size > _GRADER_RUNTIME_PROBE_MAX_BYTES
        ):
            raise RuntimeError(f"{label} produced an invalid identity")
        chunks: list[bytes] = []
        remaining = _GRADER_RUNTIME_PROBE_MAX_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 8192))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        encoded = b"".join(chunks)
    finally:
        os.close(descriptor)
    if len(encoded) != status.st_size:
        raise RuntimeError(f"{label} identity changed while it was read")
    try:
        value = strict_json_loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"{label} produced an invalid identity") from exc
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} produced an invalid identity")
    return value


def _grader_runtime_identity(
    python_executable: str,
    benchmark: str,
    *,
    grader_environment: Mapping[str, str],
    timeout_seconds: float = _GRADER_RUNTIME_PROBE_TIMEOUT_SECONDS,
) -> Mapping[str, Any]:
    """Resolve grader dependencies inside the exact isolated grader runtime."""

    executable = _current_python_executable(python_executable)
    if benchmark == "swebench":
        required = {"swebench": "4.1.0"}
    elif benchmark == "programbench":
        required = {"programbench": "1.2.4"}
    elif benchmark == "browsecomp-plus":
        required = {"numpy": "1.26.4", "tqdm": "4.67.1", "vllm": "0.9.0.1"}
    else:
        raise ValueError(f"unsupported official grader benchmark: {benchmark}")
    if not isinstance(grader_environment, Mapping) or not all(
        isinstance(name, str)
        and name
        and "\x00" not in name
        and isinstance(value, str)
        and "\x00" not in value
        for name, value in grader_environment.items()
    ):
        raise ValueError("official grader environment must contain strings")
    timeout_seconds = _require_finite_number(
        timeout_seconds, "official grader runtime probe timeout", exclusive_minimum=0
    )

    with tempfile.TemporaryDirectory(prefix="mini-agent-grader-runtime-") as temporary:
        probe_root = Path(temporary)
        probe_root.chmod(0o700)
        output = probe_root / "identity.json"
        try:
            completed = subprocess.run(
                (
                    str(executable),
                    "-I",
                    "-c",
                    _GRADER_RUNTIME_PROBE_SOURCE,
                    str(output),
                    canonical_bytes(required).decode("utf-8"),
                ),
                cwd=probe_root,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=dict(grader_environment),
                timeout=float(timeout_seconds),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError("isolated official grader runtime probe failed") from exc
        if completed.returncode != 0:
            raise RuntimeError("isolated official grader runtime probe failed")
        value = _read_isolated_probe(
            output, label="isolated official grader runtime probe"
        )

    if value.get("schema") != "mini-agent-isolated-grader-runtime-v1":
        raise RuntimeError("isolated official grader runtime identity is invalid")
    if value.get("ok") is not True:
        name = value.get("name")
        observed = value.get("observed")
        if (
            value.get("error") in {"missing", "version"}
            and isinstance(name, str)
            and name in required
            and value.get("expected") == required[name]
        ):
            found = value.get("error") == "version" and isinstance(observed, str)
            suffix = f", found {observed}" if found else ""
            raise ValueError(
                f"official grader requires {name}=={required[name]}{suffix}"
            )
        raise RuntimeError("isolated official grader runtime identity is invalid")
    expected_keys = {
        "schema",
        "ok",
        "python_executable",
        "python_version",
        "python_implementation",
        "python_prefix",
        "python_base_prefix",
        "packages",
        "modules",
    }
    if set(value) != expected_keys:
        raise RuntimeError("isolated official grader runtime identity is invalid")
    observed_executable = value.get("python_executable")
    if (
        not isinstance(observed_executable, str)
        or Path(observed_executable) != executable
    ):
        raise RuntimeError("isolated official grader used a different Python")
    packages = value.get("packages")
    modules = value.get("modules")
    if not isinstance(packages, Mapping) or dict(packages) != required:
        raise RuntimeError("isolated official grader package identity is invalid")
    if not isinstance(modules, Mapping) or set(modules) != set(required):
        raise RuntimeError("isolated official grader module identity is invalid")
    normalized_modules: dict[str, Mapping[str, str]] = {}
    for name in sorted(required):
        module = modules[name]
        if not isinstance(module, Mapping) or set(module) != {
            "origin",
            "package_root",
        }:
            raise RuntimeError("isolated official grader module identity is invalid")
        origin = module.get("origin")
        package_root = module.get("package_root")
        if not isinstance(origin, str) or not isinstance(package_root, str):
            raise RuntimeError("isolated official grader module identity is invalid")
        origin_path = Path(origin)
        root_path = Path(package_root)
        try:
            valid_paths = (
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
            valid_paths = False
        if not valid_paths:
            raise RuntimeError("isolated official grader module identity is invalid")
        normalized_modules[name] = {
            "origin": str(origin_path),
            "package_root": str(root_path),
        }
    python_version = value.get("python_version")
    implementation = value.get("python_implementation")
    python_prefix = value.get("python_prefix")
    python_base_prefix = value.get("python_base_prefix")
    if (
        not isinstance(python_version, str)
        or not python_version
        or not isinstance(implementation, str)
        or not implementation
        or not isinstance(python_prefix, str)
        or Path(python_prefix) != Path(sys.prefix).absolute()
        or not isinstance(python_base_prefix, str)
        or Path(python_base_prefix) != Path(sys.base_prefix).absolute()
    ):
        raise RuntimeError("isolated official grader Python identity is invalid")
    return {
        "isolation": "python-I",
        "python_executable": str(executable),
        "python_version": python_version,
        "python_implementation": implementation,
        "python_prefix": python_prefix,
        "python_base_prefix": python_base_prefix,
        "packages": dict(packages),
        "modules": normalized_modules,
    }


def _isolated_module_root(runtime: Mapping[str, Any], name: str) -> Path:
    modules = runtime.get("modules")
    module = modules.get(name) if isinstance(modules, Mapping) else None
    root = module.get("package_root") if isinstance(module, Mapping) else None
    if not isinstance(root, str):
        raise RuntimeError(f"isolated grader did not bind the {name} package root")
    return Path(root)


def _official_grader_environment(benchmark: str, grade_output: Path) -> dict[str, str]:
    """Build the narrow host environment exposed to an official grader."""

    allowed = {
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "PATH",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TZ",
    }
    if benchmark in {"swebench", "programbench"}:
        # The upstream harness uses Docker's normal client environment. These
        # values are an intentional authority grant, unlike solver credentials.
        allowed.update({"DOCKER_CERT_PATH", "DOCKER_HOST", "DOCKER_TLS_VERIFY"})
        if benchmark == "programbench":
            # The official evaluator downloads per-branch test blobs from
            # HuggingFace on demand; HOME below keeps that cache private.
            allowed.update({"HF_ENDPOINT", "HF_TOKEN", "HTTPS_PROXY", "NO_PROXY"})
    elif benchmark == "browsecomp-plus":
        # Device selection is useful on shared GPU nodes and does not grant a
        # grader access to provider, cloud, browser, or computer-use services.
        allowed.update(
            {"CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"}
        )
    else:  # pragma: no cover - argparse and _grade restrict this boundary
        raise ValueError(f"unsupported official grader benchmark: {benchmark}")
    environment = {
        name: value
        for name in sorted(allowed)
        if (value := os.environ.get(name)) is not None
    }
    environment.update(
        {
            "HOME": str(grade_output),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    if benchmark in {"swebench", "programbench"} and not environment.get("DOCKER_HOST"):
        # An explicit endpoint prevents Docker SDK context lookup from consulting
        # a different HOME/config than the isolated upstream grader process.
        environment["DOCKER_HOST"] = "unix:///var/run/docker.sock"
    return environment


def _verify_evaluation_manifest(manifest: Mapping[str, Any], benchmark: str) -> None:
    if manifest.get("schema") != "mini-agent-eval-v2":
        raise ValueError("evaluation manifest schema is not supported")
    if manifest.get("benchmark") != benchmark:
        raise ValueError("evaluation manifest benchmark does not match --benchmark")
    fingerprint = manifest.get("fingerprint")
    if not _is_sha256(fingerprint):
        raise ValueError("evaluation manifest fingerprint is malformed")
    unsigned = dict(manifest)
    del unsigned["fingerprint"]
    if hashlib.sha256(canonical_bytes(unsigned)).hexdigest() != fingerprint:
        raise ValueError("evaluation manifest fingerprint does not match its content")
    raw_tasks = manifest.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("evaluation manifest has no task identity list")
    seen: set[str] = set()
    for task in raw_tasks:
        if (
            not isinstance(task, Mapping)
            or not isinstance(task.get("id"), str)
            or not task["id"]
            or not _is_sha256(task.get("prompt_sha256"))
            or not _is_sha256(task.get("data_sha256"))
        ):
            raise ValueError("evaluation manifest task identity is malformed")
        if task["id"] in seen:
            raise ValueError(f"duplicate evaluation manifest task {task['id']!r}")
        seen.add(task["id"])


def _is_sha256(value: Any) -> bool:
    """Return whether ``value`` is a lowercase hexadecimal SHA-256 digest."""

    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _grade_output_path(evaluation: Path, output: Path | None, benchmark: str) -> Path:
    candidate = (
        output.expanduser() if output is not None else evaluation / "grades" / benchmark
    )
    if candidate.is_symlink():
        raise ValueError("grade output must not be a symlink")
    resolved = candidate.resolve()
    if output is None and not resolved.is_relative_to(evaluation):
        raise ValueError("default grade output escapes the evaluation directory")
    return resolved


def _create_private_grade_output(output: Path) -> None:
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"grade output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    output.chmod(0o700)


def _reject_grade_output_overlap(output: Path, protected: Path, label: str) -> None:
    resolved = protected.resolve()
    if (
        output == resolved
        or output.is_relative_to(resolved)
        or resolved.is_relative_to(output)
    ):
        raise ValueError(f"grade output overlaps {label}: {resolved}")


def _grade_eval_directory(output: Path, value: Path | None) -> Path:
    expanded = (Path("official_evals") if value is None else value).expanduser()
    candidate = expanded if expanded.is_absolute() else output / expanded
    if candidate.is_symlink():
        raise ValueError("--eval-dir must not be a symlink")
    resolved = candidate.resolve()
    if (
        resolved == output
        or not resolved.is_relative_to(output)
        or resolved.is_relative_to(output / "inputs")
    ):
        raise ValueError("--eval-dir must be a child of the private grade output")
    if resolved.exists() and (not resolved.is_dir() or any(resolved.iterdir())):
        raise ValueError(f"--eval-dir is not empty: {resolved}")
    return resolved


def _snapshot_grade_input(
    source: Path,
    destination: Path,
    expected: Mapping[str, Any],
    *,
    label: str,
    tree: bool = False,
) -> Mapping[str, Any]:
    """Copy one grader input into the private grade output and re-hash it."""

    if destination.exists() or destination.is_symlink():
        raise ValueError(f"{label} snapshot already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if tree:
        shutil.copytree(source, destination, copy_function=shutil.copyfile)
        for directory, _, files in os.walk(destination):
            Path(directory).chmod(0o700)
            for name in files:
                (Path(directory) / name).chmod(0o600)
    else:
        shutil.copyfile(source, destination)
        destination.chmod(0o600)
    identity = immutable_tree_identity if tree else immutable_file_identity
    observed = identity(destination, label=f"{label} snapshot")
    fields = _TREE_IDENTITY_FIELDS if tree else _FILE_IDENTITY_FIELDS
    if any(observed.get(name) != expected.get(name) for name in fields):
        raise RuntimeError(f"{label} changed while it was being snapshotted")
    return observed


def _reverify_identity(
    expected: Any,
    *,
    tree: bool,
    label: str,
    malformed: str,
    changed: str,
    with_path: bool = False,
) -> Mapping[str, Any]:
    """Re-hash a recorded identity and fail closed when it no longer matches."""

    if not isinstance(expected, Mapping) or not isinstance(expected.get("path"), str):
        raise ValueError(malformed)
    identity = immutable_tree_identity if tree else immutable_file_identity
    observed = identity(Path(expected["path"]), label=label)
    fields = _TREE_IDENTITY_FIELDS if tree else _FILE_IDENTITY_FIELDS
    if with_path:
        fields = ("path", *fields)
    if any(observed.get(field) != expected.get(field) for field in fields):
        raise RuntimeError(changed)
    return observed


def _verify_grade_snapshot_identities(inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    observed: dict[str, Any] = {}
    for name, tree in (
        ("predictions", False),
        ("dataset", False),
        ("ground_truth", False),
        ("qrel_evidence", False),
        ("runs", True),
    ):
        if inputs.get(name) is None:
            continue
        observed[name] = _reverify_identity(
            inputs[name],
            tree=tree,
            label=f"grade input snapshot {name}",
            malformed=f"grade input identity {name!r} is malformed",
            changed=f"grade input snapshot {name!r} changed during grading",
        )
    return observed


def _verify_browsecomp_plus_grader_assets(
    grader: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Re-hash every local grader asset recorded in the grade manifest."""

    return {
        name: _reverify_identity(
            grader.get(name),
            tree=tree,
            label=f"BrowseComp-Plus {name.replace('_', ' ')}",
            malformed=f"BrowseComp-Plus grader identity {name!r} is malformed",
            changed=f"BrowseComp-Plus grader asset {name!r} changed during grading",
            with_path=True,
        )
        for name, tree in (
            ("grader_script", False),
            ("dependency_lock", False),
            ("judge_model", True),
        )
    }


def _grade_artifact_inventory(output: Path) -> Mapping[str, Any]:
    """Harden and hash grader-created files without following output symlinks."""

    files: list[Mapping[str, Any]] = []
    links: list[Mapping[str, Any]] = []
    evidence = {"manifest.json", "stdout.log", "stderr.log", "result.json"}

    def record_symlink(candidate: Path, relative: Path) -> None:
        if relative.parts[0] == "inputs":
            return
        target = os.fsencode(os.readlink(candidate))
        links.append(
            {
                "path": relative.as_posix(),
                "target_sha256": hashlib.sha256(target).hexdigest(),
            }
        )

    output.chmod(0o700)
    for directory, directories, names in os.walk(output, followlinks=False):
        root = Path(directory)
        root.chmod(0o700)
        for name in list(directories):
            candidate = root / name
            relative = candidate.relative_to(output)
            status = candidate.lstat()
            if stat.S_ISLNK(status.st_mode):
                directories.remove(name)
                record_symlink(candidate, relative)
            elif stat.S_ISDIR(status.st_mode):
                candidate.chmod(0o700)
            else:
                raise ValueError(f"grader created a non-directory entry: {candidate}")
        for name in names:
            candidate = root / name
            relative = candidate.relative_to(output)
            status = candidate.lstat()
            if stat.S_ISLNK(status.st_mode):
                record_symlink(candidate, relative)
                continue
            if not stat.S_ISREG(status.st_mode):
                raise ValueError(f"grader created a non-regular file: {candidate}")
            candidate.chmod(0o600)
            if relative.parts[0] == "inputs" or (
                len(relative.parts) == 1 and relative.name in evidence
            ):
                continue
            identity = immutable_file_identity(candidate, label="grader artifact")
            files.append(
                {
                    "path": relative.as_posix(),
                    "size_bytes": identity["size_bytes"],
                    "sha256": identity["sha256"],
                }
            )
    files.sort(key=lambda value: str(value["path"]))
    links.sort(key=lambda value: str(value["path"]))
    encoded = canonical_bytes({"files": files, "links": links})
    return {
        "file_count": len(files),
        "symlink_count": len(links),
        "size_bytes": sum(int(value["size_bytes"]) for value in files),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "files": files,
        "symlinks": links,
    }


def _verify_grade_prompt_binding(
    manifest: Mapping[str, Any], inputs: Mapping[str, Any]
) -> None:
    raw_tasks = manifest.get("tasks")
    if not isinstance(raw_tasks, list):
        raise ValueError("evaluation manifest has no task identity list")
    expected: dict[str, str] = {}
    for task in raw_tasks:
        if (
            not isinstance(task, Mapping)
            or not isinstance(task.get("id"), str)
            or not isinstance(task.get("prompt_sha256"), str)
        ):
            raise ValueError("evaluation manifest task identity is malformed")
        if task["id"] in expected:
            raise ValueError(f"duplicate evaluation manifest task {task['id']!r}")
        expected[task["id"]] = task["prompt_sha256"]
    observed = inputs.get("task_prompt_sha256", inputs.get("query_prompt_sha256"))
    count = inputs.get("prediction_count", inputs.get("run_count"))
    if (
        not observed
        or not isinstance(count, int)
        or isinstance(count, bool)
        or count != len(expected)
        or not _binds_task_hashes(observed, expected)
    ):
        raise ValueError("official grader inputs do not match evaluation task prompts")
    observed_data = inputs.get("task_data_sha256")
    if observed_data is not None:
        expected_data = {task["id"]: task.get("data_sha256") for task in raw_tasks}
        if not _binds_task_hashes(observed_data, expected_data):
            raise ValueError(
                "official grader task data does not match evaluation task data"
            )


def _binds_task_hashes(observed: Any, expected: Mapping[str, Any]) -> bool:
    """Return whether ``observed`` maps exactly ``expected``'s string hashes."""

    return (
        isinstance(observed, Mapping)
        and set(observed) == set(expected)
        and all(
            isinstance(task_id, str)
            and isinstance(task_hash, str)
            and expected.get(task_id) == task_hash
            for task_id, task_hash in observed.items()
        )
    )
