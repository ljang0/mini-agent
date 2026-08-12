"""Environment doctor checks behind `mini-agent doctor`."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .storage import StorageLayout


async def _doctor(args: argparse.Namespace) -> int:
    reports: dict[str, Any] = {}
    okay = True
    targets = (
        ("storage", "swebench", "web", "computer")
        if args.target == "all"
        else (args.target,)
    )
    if "storage" in targets:
        layout = StorageLayout.resolve(args.home, args.scratch)
        layout.ensure()
        durable_free = layout.free_bytes()
        scratch_free = layout.free_bytes(scratch=True)
        durable_required = int(args.min_durable_free_gib * 1024**3)
        scratch_required = int(args.min_scratch_free_gib * 1024**3)
        storage_ok = (
            durable_free >= durable_required and scratch_free >= scratch_required
        )
        reports["storage"] = {
            "ok": storage_ok,
            "status": "ready" if storage_ok else "insufficient_space",
            "durable": str(layout.root),
            "scratch": str(layout.scratch),
            "durable_free_bytes": durable_free,
            "scratch_free_bytes": scratch_free,
            "durable_required_bytes": durable_required,
            "scratch_required_bytes": scratch_required,
        }
        okay = okay and storage_ok
    if "swebench" in targets:
        if args.runtime == "docker":
            from .environments.swebench import swebench_doctor

            docker = await swebench_doctor(runtime=tuple(args.container_runtime))
            swe_report: Mapping[str, Any] = {
                "runtime": "docker",
                **docker.as_dict(),
            }
            runtime_ok = docker.ok
        else:
            swe_report = await _apptainer_doctor(args)
            runtime_ok = bool(swe_report["ok"])
        reports["swebench"] = swe_report
        okay = okay and runtime_ok
    if "web" in targets:
        if args.web_mode == "live":
            credential = bool(os.environ.get("SERPAPI_API_KEY"))
            reader = await _page_reader_doctor(args.page_reader)
            web_ok = credential and bool(reader["ok"])
            web_report: Mapping[str, Any] = {
                "mode": "browsecomp_live",
                "ok": web_ok,
                "serpapi": {
                    "ok": credential,
                    "credential_env": "SERPAPI_API_KEY",
                    "network_canary_run": False,
                },
                "page_reader": reader,
            }
        else:
            web_report = _fixed_web_doctor(args.index, args.anserini_jar)
            web_ok = bool(web_report["ok"])
        reports["web"] = web_report
        okay = okay and web_ok
    if "computer" in targets:
        computer_report = await _computer_doctor(args)
        reports["computer"] = computer_report
        okay = okay and bool(computer_report["ok"])
    print(json.dumps({"ok": okay, "reports": reports}, indent=2, sort_keys=True))
    return 0 if okay else 1


async def _apptainer_doctor(args: argparse.Namespace) -> Mapping[str, Any]:
    from .environments.base import complete_in_thread
    from .environments.swe import LocalProcessRunner

    executable = shutil.which(args.apptainer_executable)
    if executable is None:
        return {
            "runtime": "apptainer",
            "ok": False,
            "detail": f"{args.apptainer_executable!r} is not on PATH",
        }
    layout = StorageLayout.resolve(args.home, args.scratch)
    layout.ensure()
    root = Path(tempfile.mkdtemp(prefix="mini-agent-doctor-", dir=layout.scratch))
    overlay = root / "overlay.img"
    runner = LocalProcessRunner()
    try:
        version = await runner.run((executable, "version"), timeout_seconds=30)
        created = await runner.run(
            (
                executable,
                "overlay",
                "create",
                "--fakeroot",
                "--size",
                "64",
                str(overlay),
            ),
            timeout_seconds=120,
        )
        ok = (
            version.returncode == 0
            and not version.timed_out
            and created.returncode == 0
            and not created.timed_out
            and overlay.is_file()
        )
        return {
            "runtime": "apptainer",
            "ok": ok,
            "executable": executable,
            "version": version.text().strip(),
            "fakeroot_overlay": "ready" if ok else created.text().strip(),
            "evaluation_overlay_size_mib": args.overlay_size_mib,
        }
    finally:
        if root.exists():
            await complete_in_thread(shutil.rmtree, root)


async def _page_reader_doctor(reader: str) -> Mapping[str, Any]:
    if reader == "http":
        return {"name": "http", "ok": True, "network_canary_run": False}
    try:
        from playwright.async_api import async_playwright  # type: ignore[import-not-found]

        playwright = await async_playwright().start()
        try:
            executable = Path(playwright.chromium.executable_path)
            return {
                "name": "playwright",
                "ok": executable.is_file(),
                "chromium_executable": str(executable),
            }
        finally:
            await playwright.stop()
    except Exception as exc:
        return {
            "name": "playwright",
            "ok": False,
            "detail": f"{type(exc).__name__}: {exc}",
        }


def _fixed_web_doctor(
    index: Path | None, anserini_jar: Path | None
) -> Mapping[str, Any]:
    import importlib.util

    from .environments.web import (
        HUGGINGFACE_HUB_VERSION,
        PYJNIUS_VERSION,
        TOKENIZERS_VERSION,
    )

    packages: dict[str, bool] = {}
    package_versions: dict[str, Mapping[str, Any]] = {}
    for name, module, required in (
        ("huggingface-hub", "huggingface_hub", HUGGINGFACE_HUB_VERSION),
        ("pyjnius", "jnius", PYJNIUS_VERSION),
        ("tokenizers", "tokenizers", TOKENIZERS_VERSION),
    ):
        try:
            present = importlib.util.find_spec(module) is not None
            observed = importlib.metadata.version(name) if present else None
        except (
            ImportError,
            AttributeError,
            ValueError,
            importlib.metadata.PackageNotFoundError,
        ):
            observed = None
        package_ok = observed == required
        packages[name] = package_ok
        package_versions[name] = {
            "ok": package_ok,
            "observed": observed,
            "required": required,
        }
    jar_report: dict[str, Any]
    if anserini_jar is None:
        jar_report = {"ok": False, "detail": "--anserini-jar not provided"}
    else:
        try:
            from .environments.web import (
                ANSERINI_VERSION,
                validate_anserini_jar,
            )

            resolved_jar, jar_sha256 = validate_anserini_jar(anserini_jar)
            jar_report = {
                "ok": True,
                "path": str(resolved_jar),
                "sha256": jar_sha256,
                "version": ANSERINI_VERSION,
            }
        except Exception as exc:
            jar_report = {
                "ok": False,
                "detail": f"{type(exc).__name__}: {exc}",
            }
    java = _java_doctor(required_major=21)
    index_report: dict[str, Any]
    if index is None:
        index_report = {"ok": False, "detail": "--index not provided"}
    else:
        try:
            from .environments.web import directory_sha256

            resolved = index.expanduser().resolve()
            index_report = {
                "ok": True,
                "path": str(resolved),
                "sha256": directory_sha256(index),
            }
        except Exception as exc:
            index_report = {
                "ok": False,
                "detail": f"{type(exc).__name__}: {exc}",
            }
    ok = (
        all(packages.values())
        and bool(java["ok"])
        and bool(index_report["ok"])
        and bool(jar_report["ok"])
    )
    return {
        "mode": "browsecomp_plus_fixed",
        "ok": ok,
        "packages": packages,
        "package_versions": package_versions,
        "java": java,
        "index": index_report,
        "anserini_jar": jar_report,
        "tokenizer_load_canary_run": False,
    }


def _java_doctor(*, required_major: int) -> Mapping[str, Any]:
    executable = shutil.which("java")
    if executable is None:
        return {
            "ok": False,
            "required_major": required_major,
            "detail": "java is not on PATH",
        }
    try:
        result = subprocess.run(
            (executable, "-version"),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "ok": False,
            "required_major": required_major,
            "executable": executable,
            "detail": f"{type(exc).__name__}: {exc}",
        }
    version = (result.stderr or result.stdout).strip().splitlines()
    first_line = version[0] if version else ""
    match = re.search(r'version\s+"(?:1\.)?(\d+)', first_line)
    major = int(match.group(1)) if match is not None else None
    return {
        "ok": result.returncode == 0 and major is not None and major >= required_major,
        "required_major": required_major,
        "executable": executable,
        "major": major,
        "version": first_line,
    }


async def _computer_doctor(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.checkout is None:
        return {"ok": False, "detail": "--checkout not provided"}
    try:
        if args.osworld_version:
            from .benchmarks.osworld import UpstreamDesktopFactory

            if args.runtime == "apptainer" and args.osworld_apptainer_image is None:
                raise ValueError(
                    "OSWorld --runtime apptainer requires --osworld-apptainer-image"
                )
            factory = UpstreamDesktopFactory(
                args.checkout,
                version=args.osworld_version,
                provider_name="docker",
                path_to_vm=args.path_to_vm,
                apptainer_image=(
                    args.osworld_apptainer_image
                    if args.runtime == "apptainer"
                    else None
                ),
                apptainer_executable=args.apptainer_executable,
            )
            runtime: Mapping[str, Any]
            if args.runtime == "apptainer":
                from .environments.osworld_apptainer import (
                    osworld_apptainer_preflight,
                )

                assert args.osworld_apptainer_image is not None
                runtime = osworld_apptainer_preflight(
                    args.osworld_apptainer_image,
                    executable=args.apptainer_executable,
                )
            else:
                from .environments.swebench import swebench_doctor

                docker = await swebench_doctor(runtime=tuple(args.container_runtime))
                runtime = {
                    "name": "docker",
                    **docker.as_dict(),
                    "daemon_canary_run": True,
                    "machine_launch_canary_run": False,
                }
            return {
                "ok": bool(runtime["ok"]),
                "benchmark": f"osworld-{args.osworld_version}",
                "checkout": dict(factory.checkout.as_dict()),
                "environment_factory": dict(factory.provenance()),
                "runtime": dict(runtime),
                "machine_launch_canary_run": False,
            }
        if args.benchmark_path is None:
            return {
                "ok": False,
                "benchmark": "cua-speed-run",
                "detail": "--benchmark-path not provided",
            }
        from .benchmarks.cua_speedrun import preflight_cua_speedrun

        if args.qemu_cache is not None:
            expanded_cache = args.qemu_cache.expanduser()
            if expanded_cache.is_symlink():
                raise ValueError("--qemu-cache must not be a symlink")
            os.environ["GYM_ANYTHING_QEMU_CACHE"] = str(expanded_cache.resolve())
        report = dict(
            preflight_cua_speedrun(
                args.checkout,
                args.benchmark_path,
                backend_name=args.backend,
            )
        )
        if args.qemu_cache is not None:
            from .benchmarks.cua_speedrun import _cua_machine_images

            report["machine_images"] = _cua_machine_images()
        return {
            "ok": True,
            **report,
            "machine_launch_canary_run": False,
        }
    except Exception as exc:
        return {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}


