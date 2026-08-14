"""Apptainer runtime: a private fakeroot overlay over a materialized SIF."""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from .._hash import stable_file_sha256
from .._lifecycle import complete_in_thread, raise_lifecycle_errors
from ..types import _require_bool, _require_no_symlink, _require_positive_int
from .base import (
    DEFAULT_MAX_OUTPUT_BYTES,
    ProcessResult,
    ProcessRunner,
    atomic_write,
    failed,
    positive_int,
    positive_number,
    require_ok,
    require_ref,
    require_runner,
    require_staging_name,
    require_workdir,
    resolve_runner,
)


DEFAULT_OVERLAY_ROOT_PREFIX = "mini-agent-apptainer-"
_OWNED_APPTAINER = object()


async def apptainer_image_identity(image: str) -> str:
    """Re-hash a local SIF so execution is bound to exact image bytes."""

    path = _require_no_symlink(Path(image), "Apptainer image path")
    if not path.is_file():
        raise ValueError("Apptainer execution requires a materialized local image")
    digest = await complete_in_thread(stable_file_sha256, path, label="Apptainer image")
    return f"sha256:{digest}"


async def materialize_apptainer_image(
    image: str,
    *,
    executable: str,
    runner: ProcessRunner,
    cache: Path | None,
    timeout_seconds: float,
    max_output_bytes: int,
) -> str:
    """Return a local SIF path, pulling into a lock-guarded cache if needed."""

    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - Apptainer is Linux-only
        raise RuntimeError("Apptainer image caching requires POSIX file locks") from exc
    require_ref(image, "Apptainer image reference must be non-empty")
    require_ref(executable, "Apptainer executable must be non-empty")
    require_runner(runner)
    resolved_timeout = positive_number(timeout_seconds, "cache timeout_seconds")
    resolved_output = positive_int(max_output_bytes, "cache max_output_bytes")
    local = _require_no_symlink(Path(image).expanduser(), "Apptainer image path")
    if local.is_file():
        return str(local.resolve())
    if cache is not None and cache.expanduser().is_symlink():
        raise ValueError("Apptainer image cache root cannot be a symlink")
    cache_root = (
        cache.expanduser().resolve()
        if cache is not None
        else Path.home() / ".cache" / "mini-agent" / "apptainer-images"
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    if cache_root.is_symlink() or not cache_root.is_dir():
        raise ValueError("Apptainer image cache must be a directory")
    key = hashlib.sha256(image.encode("utf-8")).hexdigest()
    destination = cache_root / f"{key}.sif"
    lock_path = cache_root / f"{key}.lock"
    if destination.is_symlink() or lock_path.is_symlink():
        raise ValueError("Apptainer cache entries cannot be symlinks")
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    lock_stream = os.fdopen(descriptor, "a+b")
    temporary = cache_root / f".{key}.{uuid.uuid4().hex}.sif"
    acquired = False
    lock_deadline = time.monotonic() + resolved_timeout
    try:
        while True:
            try:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                remaining = lock_deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        "timed out waiting for Apptainer image cache lock"
                    )
                await asyncio.sleep(min(0.05, remaining))
        if destination.is_symlink():
            raise ValueError("Apptainer cache entries cannot be symlinks")
        if not destination.is_file() or destination.stat().st_size == 0:
            pulled = await runner.run(
                (executable, "pull", "--force", str(temporary), image),
                timeout_seconds=resolved_timeout,
                max_output_bytes=resolved_output,
            )
            if failed(pulled) or not temporary.is_file():
                raise RuntimeError(
                    "could not materialize Apptainer image: " + pulled.text()
                )
            temporary.replace(destination)
        return str(destination)
    finally:
        temporary.unlink(missing_ok=True)
        if acquired:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
        lock_stream.close()


def apptainer_exec_argv(
    *,
    executable: str,
    overlay: Path,
    image: str,
    workdir: str,
    argv: Sequence[str],
    env: Mapping[str, str] | None = None,
    binds: Sequence[tuple[Path, str]] = (),
    writable_binds: Sequence[tuple[Path, str]] = (),
    network_disabled: bool = False,
) -> tuple[str, ...]:
    """Build one `apptainer exec` argv for a private fakeroot overlay."""

    command = [
        executable,
        "--silent",
        "exec",
        "--cleanenv",
        "--containall",
        "--fakeroot",
        "--overlay",
        str(overlay),
        "--pwd",
        workdir,
    ]
    if network_disabled:
        # A private empty network namespace: no interface, no DNS, no egress.
        command.extend(("--net", "--network", "none"))
    if env:
        command.extend(
            ("--env", ",".join(f"{key}={value}" for key, value in env.items()))
        )
    for source, target in binds:
        command.extend(("--bind", f"{source}:{target}:ro"))
    for source, target in writable_binds:
        command.extend(("--bind", f"{source}:{target}:rw"))
    command.append(image)
    command.extend(argv)
    return tuple(command)


class ApptainerRuntime:
    """One private fakeroot ext3 overlay over an immutable local SIF."""

    # `--containall` gives every exec a fresh tmpfs /tmp, so anything the
    # harness must read back afterwards has to be staged on the overlay.
    archive_staging_dir: str = "/"

    def __init__(
        self,
        *,
        executable: str,
        image: str,
        image_source: str,
        image_identity: str,
        overlay: Path,
        owned_root: Path,
        workdir: str,
        runner: ProcessRunner,
        exec_env: Mapping[str, str] | None = None,
        staging_dir: str = "/tmp",
        root_prefix: str = DEFAULT_OVERLAY_ROOT_PREFIX,
        network_disabled: bool = False,
        shared_binds: Mapping[str, Path] | None = None,
        _ownership_token: object | None = None,
    ) -> None:
        if _ownership_token is not _OWNED_APPTAINER:
            raise ValueError("Apptainer runtimes are created only by start()")
        resolved_root = owned_root.resolve()
        resolved_overlay = overlay.resolve()
        if (
            owned_root.is_symlink()
            or not resolved_root.is_dir()
            or not resolved_root.name.startswith(root_prefix)
            or resolved_overlay != resolved_root / "overlay.img"
        ):
            raise ValueError("owned Apptainer root has an invalid internal layout")
        self.executable = require_ref(
            executable, "Apptainer executable must be non-empty"
        )
        self.image = require_ref(image, "Apptainer image must be non-empty")
        self.image_source = require_ref(
            image_source, "Apptainer image source must be non-empty"
        )
        self.image_identity = require_ref(
            image_identity, "Apptainer image identity must be non-empty"
        )
        self.overlay = resolved_overlay
        self.owned_root = resolved_root
        self.workdir = require_workdir(workdir)
        self.runner = require_runner(runner)
        self.exec_env = dict(exec_env or {})
        self.network_disabled = _require_bool(network_disabled, "network_disabled")
        self.staging_dir = require_workdir(staging_dir)
        self._binds: dict[str, Path] = {}
        # Writable host paths every exec sees. The only current use is a team's
        # shared Git repository, which must live outside the workdir so it is
        # neither submitted nor destroyed when a workspace is replaced.
        self.shared_binds = self._validate_shared_binds(shared_binds, self.workdir)

    @staticmethod
    def _validate_shared_binds(
        shared_binds: Mapping[str, Path] | None, workdir: str
    ) -> dict[str, Path]:
        resolved: dict[str, Path] = {}
        for target, source in dict(shared_binds or {}).items():
            target = require_workdir(target)
            if target == workdir or target.startswith(workdir.rstrip("/") + "/"):
                raise ValueError("a shared bind must not sit inside the workdir")
            resolved[target] = _require_no_symlink(source, "shared bind").resolve()
        return resolved

    @classmethod
    async def start(
        cls,
        *,
        image: str,
        image_source: str | None = None,
        executable: str = "apptainer",
        runner: ProcessRunner | None = None,
        workdir: str,
        exec_env: Mapping[str, str] | None = None,
        scratch_root: Path | None = None,
        image_cache: Path | None = None,
        root_prefix: str = DEFAULT_OVERLAY_ROOT_PREFIX,
        overlay_size_mib: int = 16 * 1024,
        expected_identity: str | None = None,
        shared_binds: Mapping[str, Path] | None = None,
        materialize: bool = True,
        network_disabled: bool = False,
        timeout_seconds: float = 60.0,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    ) -> "ApptainerRuntime":
        """Materialize the SIF, verify its bytes, and create a private overlay."""

        require_ref(executable, "Apptainer executable must be non-empty")
        require_ref(image, "Apptainer image reference is invalid", no_dash=True)
        resolved_workdir = require_workdir(workdir)
        _require_positive_int(overlay_size_mib, "overlay_size_mib", minimum=1024)
        resolved_timeout = positive_number(timeout_seconds, "timeout_seconds")
        resolved_output = positive_int(max_output_bytes, "max_output_bytes")
        process_runner = resolve_runner(runner)
        root_parent = (
            None if scratch_root is None else scratch_root.expanduser().resolve()
        )
        if root_parent is not None:
            root_parent.mkdir(parents=True, exist_ok=True)
        owned = Path(tempfile.mkdtemp(prefix=root_prefix, dir=root_parent))
        overlay = owned / "overlay.img"
        try:
            if materialize:
                selected = await materialize_apptainer_image(
                    image,
                    executable=executable,
                    runner=process_runner,
                    cache=image_cache,
                    timeout_seconds=max(resolved_timeout, 1800),
                    max_output_bytes=resolved_output,
                )
            else:
                selected = image
            identity = await apptainer_image_identity(selected)
            if expected_identity is not None and identity != expected_identity:
                raise RuntimeError(
                    "Apptainer image bytes do not match the preflight binding"
                )
            created = await process_runner.run(
                (
                    executable,
                    "overlay",
                    "create",
                    "--fakeroot",
                    "--size",
                    str(overlay_size_mib),
                    str(overlay),
                ),
                timeout_seconds=max(resolved_timeout, 300),
                max_output_bytes=resolved_output,
            )
            require_ok(created, "could not create Apptainer overlay")
            return cls(
                executable=executable,
                image=selected,
                image_source=image if image_source is None else image_source,
                image_identity=identity,
                overlay=overlay,
                owned_root=owned,
                workdir=resolved_workdir,
                runner=process_runner,
                exec_env=exec_env,
                root_prefix=root_prefix,
                network_disabled=network_disabled,
                shared_binds=shared_binds,
                _ownership_token=_OWNED_APPTAINER,
            )
        except BaseException as operation_error:
            cleanup_error: BaseException | None = None
            try:
                if owned.exists():
                    await complete_in_thread(shutil.rmtree, owned)
            except BaseException as exc:
                cleanup_error = exc
            raise_lifecycle_errors("Apptainer setup", operation_error, cleanup_error)
            raise AssertionError("unreachable")

    async def exec(
        self,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_seconds: float = 60.0,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    ) -> ProcessResult:
        return await self.runner.run(
            apptainer_exec_argv(
                executable=self.executable,
                overlay=self.overlay,
                image=self.image,
                workdir=self.workdir if cwd is None else require_workdir(cwd),
                argv=argv,
                env=self.exec_env if env is None else env,
                binds=tuple(
                    (source, target) for target, source in sorted(self._binds.items())
                ),
                writable_binds=tuple(
                    (source, target)
                    for target, source in sorted(self.shared_binds.items())
                ),
                network_disabled=self.network_disabled,
            ),
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )

    async def read_file(self, path: str) -> bytes:
        """Copy one file out of the overlay through a private writable bind.

        The bind exists only for this one exec, so an agent command never sees
        a writable host path; only the harness's export path uses it.
        """

        require_ref(path, "container file path must be non-empty", no_dash=True)
        export = Path(tempfile.mkdtemp(prefix="export-", dir=self.owned_root))
        target = "/mini-agent-export"
        operation_error: BaseException | None = None
        content = b""
        try:
            copied = await self.runner.run(
                apptainer_exec_argv(
                    executable=self.executable,
                    overlay=self.overlay,
                    image=self.image,
                    workdir=self.workdir,
                    argv=("cp", path, f"{target}/payload"),
                    env=self.exec_env,
                    writable_binds=((export, target),),
                    network_disabled=self.network_disabled,
                ),
                timeout_seconds=300.0,
                max_output_bytes=DEFAULT_MAX_OUTPUT_BYTES,
            )
            require_ok(copied, "could not copy the overlay file")
            content = await complete_in_thread((export / "payload").read_bytes)
        except BaseException as exc:
            operation_error = exc
        cleanup_error: BaseException | None = None
        try:
            await complete_in_thread(shutil.rmtree, export)
        except BaseException as exc:
            cleanup_error = exc
        raise_lifecycle_errors("overlay file read", operation_error, cleanup_error)
        return content

    async def write_file(self, name: str, data: bytes) -> str:
        """Stage bytes on the host and bind them read-only into every exec."""

        resolved = require_staging_name(name)
        source = self.owned_root / resolved
        target = f"{self.staging_dir.rstrip('/')}/{resolved}"
        await complete_in_thread(atomic_write, source, data)
        self._binds[target] = source
        return target

    async def remove_file(self, path: str) -> None:
        source = self._binds.pop(path, None)
        if source is not None:
            source.unlink(missing_ok=True)

    async def close(self) -> None:
        self._binds.clear()
        if self.owned_root.exists():
            await complete_in_thread(shutil.rmtree, self.owned_root)

    def provenance(self) -> Mapping[str, Any]:
        return {
            "container_runtime": "apptainer",
            "container_image": self.image,
            "container_image_source": self.image_source,
            "container_image_identity": self.image_identity,
            "overlay": "private_fakeroot_ext3",
            "workdir": self.workdir,
            # Named explicitly: a writable shared bind is the one way an
            # otherwise isolated container can reach another agent's bytes.
            "shared_writable_binds": sorted(self.shared_binds),
        }

    def resource_identity(self) -> str:
        return f"apptainer-overlay:{self.overlay}"


__all__ = [
    "DEFAULT_OVERLAY_ROOT_PREFIX",
    "ApptainerRuntime",
    "apptainer_exec_argv",
    "apptainer_image_identity",
    "materialize_apptainer_image",
]
