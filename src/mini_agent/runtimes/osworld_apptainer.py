"""Docker-client compatibility for OSWorld's official image under Apptainer."""

from __future__ import annotations

import hashlib
import os
import shutil
import signal
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence


_RUNNING_CONTAINERS: list[_ApptainerContainer] = []
_RUNNING_CONTAINERS_LOCK = threading.Lock()


class OSWorldApptainerDockerClient:
    """Implement only the Docker SDK surface used by OSWorld's Docker provider."""

    def __init__(
        self,
        *,
        apptainer_image: Path,
        vm_image: Path,
        work_root: Path,
        executable: str = "apptainer",
    ) -> None:
        self.apptainer_image = _regular_file(apptainer_image, "Apptainer image")
        self.vm_image = _regular_file(vm_image, "OSWorld VM image")
        expanded_work = work_root.expanduser()
        if expanded_work.is_symlink():
            raise ValueError("OSWorld Apptainer work root must not be a symlink")
        self.work_root = expanded_work.resolve()
        if self.work_root.exists() and not self.work_root.is_dir():
            raise ValueError("OSWorld Apptainer work root must be a directory")
        resolved_executable = shutil.which(executable)
        if resolved_executable is None:
            raise ValueError(f"Apptainer executable not found: {executable}")
        self.executable = resolved_executable
        self.containers = _ContainerCollection(self)


def osworld_apptainer_preflight(
    apptainer_image: Path,
    *,
    executable: str = "apptainer",
) -> Mapping[str, Any]:
    """Check the local launcher without booting a guest or provisioning assets."""

    image = _regular_file(apptainer_image, "OSWorld Apptainer image")
    resolved_executable = shutil.which(executable)
    if resolved_executable is None:
        raise ValueError(f"Apptainer executable not found: {executable}")
    kvm = Path("/dev/kvm")
    if not kvm.exists() or not os.access(kvm, os.R_OK | os.W_OK):
        raise RuntimeError("OSWorld Apptainer runtime requires read/write /dev/kvm")
    result = subprocess.run(
        (
            resolved_executable,
            "exec",
            "--fakeroot",
            "--contain",
            "--cleanenv",
            str(image),
            "qemu-system-x86_64",
            "--version",
        ),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            "OSWorld Apptainer/QEMU preflight failed"
            + (f": {detail[-1000:]}" if detail else "")
        )
    first_line = result.stdout.strip().splitlines()
    return {
        "status": "source_ready",
        "executable": resolved_executable,
        "image": {"path": str(image), "size_bytes": image.stat().st_size},
        "kvm_read_write": True,
        "qemu_version": first_line[0] if first_line else "",
        "machine_launch_canary_run": False,
    }


class _ContainerCollection:
    def __init__(self, client: OSWorldApptainerDockerClient) -> None:
        self.client = client

    def list(self) -> list["_ApptainerContainer"]:
        return _running_containers()

    def run(
        self,
        image: str,
        *,
        environment: Mapping[str, str],
        cap_add: Sequence[str],
        devices: Sequence[str],
        volumes: Mapping[str, Mapping[str, str]],
        ports: Mapping[int, int],
        detach: bool,
    ) -> "_ApptainerContainer":
        """Launch the OSWorld guest under Apptainer instead of a Docker daemon.
        
                Every argument upstream would have passed to Docker is checked against
                what OSWorld actually requires -- the image, the KVM device, the four
                forwarded ports, the read-only VM mount -- and anything unexpected is
                refused rather than silently translated. The compatibility surface is
                narrow on purpose: it stands in for one specific Docker call, not for
                Docker.
                """
        if image != "happysixd/osworld-docker":
            raise ValueError(f"unexpected OSWorld container image: {image}")
        if not detach:
            raise ValueError("OSWorld Apptainer compatibility requires detach=True")
        if tuple(cap_add) != ("NET_ADMIN",):
            raise ValueError("unexpected OSWorld container capabilities")
        if tuple(devices) != ("/dev/kvm",):
            raise ValueError("OSWorld Apptainer compatibility requires /dev/kvm")
        required_environment = {"DISK_SIZE", "RAM_SIZE", "CPU_CORES"}
        if (
            not required_environment.issubset(environment)
            or set(environment).difference(required_environment | {"KVM"})
            or not all(isinstance(value, str) for value in environment.values())
        ):
            raise ValueError("unexpected OSWorld container environment")
        expected_ports = {8006, 5000, 9222, 8080}
        if set(ports) != expected_ports:
            raise ValueError("unexpected OSWorld container port contract")
        host_ports = tuple(ports[port] for port in sorted(expected_ports))
        if (
            any(
                not isinstance(port, int)
                or isinstance(port, bool)
                or not 1024 <= port <= 65535
                for port in host_ports
            )
            or len(host_ports) != len(set(host_ports))
        ):
            raise ValueError("OSWorld host ports must be distinct unprivileged ports")
        if len(volumes) != 1:
            raise ValueError("OSWorld must mount exactly one VM image")
        source, mount = next(iter(volumes.items()))
        if Path(source).resolve() != self.client.vm_image:
            raise ValueError("OSWorld mounted a different VM image")
        if dict(mount) != {"bind": "/System.qcow2", "mode": "ro"}:
            raise ValueError("unexpected OSWorld VM mount contract")

        identifier = uuid.uuid4().hex
        directory = self.client.work_root / identifier
        directory.mkdir(parents=True, exist_ok=False, mode=0o700)
        storage = directory / "storage"
        storage.mkdir()
        try:
            _materialize_uefi_firmware(
                executable=self.client.executable,
                image=self.client.apptainer_image,
                storage=storage,
            )
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        network = directory / "network.sh"
        network.write_text(_network_script(), encoding="utf-8")
        network.chmod(0o500)
        nginx = directory / "web.conf"
        nginx.write_text(_nginx_config(ports[8006]), encoding="utf-8")
        mac = _mac_address(identifier)
        container_environment = {
            "DISK_SIZE": environment.get("DISK_SIZE", "32G"),
            "RAM_SIZE": environment.get("RAM_SIZE", "4G"),
            "CPU_CORES": environment.get("CPU_CORES", "4"),
            "KVM": environment.get("KVM", "Y"),
            "NETWORK": "user",
            "DISPLAY": "disabled",
            "DEBUG": "Y",
            "RAM_CHECK": "Y",
            "MONITOR": "none",
            "SERIAL": "none",
            "OSWORLD_VM_MAC": mac,
            "OSWORLD_SERVER_PORT": str(ports[5000]),
            "OSWORLD_CHROMIUM_PORT": str(ports[9222]),
            "OSWORLD_VLC_PORT": str(ports[8080]),
        }
        argv = [
            self.client.executable,
            "run",
            "--fakeroot",
            "--contain",
            "--cleanenv",
            "--pid",
            "--writable-tmpfs",
            "--bind",
            "/dev/kvm",
            "--bind",
            f"{self.client.vm_image}:/System.qcow2:ro",
            "--bind",
            f"{storage}:/storage",
            "--bind",
            f"{network}:/run/network.sh:ro",
            "--bind",
            f"{nginx}:/etc/nginx/sites-enabled/web.conf:ro",
        ]
        for name, value in sorted(container_environment.items()):
            argv.extend(("--env", f"{name}={value}"))
        argv.append(str(self.client.apptainer_image))
        log_path = directory / "launcher.log"
        log_handle = log_path.open("ab", buffering=0)
        try:
            process = subprocess.Popen(
                argv,
                cwd=directory,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except BaseException:
            log_handle.close()
            shutil.rmtree(directory, ignore_errors=True)
            raise
        container = _ApptainerContainer(
            process=process,
            log_handle=log_handle,
            log_path=log_path,
            directory=directory,
            ports=ports,
        )
        _register_container(container)
        try:
            _await_apptainer_launch(
                process,
                server_port=ports[5000],
                log_path=log_path,
            )
        except BaseException:
            container.stop()
            _unregister_container(container)
            shutil.rmtree(directory, ignore_errors=True)
            raise
        return container


class _ApptainerContainer:
    def __init__(
        self,
        *,
        process: subprocess.Popen[bytes],
        log_handle: BinaryIO,
        log_path: Path,
        directory: Path,
        ports: Mapping[int, int],
    ) -> None:
        self.process = process
        self._log_handle = log_handle
        self.log_path = log_path
        self.directory = directory
        self.attrs = {
            "NetworkSettings": {
                "Ports": {
                    f"{container_port}/tcp": [
                        {"HostPort": str(host_port)}
                    ]
                    for container_port, host_port in ports.items()
                }
            }
        }

    @property
    def is_running(self) -> bool:
        return self.process.poll() is None

    def stop(self) -> None:
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=30)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=10)
        self._close_log()

    def remove(self, *, v: bool = False) -> None:
        if not isinstance(v, bool):
            raise ValueError("Docker remove volume flag must be boolean")
        if self.is_running:
            raise RuntimeError("cannot remove a running OSWorld container")
        self._close_log()
        _remove_private_tree(self.directory)
        _unregister_container(self)

    def _close_log(self) -> None:
        if not self._log_handle.closed:
            self._log_handle.close()


def _register_container(container: _ApptainerContainer) -> None:
    with _RUNNING_CONTAINERS_LOCK:
        _RUNNING_CONTAINERS.append(container)


def _unregister_container(container: _ApptainerContainer) -> None:
    with _RUNNING_CONTAINERS_LOCK:
        try:
            _RUNNING_CONTAINERS.remove(container)
        except ValueError:
            pass


def _running_containers() -> list[_ApptainerContainer]:
    with _RUNNING_CONTAINERS_LOCK:
        running = [
            container for container in _RUNNING_CONTAINERS if container.is_running
        ]
        _RUNNING_CONTAINERS[:] = running
        return list(running)


def _network_script() -> str:
    return """#!/usr/bin/env bash
set -Eeuo pipefail
: "${OSWORLD_VM_MAC:?}"
: "${OSWORLD_SERVER_PORT:?}"
: "${OSWORLD_CHROMIUM_PORT:?}"
: "${OSWORLD_VLC_PORT:?}"
VM_NET_IP="20.20.20.21"
NET_OPTS="-netdev user,id=hostnet0,host=20.20.20.1,net=20.20.20.0/24,dhcpstart=${VM_NET_IP},hostname=QEMU"
NET_OPTS+=",hostfwd=tcp:127.0.0.1:${OSWORLD_SERVER_PORT}-${VM_NET_IP}:5000"
NET_OPTS+=",hostfwd=tcp:127.0.0.1:${OSWORLD_CHROMIUM_PORT}-${VM_NET_IP}:9222"
NET_OPTS+=",hostfwd=tcp:127.0.0.1:${OSWORLD_VLC_PORT}-${VM_NET_IP}:8080"
NET_OPTS+=" -device virtio-net-pci,romfile=,netdev=hostnet0,mac=${OSWORLD_VM_MAC},id=net0"
html "Initialized user-mode network successfully..."
return 0
"""


def _nginx_config(port: int) -> str:
    return f"""server {{
    listen 127.0.0.1:{port} default_server;
    server_tokens off;
    access_log off;
    location / {{ root /run/shm; try_files /index.html =404; }}
}}
"""


def _regular_file(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    resolved = expanded.resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} must be a regular file: {resolved}")
    return resolved


def _mac_address(identifier: str) -> str:
    digest = hashlib.sha256(identifier.encode("ascii")).digest()
    return "02:" + ":".join(f"{value:02x}" for value in digest[:5])


def _remove_private_tree(path: Path, *, attempts: int = 50) -> None:
    """Remove private launch scratch, tolerating short NFS silly-rename races."""

    if not isinstance(attempts, int) or isinstance(attempts, bool) or attempts < 1:
        raise ValueError("cleanup attempts must be a positive integer")
    for attempt in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.1)


def _materialize_uefi_firmware(
    *, executable: str, image: Path, storage: Path
) -> None:
    """Copy the image's firmware with host-writable modes for fakeroot QEMU.

    Docker root can open the image's read-only 0444 firmware with DAC override;
    Apptainer fakeroot cannot.  Supplying identical bytes as 0600 is the sole
    compatibility change and leaves the official entrypoint and QEMU argv intact.
    """

    source = "/usr/share/OVMF/edk2-x86_64-code.fd"
    for name in ("uefi.rom", "uefi.vars"):
        target = storage / name
        try:
            with target.open("xb") as output:
                result = subprocess.run(
                    (
                        executable,
                        "exec",
                        "--fakeroot",
                        "--contain",
                        "--cleanenv",
                        str(image),
                        "cat",
                        source,
                    ),
                    check=False,
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.PIPE,
                    timeout=30,
                )
            if result.returncode:
                detail = result.stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(
                    "OSWorld Apptainer firmware extraction failed"
                    + (f": {detail[-1000:]}" if detail else "")
                )
            if target.stat().st_size == 0:
                raise RuntimeError(
                    "OSWorld Apptainer firmware extraction returned an empty file"
                )
            target.chmod(0o600)
        except BaseException:
            target.unlink(missing_ok=True)
            raise


def _await_apptainer_launch(
    process: subprocess.Popen[bytes],
    *,
    server_port: int,
    log_path: Path,
    timeout_seconds: float = 30.0,
) -> None:
    """Return once QEMU owns its forwarded port; fail on early guest death."""

    deadline = time.monotonic() + timeout_seconds
    while True:
        returncode = process.poll()
        if returncode is not None:
            detail = _log_tail(log_path)
            raise RuntimeError(
                "OSWorld Apptainer launcher exited early"
                f" with status {returncode}: {detail}"
            )
        try:
            with socket.create_connection(("127.0.0.1", server_port), timeout=0.2):
                return
        except OSError:
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "OSWorld Apptainer launcher did not open its server port within "
                f"{timeout_seconds:g}s: {_log_tail(log_path)}"
            )
        time.sleep(0.1)


def _log_tail(path: Path, limit: int = 4096) -> str:
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - limit))
        return stream.read().decode("utf-8", errors="replace").strip()


__all__ = ["OSWorldApptainerDockerClient", "osworld_apptainer_preflight"]
