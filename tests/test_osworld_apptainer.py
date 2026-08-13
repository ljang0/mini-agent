from __future__ import annotations

import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mini_agent.runtimes.osworld_apptainer import (
    OSWorldApptainerDockerClient,
    _materialize_uefi_firmware,
    _remove_private_tree,
    osworld_apptainer_preflight,
)


class _FakeProcess:
    def __init__(self) -> None:
        self.pid = 12345
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, *, timeout: float) -> int:
        del timeout
        self.returncode = 0
        return self.returncode


class OSWorldApptainerTests(unittest.TestCase):
    def test_running_container_registry_is_shared_across_clients(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            container_image = root / "osworld.sif"
            container_image.write_bytes(b"sif")
            vm_image = root / "Ubuntu.qcow2"
            vm_image.write_bytes(b"qcow2")
            first = OSWorldApptainerDockerClient(
                apptainer_image=container_image,
                vm_image=vm_image,
                work_root=root / "first",
                executable=sys.executable,
            )
            second = OSWorldApptainerDockerClient(
                apptainer_image=container_image,
                vm_image=vm_image,
                work_root=root / "second",
                executable=sys.executable,
            )
            process = _FakeProcess()
            with (
                patch(
                    "mini_agent.runtimes.osworld_apptainer.subprocess.Popen",
                    return_value=process,
                ),
                patch(
                    "mini_agent.runtimes.osworld_apptainer."
                    "_materialize_uefi_firmware"
                ),
                patch(
                    "mini_agent.runtimes.osworld_apptainer."
                    "_await_apptainer_launch"
                ),
                patch("mini_agent.runtimes.osworld_apptainer.os.killpg"),
            ):
                container = first.containers.run(
                    "happysixd/osworld-docker",
                    environment={
                        "DISK_SIZE": "32G",
                        "RAM_SIZE": "4G",
                        "CPU_CORES": "4",
                    },
                    cap_add=["NET_ADMIN"],
                    devices=["/dev/kvm"],
                    volumes={
                        str(vm_image): {"bind": "/System.qcow2", "mode": "ro"}
                    },
                    ports={8006: 18006, 5000: 15000, 9222: 19222, 8080: 18080},
                    detach=True,
                )
                self.assertIn(container, second.containers.list())
                container.stop()
                container.remove(v=True)
                self.assertNotIn(container, second.containers.list())

    def test_preflight_checks_fakeroot_qemu_without_launching_a_machine(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "osworld.sif"
            image.write_bytes(b"sif")
            completed = subprocess.CompletedProcess(
                (), 0, "QEMU emulator version 9.1.0\n", ""
            )
            with (
                patch(
                    "mini_agent.runtimes.osworld_apptainer.shutil.which",
                    return_value="/usr/bin/apptainer",
                ),
                patch(
                    "mini_agent.runtimes.osworld_apptainer.Path.exists",
                    return_value=True,
                ),
                patch(
                    "mini_agent.runtimes.osworld_apptainer.os.access",
                    return_value=True,
                ),
                patch(
                    "mini_agent.runtimes.osworld_apptainer.subprocess.run",
                    return_value=completed,
                ) as run,
            ):
                report = osworld_apptainer_preflight(image)
            argv = run.call_args.args[0]
            self.assertEqual(
                argv[1:5], ("exec", "--fakeroot", "--contain", "--cleanenv")
            )
            self.assertEqual(argv[-2:], ("qemu-system-x86_64", "--version"))
            self.assertEqual(report["qemu_version"], "QEMU emulator version 9.1.0")
            self.assertFalse(report["machine_launch_canary_run"])

    def test_official_container_contract_launches_and_cleans_ephemeral_storage(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            container_image = root / "osworld.sif"
            container_image.write_bytes(b"sif")
            vm_image = root / "Ubuntu.qcow2"
            vm_image.write_bytes(b"qcow2")
            work = root / "work"
            process = _FakeProcess()
            ports = {8006: 18006, 5000: 15000, 9222: 19222, 8080: 18080}
            client = OSWorldApptainerDockerClient(
                apptainer_image=container_image,
                vm_image=vm_image,
                work_root=work,
                executable=sys.executable,
            )

            with (
                patch(
                    "mini_agent.runtimes.osworld_apptainer.subprocess.Popen",
                    return_value=process,
                ) as popen,
                patch(
                    "mini_agent.runtimes.osworld_apptainer."
                    "_materialize_uefi_firmware"
                ),
                patch(
                    "mini_agent.runtimes.osworld_apptainer."
                    "_await_apptainer_launch"
                ),
                patch("mini_agent.runtimes.osworld_apptainer.os.killpg") as killpg,
            ):
                container = client.containers.run(
                    "happysixd/osworld-docker",
                    environment={
                        "DISK_SIZE": "32G",
                        "RAM_SIZE": "4G",
                        "CPU_CORES": "4",
                    },
                    cap_add=["NET_ADMIN"],
                    devices=["/dev/kvm"],
                    volumes={
                        str(vm_image): {"bind": "/System.qcow2", "mode": "ro"}
                    },
                    ports=ports,
                    detach=True,
                )
                argv = popen.call_args.args[0]
                self.assertEqual(argv[:2], [shutil.which(sys.executable), "run"])
                self.assertEqual(argv[-1], str(container_image.resolve()))
                self.assertIn("--pid", argv)
                self.assertNotIn("/run/entry.sh", argv)
                self.assertIn("DISPLAY=disabled", argv)
                self.assertIn("MONITOR=none", argv)
                network = next(work.glob("*/network.sh")).read_text()
                self.assertIn("hostfwd=tcp:127.0.0.1:${OSWORLD_SERVER_PORT}", network)
                self.assertEqual(
                    container.attrs["NetworkSettings"]["Ports"]["5000/tcp"],
                    [{"HostPort": "15000"}],
                )

                container.stop()
                killpg.assert_called_once_with(process.pid, signal.SIGTERM)
                container.remove(v=True)
                self.assertFalse(container.directory.exists())

    def test_launch_failures_remove_private_scratch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            container_image = root / "osworld.sif"
            container_image.write_bytes(b"sif")
            vm_image = root / "Ubuntu.qcow2"
            vm_image.write_bytes(b"qcow2")
            work = root / "work"
            client = OSWorldApptainerDockerClient(
                apptainer_image=container_image,
                vm_image=vm_image,
                work_root=work,
                executable=sys.executable,
            )
            arguments = {
                "environment": {
                    "DISK_SIZE": "32G",
                    "RAM_SIZE": "4G",
                    "CPU_CORES": "4",
                },
                "cap_add": ["NET_ADMIN"],
                "devices": ["/dev/kvm"],
                "volumes": {
                    str(vm_image): {"bind": "/System.qcow2", "mode": "ro"}
                },
                "ports": {8006: 18006, 5000: 15000, 9222: 19222, 8080: 18080},
                "detach": True,
            }
            with (
                patch(
                    "mini_agent.runtimes.osworld_apptainer."
                    "_materialize_uefi_firmware"
                ),
                patch(
                    "mini_agent.runtimes.osworld_apptainer.subprocess.Popen",
                    side_effect=OSError("launch failed"),
                ),
                self.assertRaisesRegex(OSError, "launch failed"),
            ):
                client.containers.run("happysixd/osworld-docker", **arguments)
            self.assertEqual(list(work.iterdir()), [])

            process = _FakeProcess()
            process.returncode = 2
            with (
                patch(
                    "mini_agent.runtimes.osworld_apptainer."
                    "_materialize_uefi_firmware"
                ),
                patch(
                    "mini_agent.runtimes.osworld_apptainer.subprocess.Popen",
                    return_value=process,
                ),
                self.assertRaisesRegex(RuntimeError, "exited early"),
            ):
                client.containers.run("happysixd/osworld-docker", **arguments)
            self.assertEqual(list(work.iterdir()), [])

    def test_fakeroot_firmware_is_image_derived_and_host_writable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "osworld.sif"
            image.write_bytes(b"sif")
            storage = root / "storage"
            storage.mkdir()

            def extract(argv: object, **kwargs: object) -> subprocess.CompletedProcess:
                output = kwargs["stdout"]
                output.write(b"exact-image-firmware")
                return subprocess.CompletedProcess(argv, 0, b"", b"")

            with patch(
                "mini_agent.runtimes.osworld_apptainer.subprocess.run",
                side_effect=extract,
            ) as run:
                _materialize_uefi_firmware(
                    executable="/usr/bin/apptainer",
                    image=image,
                    storage=storage,
                )
            self.assertEqual(run.call_count, 2)
            for name in ("uefi.rom", "uefi.vars"):
                firmware = storage / name
                self.assertEqual(firmware.read_bytes(), b"exact-image-firmware")
                self.assertEqual(firmware.stat().st_mode & 0o777, 0o600)

    def test_private_scratch_cleanup_retries_nfs_directory_races(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "launch"
            target.mkdir()
            failures = [OSError("directory not empty"), None]
            original_remove = shutil.rmtree

            def remove(path: Path) -> None:
                failure = failures.pop(0)
                if failure is not None:
                    raise failure
                original_remove(path)

            with (
                patch(
                    "mini_agent.runtimes.osworld_apptainer.shutil.rmtree",
                    side_effect=remove,
                ),
                patch("mini_agent.runtimes.osworld_apptainer.time.sleep") as sleep,
            ):
                _remove_private_tree(target, attempts=2)
            sleep.assert_called_once_with(0.1)
            self.assertFalse(target.exists())

    def test_changed_upstream_contract_is_rejected_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            container_image = root / "osworld.sif"
            container_image.write_bytes(b"sif")
            vm_image = root / "Ubuntu.qcow2"
            vm_image.write_bytes(b"qcow2")
            client = OSWorldApptainerDockerClient(
                apptainer_image=container_image,
                vm_image=vm_image,
                work_root=root / "work",
                executable=sys.executable,
            )
            with (
                patch(
                    "mini_agent.runtimes.osworld_apptainer.subprocess.Popen"
                ) as popen,
                self.assertRaisesRegex(ValueError, "port contract"),
            ):
                client.containers.run(
                    "happysixd/osworld-docker",
                    environment={
                        "DISK_SIZE": "32G",
                        "RAM_SIZE": "4G",
                        "CPU_CORES": "4",
                    },
                    cap_add=["NET_ADMIN"],
                    devices=["/dev/kvm"],
                    volumes={
                        str(vm_image): {"bind": "/System.qcow2", "mode": "ro"}
                    },
                    ports={5000: 15000},
                    detach=True,
                )
            popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
