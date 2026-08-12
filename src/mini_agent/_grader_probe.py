"""Isolated grader-runtime probe.

The harness reads this file's text and executes it verbatim via
``python -I -c <source> <output> <required-json>`` inside the official grader
runtime, so the probe must stay dependency-free and single-file. Importing the
module performs no work; only ``-c`` execution (``__name__ == "__main__"``)
runs the probe.
"""

import importlib.metadata as metadata
import importlib.util
import json
import os
import sys

SCHEMA = "mini-agent-isolated-grader-runtime-v1"


class ProbeError(Exception):
    def __init__(self, code, name="", expected="", observed=""):
        self.code = code
        self.name = name
        self.expected = expected
        self.observed = observed


def emit(output, value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if len(encoded) > 32768:
        raise SystemExit(70)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    output = sys.argv[1]
    try:
        required = json.loads(sys.argv[2])
        if not isinstance(required, dict) or not required:
            raise ProbeError("request")
        packages = {}
        modules = {}
        for name in sorted(required):
            expected = required[name]
            if not isinstance(name, str) or not isinstance(expected, str):
                raise ProbeError("request")
            try:
                observed = metadata.version(name)
            except metadata.PackageNotFoundError:
                raise ProbeError("missing", name, expected)
            if observed != expected:
                raise ProbeError("version", name, expected, observed)
            spec = importlib.util.find_spec(name)
            if spec is None or not isinstance(spec.origin, str) or not spec.origin:
                raise ProbeError("module", name)
            locations = list(spec.submodule_search_locations or ())
            if len(locations) != 1 or not isinstance(locations[0], str):
                raise ProbeError("module", name)
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
                raise ProbeError("module", name)
            packages[name] = observed
            modules[name] = {"origin": origin, "package_root": package_root}
        emit(
            output,
            {
                "schema": SCHEMA,
                "ok": True,
                "python_executable": os.path.abspath(sys.executable),
                "python_version": sys.version.split()[0],
                "python_implementation": sys.implementation.name,
                "python_prefix": os.path.abspath(sys.prefix),
                "python_base_prefix": os.path.abspath(sys.base_prefix),
                "packages": packages,
                "modules": modules,
            },
        )
    except ProbeError as error:
        emit(
            output,
            {
                "schema": SCHEMA,
                "ok": False,
                "error": error.code,
                "name": error.name,
                "expected": error.expected,
                "observed": error.observed,
            },
        )
    except BaseException:
        emit(output, {"schema": SCHEMA, "ok": False, "error": "runtime"})


if __name__ == "__main__":
    main()
