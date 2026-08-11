from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence
from urllib.parse import urlsplit

from ..environments.cua import CUA_SPEED_RUN_REVISION


CUA_SPEED_RUN_REPOSITORY = "https://github.com/Pranjal2041/cua-speed-run"

TemplateMode = Literal["mini_agent_profile", "external_reference", "fixture"]


@dataclass(frozen=True)
class CUATemplateMapping:
    name: str
    mode: TemplateMode
    source_path: str
    profile: str | None = None
    note: str = ""


TEMPLATE_MAPPINGS: tuple[CUATemplateMapping, ...] = (
    CUATemplateMapping("android_scripted", "fixture", "templates/android_scripted", note="deterministic arithmetic fixture"),
    CUATemplateMapping("android_scripted_batched", "fixture", "templates/android_scripted_batched", note="deterministic batched fixture"),
    CUATemplateMapping("linux_scripted", "fixture", "templates/linux_scripted", note="deterministic desktop fixture"),
    CUATemplateMapping("qwen_vllm", "mini_agent_profile", "templates/qwen_vllm", "qwen_vllm"),
    CUATemplateMapping("qwen3vl", "mini_agent_profile", "templates/qwen3vl", "qwen3vl"),
    CUATemplateMapping("qwen35vl", "mini_agent_profile", "templates/qwen35vl", "qwen35vl"),
    CUATemplateMapping("gemini35", "mini_agent_profile", "templates/gemini35", "gemini35"),
    CUATemplateMapping("meta", "mini_agent_profile", "templates/meta", "meta"),
    CUATemplateMapping("gemini3_flash_preview", "mini_agent_profile", "templates/gemini3_flash_preview", "gemini3_flash_preview"),
    CUATemplateMapping("glm5v_turbo", "mini_agent_profile", "templates/glm5v_turbo", "glm5v_turbo"),
    CUATemplateMapping("kimi_k3", "mini_agent_profile", "templates/kimi_k3", "kimi_k3"),
    CUATemplateMapping("minimax_m3", "mini_agent_profile", "templates/minimax_m3", "minimax_m3"),
    CUATemplateMapping("glm_cua", "mini_agent_profile", "templates/glm_cua", "glm_cua"),
    CUATemplateMapping(
        "codex_cli",
        "external_reference",
        "templates/codex_cli",
        note="Codex CLI process/runtime; do not relabel as MiniAgent",
    ),
    CUATemplateMapping(
        "claude_code",
        "external_reference",
        "templates/claude_code",
        note="Claude Code process/runtime; do not relabel as MiniAgent",
    ),
    CUATemplateMapping("claude", "mini_agent_profile", "templates/claude", "claude"),
    CUATemplateMapping("gpt54", "mini_agent_profile", "templates/gpt54", "gpt54"),
    CUATemplateMapping("qwen_vl_remote", "mini_agent_profile", "templates/qwen_vl_remote", "qwen_vl_remote"),
)

_BY_NAME = {mapping.name: mapping for mapping in TEMPLATE_MAPPINGS}


@dataclass(frozen=True)
class SubmissionExport:
    directory: Path
    template: str
    profile: str
    files: Mapping[str, str]
    runtime_wheel_sha256: str
    dependency_wheel_sha256: Mapping[str, str]
    source_revision: str = CUA_SPEED_RUN_REVISION


def template_mapping(name: str) -> CUATemplateMapping:
    try:
        return _BY_NAME[name]
    except KeyError as exc:
        raise ValueError(f"unknown cua-speed-run template {name!r}") from exc


def build_profile_environment(profile: Any, client: Any) -> Any:
    """Apply executable CUA profile policy without putting it in MiniAgent."""

    from ..environments.cua import CUAEnvironment

    return CUAEnvironment.from_policy(
        client,
        benchmark=getattr(profile, "benchmark", {}),
        observation=getattr(profile, "observation", {}),
        history=getattr(profile, "history", {}),
        tools=getattr(profile, "tools", ()),
        response_parser=getattr(profile, "response_parser", ""),
        provider=getattr(profile, "provider", ""),
    )


def _validate_text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{label} must be a non-empty string without NUL bytes")
    return value


def validate_env_url(env_url: str) -> str:
    value = _validate_text(env_url, "env_url")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("env_url must be an absolute HTTP(S) URL")
    if parsed.query or parsed.fragment:
        raise ValueError("env_url may not contain a query or fragment")
    return value.rstrip("/")


def validate_submission(directory: Path) -> tuple[Path, Path]:
    root = directory.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"submission directory does not exist: {root}")
    names = sorted(path.name for path in root.iterdir())
    if names != ["agent.py", "init.py"]:
        raise ValueError("cua-speed-run submission must contain exactly init.py and agent.py")
    init_script = root / "init.py"
    agent_script = root / "agent.py"
    for script in (init_script, agent_script):
        if script.is_symlink() or not script.is_file():
            raise ValueError(f"submission script must be a regular file: {script.name}")
    return init_script, agent_script


def build_agent_argv(
    python_executable: str | Path,
    agent_script: Path,
    env_url: str,
    task_description: str,
) -> tuple[str, ...]:
    """Build the exact agent-plane argv without invoking a shell."""

    python = _validate_text(str(python_executable), "python_executable")
    agent = agent_script.expanduser().resolve()
    if agent.name != "agent.py" or agent.is_symlink() or not agent.is_file():
        raise ValueError("agent_script must be a regular file named agent.py")
    return (
        python,
        str(agent),
        validate_env_url(env_url),
        _validate_text(task_description, "task_description"),
    )


def build_cua_speed_run_argv(
    executable: str | Path,
    *,
    submission: Path,
    benchmark: Path,
    output_root: Path,
    task_ids: Sequence[str] = (),
    remote: bool = False,
    agent_mode: str | None = None,
    agents_per_evaluation: int = 1,
    parallel_evaluations: int = 1,
) -> tuple[str, ...]:
    """Build a pinned upstream practice-evaluation argv without shell text."""

    command = _validate_text(str(executable), "cua-speedrun executable")
    validate_submission(submission)
    benchmark_root = benchmark.expanduser().resolve()
    if not benchmark_root.is_dir() or not (benchmark_root / "manifest.yaml").is_file():
        raise ValueError("benchmark must contain manifest.yaml")
    if agents_per_evaluation < 1 or parallel_evaluations < 1:
        raise ValueError("evaluation concurrency values must be positive")
    out = output_root.expanduser().resolve()
    argv = [
        command,
        "run",
        "--submission",
        str(submission.expanduser().resolve()),
        "--benchmark",
        str(benchmark_root),
        "--out",
        str(out),
        "--agents-per-evaluation",
        str(agents_per_evaluation),
        "--parallel-evaluations",
        str(parallel_evaluations),
    ]
    if remote:
        argv.append("--remote")
    if agent_mode is not None:
        if agent_mode not in {"shared", "per-task", "shared-agent-vllm@2", "per-task-vllm@1"}:
            raise ValueError("unsupported cua-speed-run agent mode")
        argv.extend(["--agent-mode", agent_mode])
    for task_id in task_ids:
        argv.extend(["--task", _validate_text(task_id, "task_id")])
    return tuple(argv)


def _script_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wheel_payload(path: Path) -> tuple[Path, bytes, str, str]:
    wheel = path.expanduser().resolve()
    if not wheel.is_file() or wheel.suffix != ".whl" or wheel.is_symlink():
        raise ValueError("bundled wheels must be regular .whl files")
    try:
        with zipfile.ZipFile(wheel) as archive:
            metadata_names = [
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_names) != 1:
                raise ValueError("wheel must contain exactly one METADATA file")
            metadata = archive.read(metadata_names[0]).decode("utf-8")
    except (OSError, UnicodeError, zipfile.BadZipFile, KeyError) as exc:
        raise ValueError(f"invalid bundled wheel: {wheel}: {exc}") from exc
    content = wheel.read_bytes()
    return wheel, content, hashlib.sha256(content).hexdigest(), metadata


def _runtime_wheel(path: Path | None) -> tuple[Path, bytes, str, str]:
    candidates: list[Path] = []
    if path is not None:
        candidates.append(path.expanduser().resolve())
    else:
        for directory in (Path.cwd() / "dist", Path(__file__).resolve().parents[3] / "dist"):
            candidates.extend(sorted(directory.glob("mini_agent-0.3.0-*.whl")))
    unique = list(dict.fromkeys(candidate.resolve() for candidate in candidates))
    if len(unique) != 1:
        raise ValueError(
            "CUA export requires exactly one mini-agent 0.3.0 wheel; pass --wheel"
        )
    wheel, content, digest, metadata = _wheel_payload(unique[0])
    with zipfile.ZipFile(wheel) as archive:
        if "mini_agent/__init__.py" not in archive.namelist():
            raise ValueError("runtime wheel is not a mini-agent wheel")
    if "\nName: mini-agent\n" not in f"\n{metadata}" or "\nVersion: 0.3.0\n" not in f"\n{metadata}":
        raise ValueError("runtime wheel must contain mini-agent version 0.3.0")
    return wheel, content, digest, metadata


def _verified_wheel_bundle(
    runtime: tuple[Path, bytes, str, str], dependency_wheels: Sequence[Path]
) -> tuple[tuple[Path, bytes, str, str], ...]:
    dependencies = tuple(_wheel_payload(path) for path in dependency_wheels)
    bundle = (runtime, *dependencies)
    names = [item[0].name for item in bundle]
    if len(names) != len(set(names)):
        raise ValueError("bundled wheel filenames must be unique")
    if any("\nName: mini-agent\n" in f"\n{item[3]}" for item in dependencies):
        raise ValueError("dependency wheels cannot contain another mini-agent runtime")
    runtime_requires_dependencies = "\nRequires-Dist:" in f"\n{runtime[3]}"
    if runtime_requires_dependencies and not dependencies:
        raise ValueError(
            "the mini-agent wheel declares dependencies; pass a complete "
            "--dependency-wheel bundle for an offline, artifact-pinned export"
        )
    with tempfile.TemporaryDirectory(prefix="mini-agent-export-wheelcheck-") as directory:
        wheelhouse = Path(directory)
        for path, content, _, _ in bundle:
            (wheelhouse / path.name).write_bytes(content)
        checked = subprocess.run(
            (
                sys.executable,
                "-m",
                "pip",
                "install",
                "--ignore-installed",
                "--no-index",
                "--find-links",
                str(wheelhouse),
                "--target",
                str(wheelhouse / "installed"),
                "--no-compile",
                str(wheelhouse / runtime[0].name),
            ),
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if checked.returncode != 0:
            detail = (checked.stderr or checked.stdout)[-4000:]
            raise ValueError(f"offline wheel bundle is not installable: {detail}")
    return bundle


def export_submission(
    output_directory: Path,
    *,
    template: str,
    model: str,
    provider: str,
    required_environment: Sequence[str] = (),
    mode: str = "single",
    max_agents: int = 4,
    child_profiles: Sequence[str] = (),
    wheel: Path | None = None,
    dependency_wheels: Sequence[Path] = (),
) -> SubmissionExport:
    """Export a safe two-file MiniAgent submission for the upstream runner.

    External process templates and deterministic fixtures intentionally cannot
    be exported through this function: they retain their upstream identity.
    """

    mapping = template_mapping(template)
    if mapping.mode != "mini_agent_profile" or mapping.profile is None:
        raise ValueError(
            f"template {template!r} is {mapping.mode}, not a MiniAgent profile"
        )
    selected_model = _validate_text(model, "model")
    selected_provider = _validate_text(provider, "provider")
    supported_providers = {
        "openai-responses": "OPENAI_API_KEY",
        "anthropic-messages": "ANTHROPIC_API_KEY",
        "openai-compatible-chat": "MODEL_API_KEY",
    }
    if selected_provider not in supported_providers:
        raise ValueError(f"unsupported MiniAgent provider {selected_provider!r}")
    from ..profiles import load_profile

    mapped_profile = load_profile("cua", mapping.profile)
    if mapped_profile.benchmark.get("template") != template:
        raise ValueError(
            f"profile {mapping.profile!r} is not coupled to template {template!r}"
        )
    if mapped_profile.provider and selected_provider != mapped_profile.provider:
        raise ValueError(
            f"provider {selected_provider!r} is incompatible with profile "
            f"{mapping.profile!r} ({mapped_profile.provider!r})"
        )
    runtime_wheel = _runtime_wheel(wheel)
    bundle = _verified_wheel_bundle(runtime_wheel, dependency_wheels)
    wheel_path, _, wheel_sha256, _ = runtime_wheel
    if mode not in {"single", "multi"}:
        raise ValueError("submission mode must be single or multi")
    if not isinstance(max_agents, int) or isinstance(max_agents, bool) or max_agents < 1:
        raise ValueError("max_agents must be a positive integer")
    selected_child_profiles = [
        _validate_text(profile, "child_profile") for profile in child_profiles
    ]
    if len(selected_child_profiles) != len(set(selected_child_profiles)):
        raise ValueError("child profiles must be unique")
    environment_names = [supported_providers[selected_provider]]
    for name in required_environment:
        if not isinstance(name, str) or not name or not name.replace("_", "").isalnum():
            raise ValueError(f"invalid environment variable name {name!r}")
        if name not in environment_names:
            environment_names.append(name)

    destination = output_directory.expanduser().resolve()
    if destination.exists():
        raise ValueError(f"submission destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    try:
        init_source = (
            "from __future__ import annotations\n\n"
            "import base64\n"
            "import hashlib\n"
            "import os\n\n"
            "import subprocess\n"
            "import sys\n"
            "import tempfile\n"
            "from pathlib import Path\n\n"
            f"REQUIRED_ENVIRONMENT = {environment_names!r}\n\n"
            f"RUNTIME_WHEEL = {wheel_path.name!r}\n"
            f"WHEELS = {[(path.name, digest, base64.b64encode(content).decode('ascii')) for path, content, digest, _ in bundle]!r}\n\n"
            "missing = [name for name in REQUIRED_ENVIRONMENT if not os.environ.get(name)]\n"
            "if missing:\n"
            "    raise SystemExit('missing required environment: ' + ', '.join(missing))\n"
            "with tempfile.TemporaryDirectory(prefix='mini-agent-wheel-') as directory:\n"
            "    root = Path(directory)\n"
            "    for name, expected, encoded in WHEELS:\n"
            "        wheel = base64.b64decode(encoded, validate=True)\n"
            "        if hashlib.sha256(wheel).hexdigest() != expected:\n"
            "            raise SystemExit('embedded wheel failed its SHA-256 check: ' + name)\n"
            "        (root / name).write_bytes(wheel)\n"
            "    subprocess.run([sys.executable, '-m', 'pip', 'install', "
            "'--disable-pip-version-check', '--no-index', '--find-links', "
            "str(root), str(root / RUNTIME_WHEEL)], check=True)\n"
            "print('mini-agent init ready')\n"
        )
        agent_source = (
            "from __future__ import annotations\n\n"
            "import subprocess\n"
            "import sys\n\n"
            f"MODEL = {selected_model!r}\n"
            f"PROFILE = {mapping.profile!r}\n"
            f"PROVIDER = {selected_provider!r}\n\n"
            f"MODE = {mode!r}\n"
            f"MAX_AGENTS = {max_agents!r}\n"
            f"CHILD_PROFILES = {selected_child_profiles!r}\n\n"
            "def main() -> int:\n"
            "    if len(sys.argv) != 3:\n"
            "        raise SystemExit('usage: python agent.py <env_url> <task_description>')\n"
            "    env_url, task = sys.argv[1:]\n"
            "    argv = [\n"
            "        sys.executable, '-m', 'mini_agent', 'run',\n"
            "        '--application', 'cua', '--model', MODEL, '--profile', PROFILE,\n"
            "        '--provider', PROVIDER, '--env-url', env_url, '--task', task,\n"
            "    ]\n"
            "    if MODE == 'multi':\n"
            "        argv.extend(['--mode', 'multi', '--max-agents', str(MAX_AGENTS)])\n"
            "        for profile in CHILD_PROFILES:\n"
            "            argv.extend(['--child-profile', profile])\n"
            "    return subprocess.call(argv)\n\n"
            "if __name__ == '__main__':\n"
            "    raise SystemExit(main())\n"
        )
        init_path = staging / "init.py"
        agent_path = staging / "agent.py"
        init_path.write_text(init_source, encoding="utf-8")
        agent_path.write_text(agent_source, encoding="utf-8")
        init_path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        agent_path.chmod(
            stat.S_IRUSR
            | stat.S_IWUSR
            | stat.S_IXUSR
            | stat.S_IRGRP
            | stat.S_IXGRP
            | stat.S_IROTH
            | stat.S_IXOTH
        )
        os.replace(staging, destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    init_path, agent_path = validate_submission(destination)
    return SubmissionExport(
        directory=destination,
        template=template,
        profile=mapping.profile,
        files={"init.py": _script_hash(init_path), "agent.py": _script_hash(agent_path)},
        runtime_wheel_sha256=wheel_sha256,
        dependency_wheel_sha256={
            path.name: digest for path, _, digest, _ in bundle[1:]
        },
    )


def mapping_manifest() -> Mapping[str, object]:
    return {
        "repository": CUA_SPEED_RUN_REPOSITORY,
        "revision": CUA_SPEED_RUN_REVISION,
        "templates": [
            {
                "name": item.name,
                "mode": item.mode,
                "source_path": item.source_path,
                "profile": item.profile,
                "note": item.note,
            }
            for item in TEMPLATE_MAPPINGS
        ],
    }


def preflight(
    *,
    profiles_directory: Path | None = None,
    submission: Path | None = None,
    benchmark: Path | None = None,
) -> Mapping[str, object]:
    """Return doctor-readable, side-effect-free CUA integration checks."""

    profile_root = (
        profiles_directory.expanduser().resolve()
        if profiles_directory is not None
        else Path(__file__).resolve().parents[1] / "profiles" / "cua"
    )
    names = [item.name for item in TEMPLATE_MAPPINGS]
    modes = {
        mode: sum(item.mode == mode for item in TEMPLATE_MAPPINGS)
        for mode in ("mini_agent_profile", "external_reference", "fixture")
    }
    missing_profiles = sorted(
        item.profile
        for item in TEMPLATE_MAPPINGS
        if item.mode == "mini_agent_profile"
        and item.profile is not None
        and not (profile_root / f"{item.profile}.yaml").is_file()
    )
    checks: list[dict[str, object]] = [
        {
            "name": "template_catalog",
            "ok": len(names) == 18 and len(set(names)) == 18,
            "detail": {"total": len(names), **modes},
        },
        {
            "name": "mini_agent_profiles",
            "ok": not missing_profiles and modes["mini_agent_profile"] == 13,
            "detail": {"directory": str(profile_root), "missing": missing_profiles},
        },
        {
            "name": "adapter_source_contract",
            "ok": True,
            "detail": {
                "repository": CUA_SPEED_RUN_REPOSITORY,
                "expected_revision": CUA_SPEED_RUN_REVISION,
                "scope": "packaged agent-side adapter",
            },
        },
    ]
    if submission is not None:
        try:
            validate_submission(submission)
        except ValueError as exc:
            checks.append({"name": "submission", "ok": False, "detail": str(exc)})
        else:
            checks.append(
                {
                    "name": "submission",
                    "ok": True,
                    "detail": str(submission.expanduser().resolve()),
                }
            )
    if benchmark is not None:
        root = benchmark.expanduser().resolve()
        ok = root.is_dir() and (root / "manifest.yaml").is_file()
        checks.append(
            {
                "name": "benchmark",
                "ok": ok,
                "detail": str(root),
            }
        )
    return {
        "application": "cua",
        "ok": all(bool(check["ok"]) for check in checks),
        "checks": checks,
    }


def write_mapping_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(mapping_manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "CUA_SPEED_RUN_REPOSITORY",
    "CUATemplateMapping",
    "SubmissionExport",
    "TEMPLATE_MAPPINGS",
    "build_agent_argv",
    "build_profile_environment",
    "build_cua_speed_run_argv",
    "export_submission",
    "mapping_manifest",
    "preflight",
    "template_mapping",
    "validate_env_url",
    "validate_submission",
    "write_mapping_manifest",
]
