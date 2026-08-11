from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from mini_agent.cli import main
from mini_agent.references import (
    StudyRuntime,
    get_reference,
    get_study_runtime,
    list_study_runtimes,
)


EXPECTED_STUDIES = (
    "web/browser-use-parallel-pattern",
    "web/macu-text-dag",
    "web/anthropic-fable5-team-3",
    "web/anthropic-fable5-team-5",
    "web/anthropic-fable5-team-10",
    "web/anthropic-fable5-blocking",
    "web/anthropic-fable5-async-browsecomp",
    "web/anthropic-opus5-team-5",
    "web/anthropic-opus5-team-10",
    "web/anthropic-opus5-async",
    "cua/openai-hosted-multi-agent-computer",
    "cua/macu-text-dag",
    "swe/single-agent-control",
    "swe/parallel-best-of-3",
    "swe/openai-hosted-multi-agent-shell",
    "swe/openai-hosted-multi-agent-swe-computer",
    "swe/prime-agent-source-0.7.1",
    "swe/rlm-0.1.3-contract",
    "swe/platoon-recursive-inference",
    "swe/macu-swe-computer-experiment",
    "swe/anthropic-fable5-team-3",
    "swe/anthropic-fable5-team-5",
    "swe/anthropic-fable5-team-10",
    "swe/anthropic-fable5-blocking",
    "swe/anthropic-fable5-async-programbench",
    "swe/anthropic-opus5-team-5",
    "swe/anthropic-opus5-team-10",
    "swe/anthropic-opus5-async",
)

ENVIRONMENT_CONFIG = {
    "browser": "configs/domain_browser.json",
    "computer": "configs/domain_computer.json",
    "swe": "configs/domain_swe.json",
    "swe_computer": "configs/domain_swe_computer.json",
}


class AllStudyAcceptanceTests(unittest.TestCase):
    def test_inventory_is_explicit_and_studies_never_become_references(self) -> None:
        studies = list_study_runtimes()
        self.assertEqual(len(studies), 28)
        self.assertEqual(tuple(study.key for study in studies), EXPECTED_STUDIES)
        for study in studies:
            with self.subTest(study=study.key):
                self.assertEqual(study.profile.catalog_kind, "study")
                self.assertEqual(study.manifest()["execution_mode"], "study")
        with self.assertRaisesRegex(ValueError, "not a study"):
            get_study_runtime("web", "openai-hosted-web-search")
        with self.assertRaisesRegex(ValueError, "not a runnable study"):
            StudyRuntime(
                "web",
                get_reference("web", "openai-hosted-web-search").profile,
            )

    def test_mini_agent_cli_validates_every_study_offline(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks = root / "tasks.jsonl"
            tasks.write_text(
                json.dumps(
                    {
                        "id": "one",
                        "prompt": "answer",
                        "context": "deterministic context",
                        "parallel_tasks": [
                            {"id": "child", "prompt": "answer independently"}
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            for key in EXPECTED_STUDIES:
                application, name = key.split("/", 1)
                with self.subTest(study=key):
                    study = get_study_runtime(application, name)
                    environment_type = study.profile.environment_types[0]
                    raw: dict[str, object] = {
                        "application": {"name": application, "study": name},
                        "limits": {"max_model_calls": 1},
                    }
                    if environment_type != "none":
                        environment_source = (
                            repository / ENVIRONMENT_CONFIG[environment_type]
                        )
                        raw["environment"] = json.loads(
                            environment_source.read_text(encoding="utf-8")
                        )["environment"]
                    config = root / f"{application}-{name}.json"
                    config.write_text(json.dumps(raw), encoding="utf-8")
                    output = io.StringIO()
                    with redirect_stdout(output):
                        result = main(
                            (
                                "validate-study",
                                "--application",
                                application,
                                "--study",
                                name,
                                "--tasks",
                                str(tasks),
                                "--config",
                                str(config),
                                "--provider",
                                study.providers[0],
                            )
                        )
                    self.assertEqual(result, 0)
                    payload = json.loads(output.getvalue())
                    self.assertIs(payload["valid"], True)
                    self.assertEqual(
                        payload["application"]["study"]["key"],
                        study.profile.key,
                    )

    def test_eval_study_forwards_literal_runtime_arguments(self) -> None:
        study = Mock()
        study.run.return_value = 0
        with patch("mini_agent.references.get_study_runtime", return_value=study):
            result = main(
                (
                    "eval-study",
                    "--application",
                    "swe",
                    "--study",
                    "single-agent-control",
                    "--tasks",
                    "tasks.jsonl",
                    "--config",
                    "config.json",
                    "--output",
                    "runs/study",
                    "--",
                    "--model",
                    "example-model",
                )
            )
        self.assertEqual(result, 0)
        study.run.assert_called_once_with(
            tasks=Path("tasks.jsonl"),
            config=Path("config.json"),
            output=Path("runs/study"),
            provider=None,
            arguments=("--model", "example-model"),
        )


if __name__ == "__main__":
    unittest.main()
