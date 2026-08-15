"""Loading, answer extraction, and grading for the reasoning benchmarks."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mini_agent.benchmarks.base import BenchmarkTask
from mini_agent.benchmarks.reasoning import (
    DATASETS,
    dataset_identity,
    dataset_names,
    extract_answer,
    grade_reasoning,
    load_reasoning,
    unjudged_score,
)


def _task(benchmark: str, answer: str) -> BenchmarkTask:
    return BenchmarkTask(
        "t1",
        "prompt",
        {"benchmark": benchmark, "problem": "what is it?", "answer": answer},
    )


class AnswerExtractionTests(unittest.TestCase):
    def test_the_answer_line_wins_over_earlier_work(self) -> None:
        self.assertEqual(extract_answer("I think 3.\nAnswer: 7"), "7")

    def test_the_last_answer_line_is_the_final_one(self) -> None:
        self.assertEqual(extract_answer("Answer: 12\nAnswer: 15"), "15")

    def test_a_mid_sentence_answer_word_is_not_an_answer_line(self) -> None:
        # Otherwise a model narrating "my answer: 3 was wrong" would overwrite
        # the answer it actually stated.
        self.assertEqual(extract_answer("my answer: 3 was wrong\nAnswer: 8"), "8")

    def test_markdown_emphasis_and_math_delimiters_are_stripped(self) -> None:
        self.assertEqual(extract_answer("**Answer:** $\\boxed{7}$"), "7")

    def test_a_boxed_value_is_the_fallback(self) -> None:
        self.assertEqual(extract_answer("so \\boxed{\\frac{1}{2}}"), "\\frac{1}{2}")

    def test_nested_braces_inside_a_box_are_balanced(self) -> None:
        self.assertEqual(extract_answer("\\boxed{\\frac{1}{2}+x}"), "\\frac{1}{2}+x")

    def test_a_response_with_no_answer_extracts_nothing(self) -> None:
        self.assertIsNone(extract_answer("I could not solve it."))


class GradingTests(unittest.TestCase):
    def test_an_aime_answer_compares_as_an_integer(self) -> None:
        task = _task("aime", "204")
        self.assertEqual(grade_reasoning(task, "Answer: 204"), (1.0, "integer-exact"))
        self.assertEqual(grade_reasoning(task, "Answer: 0204")[0], 1.0)
        self.assertEqual(grade_reasoning(task, "Answer: 205")[0], 0.0)

    def test_a_non_integer_aime_answer_is_wrong_not_undecided(self) -> None:
        # The answer space is integers, so "1/2" is a wrong answer rather than
        # something a judge could rescue.
        score, reason = grade_reasoning(_task("aime", "204"), "Answer: 1/2")
        self.assertEqual((score, reason), (0.0, "not-an-integer"))

    def test_a_missing_answer_scores_zero(self) -> None:
        score, reason = grade_reasoning(_task("aime", "204"), "no idea")
        self.assertEqual((score, reason), (0.0, "no-answer-line"))

    def test_equivalent_latex_forms_match_without_a_judge(self) -> None:
        task = _task("math500", "\\frac{1}{2}")
        for response in ("Answer: \\dfrac{1}{2}", "Answer: 1/2", "Answer: \\frac12"):
            with self.subTest(response=response):
                self.assertEqual(grade_reasoning(task, response)[0], 1.0)

    def test_normalization_declines_rather_than_guessing(self) -> None:
        # 0.5 and 1/2 are the same value, but deciding that is a judge's job;
        # returning None is what lets the caller ask one.
        score, reason = grade_reasoning(_task("math500", "\\frac{1}{2}"), "Answer: 0.5")
        self.assertIsNone(score)
        self.assertEqual(reason, "normalization-undecided")

    def test_units_and_separators_do_not_change_a_value(self) -> None:
        self.assertEqual(
            grade_reasoning(_task("math500", "1000"), "Answer: 1,000")[0], 1.0
        )
        self.assertEqual(
            grade_reasoning(_task("math500", "45"), "Answer: 45^\\circ")[0], 1.0
        )

    def test_an_unknown_benchmark_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            grade_reasoning(_task("nope", "1"), "Answer: 1")


class UnjudgedPolicyTests(unittest.TestCase):
    """Without a judge, undecided is scored zero and says so."""

    def test_a_decided_grade_passes_through_with_its_reason(self) -> None:
        self.assertEqual(
            unjudged_score(1.0, "integer-exact"),
            (1.0, {"grader": "integer-exact"}),
        )

    def test_undecided_scores_zero_and_records_that_it_did(self) -> None:
        # Dropping these instead would quietly remove the answers most likely
        # to be wrong from the mean; the flag keeps the undercount measurable.
        score, grading = unjudged_score(None, "normalization-undecided")
        self.assertEqual(score, 0.0)
        self.assertTrue(grading["undecided_scored_zero"])
        self.assertEqual(grading["grader"], "normalization-undecided")


class LoadingTests(unittest.TestCase):
    def _write(self, root: Path, rows: list[dict]) -> Path:
        path = root / "tasks.jsonl"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return path

    def test_the_registered_datasets_are_the_four_documented_ones(self) -> None:
        self.assertEqual(
            set(dataset_names()),
            {"aime", "math500", "olympiadbench", "minervamath"},
        )

    def test_tasks_load_with_the_hidden_answer_out_of_the_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self._write(
                root, [{"id": "a-1", "problem": "1+1?", "answer": "2"}]
            )
            (task,) = load_reasoning(path, benchmark="aime")
            self.assertEqual(task.task_id, "a-1")
            self.assertIn("1+1?", task.prompt)
            self.assertNotIn("Answer: 2", task.prompt)
            self.assertEqual(task.data["answer"], "2")

    def test_alternate_field_spellings_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write(
                Path(temporary), [{"ID": "x", "Problem": "p", "Answer": "5"}]
            )
            (task,) = load_reasoning(path, benchmark="aime")
            self.assertEqual((task.task_id, task.data["answer"]), ("x", "5"))

    def test_an_olympiadbench_single_element_answer_list_is_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write(
                Path(temporary),
                [{"id": "o1", "question": "q", "final_answer": ["\\frac{1}{3}"]}],
            )
            (task,) = load_reasoning(path, benchmark="olympiadbench")
            self.assertEqual(task.data["answer"], "\\frac{1}{3}")

    def test_a_non_integer_aime_answer_fails_closed_at_load(self) -> None:
        # A dataset whose answers are not integers is not AIME; grading it as
        # AIME would silently score every task zero.
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write(
                Path(temporary), [{"id": "a", "problem": "p", "answer": "1/2"}]
            )
            with self.assertRaises(ValueError):
                load_reasoning(path, benchmark="aime")

    def test_duplicate_ids_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write(
                Path(temporary),
                [
                    {"id": "a", "problem": "p", "answer": "1"},
                    {"id": "a", "problem": "q", "answer": "2"},
                ],
            )
            with self.assertRaises(ValueError):
                load_reasoning(path, benchmark="aime")

    def test_a_row_without_an_answer_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write(Path(temporary), [{"id": "a", "problem": "p"}])
            with self.assertRaises(ValueError):
                load_reasoning(path, benchmark="aime")

    def test_an_empty_file_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write(Path(temporary), [])
            with self.assertRaises(ValueError):
                load_reasoning(path, benchmark="aime")

    def test_limit_takes_the_leading_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write(
                Path(temporary),
                [{"id": f"a{n}", "problem": "p", "answer": str(n)} for n in range(5)],
            )
            tasks = load_reasoning(path, benchmark="aime", limit=2)
            self.assertEqual([task.task_id for task in tasks], ["a0", "a1"])

    def test_dataset_identity_binds_the_exact_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write(
                Path(temporary), [{"id": "a", "problem": "p", "answer": "1"}]
            )
            identity = dataset_identity(path, benchmark="aime")
            self.assertEqual(identity["benchmark"], "aime")
            self.assertEqual(identity["upstream_dataset"], DATASETS["aime"].upstream)
            self.assertEqual(len(identity["export"]["sha256"]), 64)
            self.assertEqual(identity["export"]["size_bytes"], path.stat().st_size)


if __name__ == "__main__":
    unittest.main()
