"""Pin the byte values of every digest that lands in a durable artifact.

Operators record these hashes (`--index-sha256`, evaluation manifests, grade
provenance) and compare them across runs and machines, so a refactor that
changes a digest silently invalidates recorded evidence. A prior refactor did
exactly that: unifying four stat-identity tuples reordered the fields, and
`directory_sha256` began folding `st_mtime_ns` where it had folded `st_size` —
a per-machine value in every recorded index hash. The full suite passed.

These expectations are golden values. If one fails, the digest changed: either
revert the change or treat it as a documented format break.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mini_agent._hash import (
    canonical_bytes,
    canonical_digest,
    canonical_text,
    immutable_file_identity,
    immutable_tree_identity,
)
from mini_agent.environments.web import directory_sha256


def _fixture_tree(root: Path) -> None:
    """A tree with nesting, a unicode name, and an empty file."""

    (root / "sub").mkdir()
    (root / "a.txt").write_bytes(b"alpha\n")
    (root / "sub" / "b.txt").write_bytes(b"beta\n")
    (root / "sub" / "é.txt").write_bytes(b"accent\n")
    (root / "empty.bin").write_bytes(b"")


class CanonicalEncodingTests(unittest.TestCase):
    """The encoders that feed spec fingerprints and manifest identities."""

    VALUE = {
        "b": 1,
        "a": [3, {"z": None, "y": True}],
        "unicode": "café — naïve",
        "escape": 'quote " backslash \\ newline \n',
    }

    def test_canonical_bytes_is_sorted_compact_ascii_escaped(self) -> None:
        self.assertEqual(
            canonical_bytes(self.VALUE),
            b'{"a":[3,{"y":true,"z":null}],"b":1,'
            b'"escape":"quote \\" backslash \\\\ newline \\n",'
            b'"unicode":"caf\\u00e9 \\u2014 na\\u00efve"}',
        )

    def test_canonical_text_keeps_non_ascii_literal(self) -> None:
        # specs.py fingerprints depend on ensure_ascii=False; a switch to the
        # escaped form would change every recorded agent-spec fingerprint.
        self.assertIn("café — naïve", canonical_text(self.VALUE))

    def test_canonical_digest_is_the_escaped_byte_form(self) -> None:
        # Golden value, verified equal to the pre-unification
        # runtime._canonical_digest and benchmarks.base._canonical_bytes.
        self.assertEqual(
            canonical_digest(self.VALUE),
            "23c30593c42ff57a8e0a9ed4a1df16d3097204659d86990dc3a58be85e99784e",
        )

    def test_rejects_non_finite_numbers(self) -> None:
        with self.assertRaises(ValueError):
            canonical_bytes({"nan": float("nan")})


class FileAndTreeDigestTests(unittest.TestCase):
    def test_file_digest_is_content_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "a.txt"
            path.write_bytes(b"alpha\n")
            identity = immutable_file_identity(path)
            self.assertEqual(
                identity["sha256"],
                "b6a98d9ce9a2d9149288fa3df42d377c3e42737afdcdaf714e33c0a100b51060",
            )
            self.assertEqual(identity["size_bytes"], 6)

    def test_tree_digest_is_content_and_relative_paths_only(self) -> None:
        # Must not depend on mtime, inode, mode, or absolute location: the same
        # tree materialized on another machine has to hash the same.
        digests = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _fixture_tree(root)
                digests.append(immutable_tree_identity(root)["sha256"])
        self.assertEqual(digests[0], digests[1])

    def test_directory_sha256_is_stable_across_machines(self) -> None:
        digests = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _fixture_tree(root)
                digests.append(directory_sha256(root))
        self.assertEqual(digests[0], digests[1])

    def test_directory_sha256_reacts_to_content_and_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _fixture_tree(root)
            baseline = directory_sha256(root)
            (root / "a.txt").write_bytes(b"alphb\n")
            self.assertNotEqual(directory_sha256(root), baseline)
            (root / "a.txt").write_bytes(b"alpha\n")
            self.assertEqual(directory_sha256(root), baseline)
            (root / "a.txt").rename(root / "renamed.txt")
            self.assertNotEqual(directory_sha256(root), baseline)


if __name__ == "__main__":
    unittest.main()
