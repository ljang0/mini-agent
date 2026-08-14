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

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mini_agent._hash import (
    canonical_bytes,
    canonical_digest,
    canonical_text,
    immutable_file_identity,
    immutable_tree_identity,
    machine_image_identity,
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


class MachineImageIdentityCacheTests(unittest.TestCase):
    """The image cache must never answer for bytes it did not hash.

    Machine images are tens of gigabytes and every desktop launch asks for the
    same file's identity, so the result is memoized. A cache that outlived the
    bytes it named would turn the integrity check into a no-op, which is the
    one failure this layer exists to prevent.
    """

    def _sidecar(self, image: Path, digest: str) -> Path:
        sidecar = Path(str(image) + ".provenance.json")
        sidecar.write_text(
            json.dumps({"schema": "fixture-v1", "final_image_sha256": digest})
        )
        return sidecar

    def test_repeated_identity_of_an_unchanged_image_is_not_rehashed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "image.qcow2"
            image.write_bytes(b"image-bytes")
            first = machine_image_identity(image, label="fixture image")
            with mock.patch(
                "mini_agent._hash.immutable_file_identity",
                side_effect=AssertionError("re-hashed an unchanged image"),
            ):
                second = machine_image_identity(image, label="fixture image")
            self.assertEqual(second, first)

    def test_rewriting_the_image_invalidates_the_cached_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "image.qcow2"
            image.write_bytes(b"image-bytes")
            first = machine_image_identity(image, label="fixture image")
            image.write_bytes(b"different-bytes")
            second = machine_image_identity(image, label="fixture image")
            self.assertNotEqual(second["sha256"], first["sha256"])
            self.assertEqual(
                second["sha256"], hashlib.sha256(b"different-bytes").hexdigest()
            )

    def test_a_changed_sidecar_is_not_answered_from_the_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "image.qcow2"
            image.write_bytes(b"image-bytes")
            digest = hashlib.sha256(b"image-bytes").hexdigest()
            sidecar = self._sidecar(image, digest)
            self.assertEqual(
                machine_image_identity(image, label="fixture image")[
                    "provenance_schema"
                ],
                "fixture-v1",
            )
            # The image is untouched; only its provenance moved.
            sidecar.unlink()
            sidecar.symlink_to(Path(temporary) / "missing.json")
            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                machine_image_identity(image, label="fixture image")

    def test_a_removed_sidecar_is_not_answered_from_the_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "image.qcow2"
            image.write_bytes(b"image-bytes")
            sidecar = self._sidecar(
                image, hashlib.sha256(b"image-bytes").hexdigest()
            )
            self.assertIn(
                "provenance", machine_image_identity(image, label="fixture image")
            )
            sidecar.unlink()
            self.assertNotIn(
                "provenance", machine_image_identity(image, label="fixture image")
            )


if __name__ == "__main__":
    unittest.main()
