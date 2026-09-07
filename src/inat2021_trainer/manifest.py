#!/usr/bin/env python3
"""Create and validate immutable identities for published ImageFolder datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torchvision

MANIFEST_SCHEMA = 2


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def class_mapping_identity(class_to_idx: dict[str, int]) -> str:
    encoded = json.dumps(class_to_idx, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def expected_dataset_contract(
    train_split: str,
    val_split: str,
    expected_train_samples: int,
    expected_validation_samples: int,
    expected_classes: int,
) -> dict[str, Any]:
    for name, value in (
        ("expected_train_samples", expected_train_samples),
        ("expected_validation_samples", expected_validation_samples),
        ("expected_classes", expected_classes),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    return {
        "train_split": train_split,
        "validation_split": val_split,
        "train_samples": expected_train_samples,
        "validation_samples": expected_validation_samples,
        "classes": expected_classes,
    }


def stable_file_identity(
    path: Path, chunk_size: int = 8 * 1024 * 1024
) -> tuple[int, str]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity:
        raise RuntimeError(f"dataset file changed while hashing: {path}")
    return after.st_size, digest.hexdigest()


def build_split_manifest_from_samples(
    root: Path, samples: Sequence[tuple[str, int]]
) -> dict[str, Any]:
    digest = hashlib.sha256()
    total_bytes = 0
    root = root.resolve()
    for path_value, target in samples:
        path = Path(path_value)
        relative_path = path.resolve().relative_to(root).as_posix()
        size, content_sha256 = stable_file_identity(path)
        record = f"{relative_path}\0{target}\0{size}\0{content_sha256}\n".encode()
        digest.update(record)
        total_bytes += size
    return {
        "sample_count": len(samples),
        "total_bytes": total_bytes,
        "content_identity": digest.hexdigest(),
    }


def build_split_manifest(root: Path) -> tuple[dict[str, Any], dict[str, int]]:
    dataset = torchvision.datasets.ImageFolder(str(root))
    return build_split_manifest_from_samples(
        root, dataset.samples
    ), dataset.class_to_idx


def atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    encoded = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def create_manifest(
    data_root: Path,
    train_split: str,
    val_split: str,
    expected_train_samples: int,
    expected_validation_samples: int,
    expected_classes: int,
) -> dict[str, Any]:
    if train_split == val_split:
        raise ValueError("train_split and val_split must differ")
    expectations = expected_dataset_contract(
        train_split,
        val_split,
        expected_train_samples,
        expected_validation_samples,
        expected_classes,
    )
    train, train_classes = build_split_manifest(data_root / train_split)
    validation, validation_classes = build_split_manifest(data_root / val_split)
    if train_classes != validation_classes:
        raise RuntimeError("train and validation class mappings differ")
    if train["sample_count"] != expected_train_samples:
        raise RuntimeError(
            f"expected {expected_train_samples} training samples, "
            f"found {train['sample_count']}"
        )
    if validation["sample_count"] != expected_validation_samples:
        raise RuntimeError(
            f"expected {expected_validation_samples} validation samples, "
            f"found {validation['sample_count']}"
        )
    if len(train_classes) != expected_classes:
        raise RuntimeError(
            f"expected {expected_classes} classes, found {len(train_classes)}"
        )
    return {
        "schema_version": MANIFEST_SCHEMA,
        "format": "sha256(relative-path\\0label\\0size\\0content-sha256\\n)",
        "expectations": expectations,
        "class_mapping_identity": class_mapping_identity(train_classes),
        "splits": {
            train_split: train,
            val_split: validation,
        },
    }


def _validate_manifest(
    manifest: dict[str, Any],
    expected_contract: dict[str, Any],
    train_samples: Sequence[tuple[str, int]],
    val_samples: Sequence[tuple[str, int]],
    class_to_idx: dict[str, int],
    val_class_to_idx: dict[str, int],
) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise RuntimeError("dataset manifest schema mismatch")
    if manifest.get("expectations") != expected_contract:
        raise RuntimeError("dataset manifest expectations mismatch")
    if (
        manifest.get("format")
        != "sha256(relative-path\\0label\\0size\\0content-sha256\\n)"
    ):
        raise RuntimeError("dataset manifest format mismatch")
    if class_to_idx != val_class_to_idx:
        raise RuntimeError("train and validation class mappings differ")
    if len(class_to_idx) != expected_contract["classes"]:
        raise RuntimeError("dataset manifest class count mismatch")
    if manifest.get("class_mapping_identity") != class_mapping_identity(class_to_idx):
        raise RuntimeError("dataset manifest class mapping mismatch")
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        raise TypeError("dataset manifest splits are missing")
    for split, samples, count in (
        (
            expected_contract["train_split"],
            train_samples,
            expected_contract["train_samples"],
        ),
        (
            expected_contract["validation_split"],
            val_samples,
            expected_contract["validation_samples"],
        ),
    ):
        expected = splits.get(split)
        if not isinstance(expected, dict):
            raise TypeError(f"dataset manifest split is missing: {split}")
        if len(samples) != count or expected.get("sample_count") != count:
            raise RuntimeError(f"dataset manifest sample_count mismatch for {split}")
        total_bytes = expected.get("total_bytes")
        if type(total_bytes) is not int or total_bytes < 0:
            raise RuntimeError(f"dataset manifest total_bytes is invalid for {split}")
        content_identity = expected.get("content_identity")
        if (
            not isinstance(content_identity, str)
            or len(content_identity) != 64
            or any(
                character not in "0123456789abcdef" for character in content_identity
            )
        ):
            raise RuntimeError(
                f"dataset manifest content_identity is invalid for {split}"
            )


def load_manifest_identity(
    path: Path,
    train_split: str,
    val_split: str,
    expected_train_samples: int,
    expected_validation_samples: int,
    expected_classes: int,
    train_samples: Sequence[tuple[str, int]],
    val_samples: Sequence[tuple[str, int]],
    class_to_idx: dict[str, int],
    val_class_to_idx: dict[str, int],
) -> str:
    """Check manifest/counts/classes without opening or statting image files.

    Callers must enumerate both ImageFolders and supply their samples and mappings.
    Trusts a previously verified, unchanged dataset: file paths, sizes and content
    are not rechecked, including same-size edits. Use verify_manifest for that.
    Returns SHA-256 of the original manifest bytes, preserving checkpoint identity.
    """
    if train_split == val_split:
        raise ValueError("train_split and val_split must differ")
    encoded = path.read_bytes()
    expectations = expected_dataset_contract(
        train_split,
        val_split,
        expected_train_samples,
        expected_validation_samples,
        expected_classes,
    )
    _validate_manifest(
        json.loads(encoded),
        expectations,
        train_samples,
        val_samples,
        class_to_idx,
        val_class_to_idx,
    )
    return hashlib.sha256(encoded).hexdigest()


def verify_manifest(
    path: Path,
    data_root: Path,
    train_split: str,
    val_split: str,
    expected_train_samples: int,
    expected_validation_samples: int,
    expected_classes: int,
) -> str:
    """Read every image and verify an existing manifest; never write or replace it.

    The caller must keep the dataset unchanged during verification and subsequent
    training. Per-file fstat guards detect changes while hashing, not a global
    snapshot. This verifies bytes, paths and labels, not JPEG decodability.
    Returns SHA-256 of the exact manifest bytes checked.
    """
    if train_split == val_split:
        raise ValueError("train_split and val_split must differ")
    expectations = expected_dataset_contract(
        train_split,
        val_split,
        expected_train_samples,
        expected_validation_samples,
        expected_classes,
    )
    encoded = path.read_bytes()
    manifest = json.loads(encoded)
    train = torchvision.datasets.ImageFolder(str(data_root / train_split))
    validation = torchvision.datasets.ImageFolder(str(data_root / val_split))
    _validate_manifest(
        manifest,
        expectations,
        train.samples,
        validation.samples,
        train.class_to_idx,
        validation.class_to_idx,
    )
    for split, samples in (
        (train_split, train.samples),
        (val_split, validation.samples),
    ):
        actual = build_split_manifest_from_samples(data_root / split, samples)
        for field in ("sample_count", "total_bytes", "content_identity"):
            if manifest["splits"][split][field] != actual[field]:
                raise RuntimeError(f"dataset manifest {field} mismatch for {split}")
    return hashlib.sha256(encoded).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--expected-train-samples", type=int, required=True)
    parser.add_argument("--expected-validation-samples", type=int, required=True)
    parser.add_argument("--expected-classes", type=int, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--output", type=Path, help="create a manifest by hashing every image"
    )
    action.add_argument(
        "--verify-existing",
        type=Path,
        help="read every image to verify this existing manifest; never overwrite it",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    if args.verify_existing is not None:
        path = args.verify_existing.resolve()
        identity = verify_manifest(
            path,
            data_root,
            args.train_split,
            args.val_split,
            args.expected_train_samples,
            args.expected_validation_samples,
            args.expected_classes,
        )
        print(
            json.dumps(
                {
                    "status": "verified",
                    "manifest": str(path),
                    "manifest_sha256": identity,
                },
                sort_keys=True,
            )
        )
        return 0
    output = args.output.resolve()
    manifest = create_manifest(
        data_root,
        args.train_split,
        args.val_split,
        args.expected_train_samples,
        args.expected_validation_samples,
        args.expected_classes,
    )
    atomic_json_write(output, manifest)
    print(
        json.dumps(
            {
                "status": "completed",
                "manifest": str(output),
                "splits": manifest["splits"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
