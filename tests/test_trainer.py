import argparse
import json
import os
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from inat2021_trainer import cli as trainer
from inat2021_trainer import manifest as dataset_manifest


class DownloadScriptTest(unittest.TestCase):
    def test_missing_storage_root_is_rejected_before_external_commands(self) -> None:
        environment = os.environ.copy()
        environment.pop("BASE", None)
        environment["PATH"] = ""
        result = subprocess.run(
            [
                "/bin/bash",
                str(Path(__file__).parents[1] / "scripts" / "download_inat2021.sh"),
                "mini",
            ],
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("BASE", result.stderr)

    def test_successful_short_transfer_is_retried_without_losing_partial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_wget = fake_bin / "wget"
            fake_wget.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "for argument in \"$@\"; do\n"
                "  case \"$argument\" in\n"
                "    --output-document=*) output=${argument#*=} ;;\n"
                "  esac\n"
                "done\n"
                "printf partial > \"$output\"\n",
                encoding="utf-8",
            )
            fake_wget.chmod(0o755)
            md5_marker = root / "md5-called"
            fake_md5sum = fake_bin / "md5sum"
            fake_md5sum.write_text(
                "#!/usr/bin/env bash\n"
                "touch \"$MD5_MARKER\"\n"
                "exit 99\n",
                encoding="utf-8",
            )
            fake_md5sum.chmod(0o755)
            base = root / "dataset"
            environment = os.environ.copy()
            environment.update(
                {
                    "BASE": str(base),
                    "DOWNLOAD_PROXY": "",
                    "HEARTBEAT_INTERVAL": "1",
                    "MAX_ATTEMPTS": "1",
                    "MAX_MD5_REDOWNLOADS": "0",
                    "MD5_MARKER": str(md5_marker),
                    "PATH": f"{fake_bin}:{environment['PATH']}",
                    "RETRY_BASE_DELAY": "0",
                    "RUN_ID": "short-success",
                }
            )

            result = subprocess.run(
                [
                    str(
                        Path(__file__).parents[1]
                        / "scripts"
                        / "download_inat2021.sh"
                    ),
                    "full",
                ],
                check=False,
                capture_output=True,
                env=environment,
                text=True,
            )

            partial = base / "archive" / "train.tar.gz.part"
            self.assertEqual(result.returncode, 75)
            self.assertEqual(partial.read_bytes(), b"partial")
            self.assertFalse(md5_marker.exists())
            self.assertEqual(list((base / "archive").glob("*.bad.*")), [])
            self.assertIn(
                "retryable-incomplete-success name=train.tar.gz "
                "expected_bytes=239909164970 preserved_bytes=7",
                result.stdout,
            )
            status = dict(
                line.split("=", 1)
                for line in (
                    base / "metadata" / "short-success" / "status"
                ).read_text(encoding="utf-8").splitlines()
            )
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["failure_reason"], "transfer-attempts-exhausted")
            self.assertEqual(status["downloaded_bytes"], "7")


class EpochPermutationSamplerTest(unittest.TestCase):
    def test_resume_starts_at_committed_cursor(self) -> None:
        full = list(trainer.EpochPermutationSampler(size=31, seed=17, epoch=4))
        resumed = list(
            trainer.EpochPermutationSampler(size=31, seed=17, epoch=4, start_index=13)
        )
        self.assertEqual(resumed, full[13:])
        self.assertEqual(len({index for index, _ in full}), 31)
        self.assertTrue(all(epoch == 4 for _, epoch in full))

    def test_epoch_changes_permutation(self) -> None:
        first = list(trainer.EpochPermutationSampler(size=31, seed=17, epoch=4))
        second = list(trainer.EpochPermutationSampler(size=31, seed=17, epoch=5))
        self.assertNotEqual(first, second)


class TimedImageFolderTest(unittest.TestCase):
    def test_augmentation_is_stable_for_sample_and_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "class-a").mkdir()
            pixels = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
            Image.fromarray(pixels).save(root / "class-a" / "sample.jpg")
            transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(16),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
            dataset = trainer.TimedImageFolder(root, transform, seed=123)
            first, first_target, _, _ = dataset[(0, 7)]
            second, second_target, _, _ = dataset[(0, 7)]
            self.assertEqual(first_target, second_target)
            self.assertTrue(torch.equal(first, second))

    def test_augmentation_seed_does_not_repeat_old_epoch_index_collision(self) -> None:
        first = trainer.deterministic_sample_seed(123, epoch=0, index=97_409)
        second = trainer.deterministic_sample_seed(123, epoch=1, index=0)
        self.assertNotEqual(first, second)
        self.assertEqual(
            first, trainer.deterministic_sample_seed(123, epoch=0, index=97_409)
        )


class OutputDirectoryTest(unittest.TestCase):
    def test_fresh_run_rejects_existing_training_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "train.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "existing training run"):
                trainer.claim_output_directory(output, resume="", run_id="fresh")
            lock = trainer.claim_output_directory(
                output, resume="latest", run_id="resume"
            )
            lock.close()

    def test_output_lock_rejects_concurrent_owner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            first = trainer.claim_output_directory(output, resume="", run_id="first")
            try:
                with self.assertRaisesRegex(RuntimeError, "owned by another trainer"):
                    trainer.claim_output_directory(
                        output, resume="latest", run_id="second"
                    )
            finally:
                first.close()

    def test_jsonl_events_always_include_logger_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.jsonl"
            logger = trainer.JsonlLogger(path, "run-expected")
            logger.write({"event": "test", "run_id": "run-spoofed"})
            logger.close()
            event = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(event["run_id"], "run-expected")

    def test_cleanup_removes_orphans_but_preserves_live_writer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_dir = Path(temporary)
            orphan = checkpoint_dir / "step-000000001.pt.tmp.interrupted"
            live = checkpoint_dir / f"step-000000002.pt.tmp.{os.getpid()}.active"
            orphan.write_bytes(b"orphan")
            live.write_bytes(b"live")
            removed = trainer.cleanup_checkpoint_temporaries(checkpoint_dir)
            self.assertEqual(removed, [str(orphan)])
            self.assertFalse(orphan.exists())
            self.assertTrue(live.exists())

    def test_atomic_json_failure_removes_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "value.json"
            with (
                mock.patch.object(os, "replace", side_effect=OSError("replace failed")),
                self.assertRaisesRegex(OSError, "replace failed"),
            ):
                trainer.atomic_json_write(path, {"value": 1})
            self.assertEqual(list(directory.glob("value.json.tmp.*")), [])

    def test_worker_closes_inherited_output_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            lock = trainer.claim_output_directory(output, resume="", run_id="parent")
            trainer._OUTPUT_LOCK = lock
            ready_read, ready_write = os.pipe()
            release_read, release_write = os.pipe()
            child_pid = os.fork()
            if child_pid == 0:
                os.close(ready_read)
                os.close(release_write)
                try:
                    trainer.seed_worker(0)
                    os.write(ready_write, b"1")
                    os.read(release_read, 1)
                finally:
                    os._exit(0)

            os.close(ready_write)
            os.close(release_read)
            try:
                self.assertEqual(os.read(ready_read, 1), b"1")
                lock.close()
                trainer._OUTPUT_LOCK = None
                contender = trainer.claim_output_directory(
                    output, resume="", run_id="contender"
                )
                contender.close()
            finally:
                os.write(release_write, b"1")
                os.close(release_write)
                os.close(ready_read)
                os.waitpid(child_pid, 0)
                if not lock.closed:
                    lock.close()
                trainer._OUTPUT_LOCK = None


class DatasetManifestTest(unittest.TestCase):
    @staticmethod
    def create_dataset(root: Path) -> None:
        for split, offset in (("train", 0), ("val", 10)):
            for class_name, class_offset in (("class-a", 0), ("class-b", 1)):
                directory = root / split / class_name
                directory.mkdir(parents=True)
                pixels = np.full((8, 8, 3), offset + class_offset, dtype=np.uint8)
                Image.fromarray(pixels).save(directory / "sample.jpg")

    def test_manifest_rejects_identical_train_and_validation_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.create_dataset(root)
            with self.assertRaisesRegex(ValueError, "must differ"):
                dataset_manifest.create_manifest(root, "train", "train", 2, 2, 2)

    def test_manifest_rejects_unexpected_sample_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.create_dataset(root)
            with self.assertRaisesRegex(RuntimeError, "training samples"):
                dataset_manifest.create_manifest(root, "train", "val", 3, 2, 2)
            with self.assertRaisesRegex(RuntimeError, "validation samples"):
                dataset_manifest.create_manifest(root, "train", "val", 2, 3, 2)
            with self.assertRaisesRegex(RuntimeError, "classes"):
                dataset_manifest.create_manifest(root, "train", "val", 2, 2, 3)

    def test_manifest_identity_changes_with_file_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.create_dataset(root)
            path = root / "manifest.json"
            first = dataset_manifest.create_manifest(root, "train", "val", 2, 2, 2)
            dataset_manifest.atomic_json_write(path, first)
            first_identity = dataset_manifest.sha256_file(path)

            pixels = np.full((8, 8, 3), 99, dtype=np.uint8)
            Image.fromarray(pixels).save(root / "train" / "class-a" / "sample.jpg")
            second = dataset_manifest.create_manifest(root, "train", "val", 2, 2, 2)
            dataset_manifest.atomic_json_write(path, second)
            self.assertNotEqual(first_identity, dataset_manifest.sha256_file(path))

    @staticmethod
    def load_identity(path: Path, root: Path, **overrides) -> str:
        train = trainer.TimedImageFolder(root / "train", None, seed=1)
        validation = trainer.TimedImageFolder(root / "val", None, seed=1)
        arguments = {
            "path": path,
            "train_split": "train",
            "val_split": "val",
            "expected_train_samples": 2,
            "expected_validation_samples": 2,
            "expected_classes": 2,
            "train_samples": train.samples,
            "val_samples": validation.samples,
            "class_to_idx": train.class_to_idx,
            "val_class_to_idx": validation.class_to_idx,
        }
        return dataset_manifest.load_manifest_identity(**(arguments | overrides))

    def test_full_verification_detects_same_size_tampering_without_rewriting(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.create_dataset(root)
            path = root / "manifest.json"
            value = dataset_manifest.create_manifest(root, "train", "val", 2, 2, 2)
            dataset_manifest.atomic_json_write(path, value)
            original = path.read_bytes()
            identity = dataset_manifest.verify_manifest(
                path, root, "train", "val", 2, 2, 2
            )
            self.assertEqual(identity, dataset_manifest.sha256_file(path))

            changed = root / "train" / "class-a" / "sample.jpg"
            content = bytearray(changed.read_bytes())
            content[len(content) // 2] ^= 1
            changed.write_bytes(content)
            self.assertEqual(self.load_identity(path, root), identity)
            with self.assertRaisesRegex(RuntimeError, "content_identity mismatch"):
                dataset_manifest.verify_manifest(path, root, "train", "val", 2, 2, 2)
            self.assertEqual(path.read_bytes(), original)

    def test_training_startup_does_not_open_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.create_dataset(root)
            path = root / "manifest.json"
            dataset_manifest.atomic_json_write(
                path, dataset_manifest.create_manifest(root, "train", "val", 2, 2, 2)
            )
            expected_identity = dataset_manifest.sha256_file(path)
            original_open = open

            def guarded_open(file, *args, **kwargs):
                if isinstance(file, (str, os.PathLike)) and Path(file).suffix == ".jpg":
                    raise AssertionError(f"startup opened image content: {file}")
                return original_open(file, *args, **kwargs)

            def guarded_path_open(file, *args, **kwargs):
                return guarded_open(file, *args, **kwargs)

            args = ArgumentValidationTest.valid_args()
            args.data_root = root
            args.dataset_manifest = path
            args.output_dir = root / "output"
            args.resume = ""
            args.deterministic = False
            args.seed = 1
            args.model = "resnet50"
            args.weights = "none"
            with (
                mock.patch("builtins.open", side_effect=guarded_open),
                mock.patch.object(Path, "open", guarded_path_open),
                mock.patch.object(trainer, "parse_args", return_value=args),
                mock.patch.object(
                    trainer.torch.cuda, "is_available", return_value=True
                ),
                mock.patch.object(
                    trainer,
                    "build_model",
                    side_effect=RuntimeError("reached model setup"),
                ),
            ):
                self.assertEqual(self.load_identity(path, root), expected_identity)
                with self.assertRaisesRegex(RuntimeError, "reached model setup"):
                    trainer.main()

    def test_lightweight_checks_reject_manifest_and_dataset_contract_errors(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.create_dataset(root)
            path = root / "manifest.json"
            value = dataset_manifest.create_manifest(root, "train", "val", 2, 2, 2)
            cases = [
                (("schema_version",), 1, "schema mismatch"),
                (("format",), "other", "format mismatch"),
                (("expectations", "train_samples"), 3, "expectations mismatch"),
                (("class_mapping_identity",), "0" * 64, "class mapping mismatch"),
                (("splits",), None, "splits are missing"),
                (("splits", "val"), None, "split is missing"),
                (("splits", "train", "sample_count"), 3, "sample_count mismatch"),
                (("splits", "train", "total_bytes"), -1, "total_bytes is invalid"),
                (
                    ("splits", "val", "content_identity"),
                    "invalid",
                    "content_identity is invalid",
                ),
            ]
            for keys, replacement, error in cases:
                with self.subTest(keys=keys):
                    changed = json.loads(json.dumps(value))
                    target = changed
                    for key in keys[:-1]:
                        target = target[key]
                    target[keys[-1]] = replacement
                    dataset_manifest.atomic_json_write(path, changed)
                    with self.assertRaisesRegex((RuntimeError, TypeError), error):
                        self.load_identity(path, root)
                    with self.assertRaisesRegex((RuntimeError, TypeError), error):
                        dataset_manifest.verify_manifest(
                            path, root, "train", "val", 2, 2, 2
                        )

            dataset_manifest.atomic_json_write(path, value)
            with self.assertRaisesRegex(RuntimeError, "expectations mismatch"):
                self.load_identity(path, root, expected_train_samples=3)
            with self.assertRaisesRegex(RuntimeError, "class count mismatch"):
                self.load_identity(
                    path,
                    root,
                    class_to_idx={"class-a": 0},
                    val_class_to_idx={"class-a": 0},
                )
            for split in ("train", "val"):
                with self.subTest(extra_sample=split):
                    extra = root / split / "class-a" / "extra.jpg"
                    extra.write_bytes(b"not decoded during startup")
                    with self.assertRaisesRegex(RuntimeError, "sample_count mismatch"):
                        self.load_identity(path, root)
                    extra.rename(root / f"{split}-extra.jpg")
            (root / "val" / "class-a").rename(root / "val" / "class-c")
            with self.assertRaisesRegex(RuntimeError, "class mappings differ"):
                self.load_identity(path, root)
            with self.assertRaisesRegex(RuntimeError, "class mappings differ"):
                dataset_manifest.verify_manifest(path, root, "train", "val", 2, 2, 2)

    def test_manifest_byte_identity_remains_bound_to_checkpoint_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.create_dataset(root)
            path = root / "manifest.json"
            dataset_manifest.atomic_json_write(
                path, dataset_manifest.create_manifest(root, "train", "val", 2, 2, 2)
            )
            original = path.read_bytes()
            identity = self.load_identity(path, root)
            self.assertEqual(identity, dataset_manifest.sha256_file(path))
            config = trainer.config_identity({"dataset_identity": identity})
            checkpoints = root / "checkpoints"
            trainer.save_checkpoint_atomic(
                checkpoints,
                AtomicCheckpointTest.checkpoint(1, config, identity),
                global_step=1,
                expected_config_identity=config,
                expected_dataset_identity=identity,
            )
            self.assertEqual(
                trainer.load_checkpoint_strict("latest", checkpoints, config, identity)[
                    "global_step"
                ],
                1,
            )
            # Even a whitespace-only change retains the pre-existing byte identity contract.
            path.write_bytes(original + b"\n")
            changed_identity = self.load_identity(path, root)
            changed_config = trainer.config_identity(
                {"dataset_identity": changed_identity}
            )
            with self.assertRaisesRegex(RuntimeError, "config identity mismatch"):
                trainer.load_checkpoint_strict(
                    "latest", checkpoints, changed_config, changed_identity
                )
            with self.assertRaisesRegex(RuntimeError, "dataset identity mismatch"):
                trainer.load_checkpoint_strict(
                    "latest", checkpoints, config, changed_identity
                )


class ArgumentValidationTest(unittest.TestCase):
    @staticmethod
    def valid_args() -> argparse.Namespace:
        return argparse.Namespace(
            train_split="train",
            val_split="val",
            batch_size=1,
            workers=0,
            max_steps=2,
            warmup_steps=1,
            checkpoint_every=1,
            validation_batches=1,
            log_every=1,
            fault_sigkill_after_step=0,
            expected_classes=2,
            expected_train_samples=2,
            expected_validation_samples=2,
            train_crop_size=8,
            val_resize_size=8,
            val_crop_size=8,
            learning_rate=0.1,
            momentum=0.9,
            weight_decay=0.0,
        )

    def test_negative_validation_batches_are_rejected(self) -> None:
        args = self.valid_args()
        args.validation_batches = -1
        with self.assertRaisesRegex(ValueError, "validation_batches"):
            trainer.validate_args(args)

    def test_zero_log_interval_is_rejected(self) -> None:
        args = self.valid_args()
        args.log_every = 0
        with self.assertRaisesRegex(ValueError, "log_every"):
            trainer.validate_args(args)

    def test_identical_train_and_validation_splits_are_rejected(self) -> None:
        args = self.valid_args()
        args.val_split = args.train_split
        with self.assertRaisesRegex(ValueError, "must differ"):
            trainer.validate_args(args)

    def test_invalid_image_and_optimizer_parameters_are_rejected(self) -> None:
        invalid = {
            "expected_classes": 0,
            "expected_train_samples": 0,
            "expected_validation_samples": 0,
            "train_crop_size": 0,
            "val_resize_size": 0,
            "val_crop_size": 0,
            "learning_rate": float("nan"),
            "momentum": float("inf"),
            "weight_decay": -0.1,
        }
        for name, value in invalid.items():
            with self.subTest(name=name):
                args = self.valid_args()
                setattr(args, name, value)
                with self.assertRaisesRegex(ValueError, name):
                    trainer.validate_args(args)


class FailureEventTest(unittest.TestCase):
    def test_main_records_catchable_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.jsonl"

            def fail() -> int:
                trainer._ACTIVE_LOGGER = trainer.JsonlLogger(path, "failed-run")
                trainer._FAILURE_STATE = {
                    "phase": "validation",
                    "global_step": 20,
                    "checkpoint": "step-20.pt",
                }
                raise RuntimeError("validation failed")

            with (
                mock.patch.object(trainer, "_main", side_effect=fail),
                self.assertRaisesRegex(RuntimeError, "validation failed"),
            ):
                trainer.main()
            event = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(event["event"], "failed")
            self.assertEqual(event["phase"], "validation")
            self.assertEqual(event["global_step"], 20)
            self.assertEqual(event["error_type"], "RuntimeError")

    def test_owned_startup_failure_records_terminal_event(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = ArgumentValidationTest.valid_args()
            args.data_root = root / "data"
            args.dataset_manifest = root / "manifest.json"
            args.output_dir = root / "output"
            args.train_split = "train"
            args.val_split = "val"
            args.resume = ""
            args.deterministic = False
            args.seed = 1

            with (
                mock.patch.object(trainer, "parse_args", return_value=args),
                mock.patch.object(trainer, "configure_determinism"),
                mock.patch.object(
                    trainer.torch.cuda, "is_available", return_value=True
                ),
                mock.patch.object(trainer.torch.cuda, "manual_seed_all"),
                mock.patch.object(
                    trainer,
                    "TimedImageFolder",
                    side_effect=RuntimeError("dataset startup failed"),
                ),
                self.assertRaisesRegex(RuntimeError, "dataset startup failed"),
            ):
                trainer.main()

            event = json.loads(
                (args.output_dir / "train.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(event["event"], "failed")
            self.assertEqual(event["phase"], "startup")
            self.assertEqual(event["global_step"], 0)
            self.assertEqual(event["error_type"], "RuntimeError")


class DeterminismTest(unittest.TestCase):
    def test_deterministic_mode_is_strict_and_configures_cublas(self) -> None:
        previous_workspace = os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        previous_algorithms = torch.are_deterministic_algorithms_enabled()
        previous_benchmark = torch.backends.cudnn.benchmark
        previous_deterministic = torch.backends.cudnn.deterministic
        try:
            trainer.configure_determinism(True)
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertTrue(torch.backends.cudnn.deterministic)
            self.assertFalse(torch.backends.cudnn.benchmark)
            self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        finally:
            torch.use_deterministic_algorithms(previous_algorithms)
            torch.backends.cudnn.benchmark = previous_benchmark
            torch.backends.cudnn.deterministic = previous_deterministic
            if previous_workspace is None:
                os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            else:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous_workspace


class ValidationAndTelemetryTest(unittest.TestCase):
    def test_validation_subset_never_exposes_an_extra_batch(self) -> None:
        dataset = list(range(10))
        bounded = trainer.bounded_validation_dataset(
            dataset, batch_size=3, max_batches=2
        )
        self.assertEqual(list(bounded), list(range(6)))
        self.assertIs(
            trainer.bounded_validation_dataset(dataset, batch_size=3, max_batches=0),
            dataset,
        )

    def test_step_throughput_includes_dataloader_wait(self) -> None:
        metrics = trainer.step_throughput_metrics(
            batch_size=256,
            dataloader_wait_ms=400.0,
            batch_total_ms=100.0,
        )
        self.assertEqual(metrics["end_to_end_batch_ms"], 500.0)
        self.assertEqual(metrics["end_to_end_images_per_second"], 512.0)
        self.assertEqual(metrics["gpu_path_images_per_second"], 2560.0)

    def test_gpu_snapshot_uses_first_visible_device(self) -> None:
        result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="7, 42\n", stderr=""
        )
        with (
            mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3,1"}),
            mock.patch.object(trainer.subprocess, "run", return_value=result) as run,
        ):
            snapshot = trainer.gpu_snapshot()
        self.assertEqual(snapshot["gpu_utilization_pct"], 7)
        self.assertEqual(snapshot["gpu_memory_used_mib"], 42)
        self.assertIn("--id=3", run.call_args.args[0])


class AtomicCheckpointTest(unittest.TestCase):
    @staticmethod
    def checkpoint(step: int, config: str, dataset: str) -> dict:
        return {
            "schema_version": trainer.CHECKPOINT_SCHEMA,
            "global_step": step,
            "config_identity": config,
            "dataset_identity": dataset,
            "payload": torch.arange(8),
        }

    def test_latest_selects_only_published_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = "config-id"
            dataset = "dataset-id"
            metrics = trainer.save_checkpoint_atomic(
                directory,
                self.checkpoint(20, config, dataset),
                global_step=20,
                expected_config_identity=config,
                expected_dataset_identity=dataset,
            )
            (directory / "step-000000021.pt.tmp.interrupted").write_bytes(b"partial")
            loaded = trainer.load_checkpoint_strict(
                "latest", directory, config, dataset
            )
            self.assertEqual(loaded["global_step"], 20)
            self.assertEqual(loaded["resolved_path"], metrics["checkpoint_path"])

    def test_resume_adopts_checkpoint_published_before_latest_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = "config-id"
            dataset = "dataset-id"
            trainer.save_checkpoint_atomic(
                directory,
                self.checkpoint(20, config, dataset),
                global_step=20,
                expected_config_identity=config,
                expected_dataset_identity=dataset,
            )
            failure_state = {
                "phase": "training",
                "global_step": 20,
                "checkpoint": str(directory / "step-000000020.pt"),
            }
            with (
                mock.patch.object(
                    trainer,
                    "atomic_json_write",
                    side_effect=OSError("latest publish failed"),
                ),
                mock.patch.object(trainer, "_FAILURE_STATE", failure_state),
                self.assertRaisesRegex(OSError, "latest publish failed"),
            ):
                trainer.save_checkpoint_atomic(
                    directory,
                    self.checkpoint(21, config, dataset),
                    global_step=21,
                    expected_config_identity=config,
                    expected_dataset_identity=dataset,
                )
            self.assertEqual(failure_state["global_step"], 21)
            self.assertEqual(
                failure_state["checkpoint"],
                str(directory / "step-000000021.pt"),
            )

            loaded = trainer.load_checkpoint_strict(
                "latest", directory, config, dataset
            )
            latest = json.loads((directory / "latest.json").read_text(encoding="utf-8"))
            self.assertEqual(loaded["global_step"], 21)
            self.assertEqual(latest["global_step"], 21)
            self.assertEqual(latest["checkpoint"], "step-000000021.pt")

    def test_resume_adopts_first_checkpoint_when_latest_pointer_is_missing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = "config-id"
            dataset = "dataset-id"
            with (
                mock.patch.object(
                    trainer,
                    "atomic_json_write",
                    side_effect=OSError("latest publish failed"),
                ),
                self.assertRaisesRegex(OSError, "latest publish failed"),
            ):
                trainer.save_checkpoint_atomic(
                    directory,
                    self.checkpoint(1, config, dataset),
                    global_step=1,
                    expected_config_identity=config,
                    expected_dataset_identity=dataset,
                )

            loaded = trainer.load_checkpoint_strict(
                "latest", directory, config, dataset
            )
            self.assertEqual(loaded["global_step"], 1)
            self.assertTrue((directory / "latest.json").is_file())

    def test_checksum_mismatch_blocks_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = "config-id"
            dataset = "dataset-id"
            metrics = trainer.save_checkpoint_atomic(
                directory,
                self.checkpoint(20, config, dataset),
                global_step=20,
                expected_config_identity=config,
                expected_dataset_identity=dataset,
            )
            with Path(str(metrics["checkpoint_path"])).open("ab") as handle:
                handle.write(b"corruption")
            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                trainer.load_checkpoint_strict("latest", directory, config, dataset)

    def test_identity_mismatch_blocks_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            trainer.save_checkpoint_atomic(
                directory,
                self.checkpoint(20, "config-id", "dataset-id"),
                global_step=20,
                expected_config_identity="config-id",
                expected_dataset_identity="dataset-id",
            )
            with self.assertRaisesRegex(RuntimeError, "config identity mismatch"):
                trainer.load_checkpoint_strict(
                    "latest", directory, "other-config", "dataset-id"
                )

    def test_checkpoint_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            checkpoint = self.checkpoint(20, "config-id", "dataset-id")
            trainer.save_checkpoint_atomic(
                directory,
                checkpoint,
                global_step=20,
                expected_config_identity="config-id",
                expected_dataset_identity="dataset-id",
            )
            with self.assertRaisesRegex(RuntimeError, "will not be overwritten"):
                trainer.save_checkpoint_atomic(
                    directory,
                    checkpoint,
                    global_step=20,
                    expected_config_identity="config-id",
                    expected_dataset_identity="dataset-id",
                )

    def test_only_latest_resume_is_supported(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            self.assertRaisesRegex(ValueError, "only --resume latest"),
        ):
            trainer.resolve_resume_checkpoint("step-000000020.pt", Path(temporary))

    def test_resumed_final_checkpoint_satisfies_completion_guard(self) -> None:
        trainer.require_final_checkpoint(global_step=20, last_checkpoint_step=20)
        with self.assertRaisesRegex(RuntimeError, "final step was not checkpointed"):
            trainer.require_final_checkpoint(global_step=20, last_checkpoint_step=-1)


class DownloadResumeTest(unittest.TestCase):
    def test_invalid_heartbeat_interval_is_rejected_before_side_effects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_root = Path(__file__).resolve().parents[1]
            base = root / "data"
            result = subprocess.run(
                [
                    "bash",
                    str(project_root / "scripts" / "download_inat2021.sh"),
                    "mini",
                ],
                env={
                    **os.environ,
                    "BASE": str(base),
                    "HEARTBEAT_INTERVAL": "0",
                },
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 64)
            self.assertIn("HEARTBEAT_INTERVAL", result.stderr)
            self.assertFalse(base.exists())

    def test_unsafe_run_id_is_rejected_before_side_effects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_root = Path(__file__).resolve().parents[1]
            base = root / "data"
            result = subprocess.run(
                [
                    "bash",
                    str(project_root / "scripts" / "download_inat2021.sh"),
                    "mini",
                ],
                env={
                    **os.environ,
                    "BASE": str(base),
                    "RUN_ID": "../../escaped\nstatus=passed",
                },
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 64)
            self.assertIn("RUN_ID", result.stderr)
            self.assertFalse(base.exists())

    def test_failed_attempts_preserve_partial_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mock_bin = root / "bin"
            mock_bin.mkdir()
            mock_wget = mock_bin / "wget"
            mock_wget.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "output=\n"
                'for argument in "$@"; do\n'
                '  case "$argument" in\n'
                "    --output-document=*) output=${argument#*=} ;;\n"
                "  esac\n"
                "done\n"
                'test -n "$output"\n'
                'printf x >> "$output"\n'
                "exit 4\n",
                encoding="utf-8",
            )
            mock_wget.chmod(0o755)
            project_root = Path(__file__).resolve().parents[1]
            base = root / "data"
            environment = {
                **os.environ,
                "PATH": f"{mock_bin}:{os.environ['PATH']}",
                "BASE": str(base),
                "RUN_ID": "resume-test",
                "MAX_ATTEMPTS": "2",
                "RETRY_BASE_DELAY": "0",
            }
            result = subprocess.run(
                [
                    "bash",
                    str(project_root / "scripts" / "download_inat2021.sh"),
                    "mini",
                ],
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            partial = base / "archive" / "train_mini.tar.gz.part"
            status = (base / "metadata" / "resume-test" / "status").read_text(
                encoding="utf-8"
            )
            self.assertEqual(result.returncode, 75)
            self.assertEqual(partial.read_bytes(), b"xx")
            self.assertIn("resume_bytes=0", result.stdout)
            self.assertIn("resume_bytes=1", result.stdout)
            self.assertIn("status=failed", status)
            self.assertIn("downloaded_bytes=2", status)

    def test_permanent_http_error_is_not_retried(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mock_bin = root / "bin"
            mock_bin.mkdir()
            mock_wget = mock_bin / "wget"
            mock_wget.write_text(
                "#!/usr/bin/env bash\n"
                "echo '  HTTP/1.1 407 Proxy Authentication Required' >&2\n"
                "exit 8\n",
                encoding="utf-8",
            )
            mock_wget.chmod(0o755)
            project_root = Path(__file__).resolve().parents[1]
            base = root / "data"
            result = subprocess.run(
                [
                    "bash",
                    str(project_root / "scripts" / "download_inat2021.sh"),
                    "mini",
                ],
                env={
                    **os.environ,
                    "PATH": f"{mock_bin}:{os.environ['PATH']}",
                    "BASE": str(base),
                    "RUN_ID": "permanent-test",
                    "MAX_ATTEMPTS": "3",
                    "RETRY_BASE_DELAY": "0",
                },
                capture_output=True,
                text=True,
                check=False,
            )
            status = (base / "metadata" / "permanent-test" / "status").read_text(
                encoding="utf-8"
            )
            self.assertEqual(result.returncode, 69)
            self.assertEqual(result.stdout.count("[DOWNLOAD] attempt="), 1)
            self.assertIn("permanent-http-failure", result.stderr)
            self.assertIn("failure_reason=permanent-http", status)

    def test_md5_mismatch_quarantines_and_redownloads_from_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mock_bin = root / "bin"
            mock_bin.mkdir()
            mock_wget = mock_bin / "wget"
            mock_wget.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "output=\n"
                'for argument in "$@"; do\n'
                '  case "$argument" in\n'
                "    --output-document=*) output=${argument#*=} ;;\n"
                "  esac\n"
                "done\n"
                'stat -c %s "$output" >> "$TRANSFER_STARTS"\n'
                'printf corrupt > "$output"\n'
                'truncate -s 44636137542 "$output"\n',
                encoding="utf-8",
            )
            mock_md5 = mock_bin / "md5sum"
            mock_md5.write_text(
                "#!/usr/bin/env bash\ncat >/dev/null\nexit 1\n", encoding="utf-8"
            )
            mock_wget.chmod(0o755)
            mock_md5.chmod(0o755)
            project_root = Path(__file__).resolve().parents[1]
            base = root / "data"
            result = subprocess.run(
                [
                    "bash",
                    str(project_root / "scripts" / "download_inat2021.sh"),
                    "mini",
                ],
                env={
                    **os.environ,
                    "PATH": f"{mock_bin}:{os.environ['PATH']}",
                    "BASE": str(base),
                    "RUN_ID": "md5-test",
                    "TRANSFER_STARTS": str(root / "transfer-starts"),
                    "MAX_ATTEMPTS": "1",
                    "MAX_MD5_REDOWNLOADS": "1",
                    "RETRY_BASE_DELAY": "0",
                },
                capture_output=True,
                text=True,
                check=False,
            )
            quarantined = list((base / "archive").glob("train_mini.tar.gz.part.bad.*"))
            self.assertEqual(result.returncode, 74)
            self.assertEqual(len(quarantined), 2)
            self.assertEqual(
                (root / "transfer-starts").read_text().splitlines(), ["0", "0"]
            )
            for path in quarantined:
                self.assertEqual(path.stat().st_size, 44636137542)
                with path.open("rb") as handle:
                    self.assertEqual(handle.read(7), b"corrupt")

    def test_passed_state_retains_last_completed_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mock_bin = root / "bin"
            mock_bin.mkdir()
            mock_wget = mock_bin / "wget"
            mock_wget.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "output=\n"
                'for argument in "$@"; do\n'
                '  case "$argument" in\n'
                "    --output-document=*) output=${argument#*=} ;;\n"
                "  esac\n"
                "done\n"
                'case "${output##*/}" in\n'
                "  train_mini.tar.gz.part) size=44636137542 ;;\n"
                "  val.tar.gz.part) size=8931661582 ;;\n"
                "  *) exit 64 ;;\n"
                "esac\n"
                'truncate -s "$size" "$output"\n',
                encoding="utf-8",
            )
            mock_md5 = mock_bin / "md5sum"
            mock_md5.write_text(
                "#!/usr/bin/env bash\ncat >/dev/null\nexit 0\n", encoding="utf-8"
            )
            mock_wget.chmod(0o755)
            mock_md5.chmod(0o755)
            project_root = Path(__file__).resolve().parents[1]
            base = root / "data"
            result = subprocess.run(
                [
                    "bash",
                    str(project_root / "scripts" / "download_inat2021.sh"),
                    "mini",
                ],
                env={
                    **os.environ,
                    "PATH": f"{mock_bin}:{os.environ['PATH']}",
                    "BASE": str(base),
                    "RUN_ID": "passed-test",
                    "MAX_ATTEMPTS": "1",
                },
                capture_output=True,
                text=True,
                check=False,
            )
            status = dict(
                line.split("=", 1)
                for line in (base / "metadata" / "passed-test" / "status")
                .read_text(encoding="utf-8")
                .splitlines()
            )
            self.assertEqual(result.returncode, 0)
            self.assertEqual(status["status"], "passed")
            self.assertEqual(status["last_completed_file"], "val.tar.gz")
            self.assertEqual(int(status["last_completed_bytes"]), 8931661582)
            self.assertEqual(
                (base / "archive" / "val.tar.gz").stat().st_size,
                int(status["last_completed_bytes"]),
            )
            self.assertEqual(
                status["last_completed_md5"], "f6f6e0e242e3d4c9569ba56400938afc"
            )

    def test_range_416_promotes_a_complete_partial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mock_bin = root / "bin"
            mock_bin.mkdir()
            mock_wget = mock_bin / "wget"
            mock_wget.write_text(
                "#!/usr/bin/env bash\n"
                "echo '  HTTP/1.1 416 Range Not Satisfiable' >&2\n"
                "exit 8\n",
                encoding="utf-8",
            )
            mock_md5 = mock_bin / "md5sum"
            mock_md5.write_text(
                "#!/usr/bin/env bash\ncat >/dev/null\nexit 0\n", encoding="utf-8"
            )
            mock_wget.chmod(0o755)
            mock_md5.chmod(0o755)
            project_root = Path(__file__).resolve().parents[1]
            base = root / "data"
            archive = base / "archive"
            archive.mkdir(parents=True)
            (archive / "train_mini.tar.gz.part").write_bytes(b"complete")
            result = subprocess.run(
                [
                    "bash",
                    str(project_root / "scripts" / "download_inat2021.sh"),
                    "mini",
                ],
                env={
                    **os.environ,
                    "PATH": f"{mock_bin}:{os.environ['PATH']}",
                    "BASE": str(base),
                    "RUN_ID": "range-test",
                    "MAX_ATTEMPTS": "1",
                },
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0)
            self.assertTrue((archive / "train_mini.tar.gz").is_file())
            self.assertIn(
                "complete-partial-verified name=train_mini.tar.gz", result.stdout
            )

    def test_reused_run_id_keeps_each_attempt_log(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mock_bin = root / "bin"
            mock_bin.mkdir()
            mock_wget = mock_bin / "wget"
            mock_wget.write_text(
                "#!/usr/bin/env bash\necho '  HTTP/1.1 404 Not Found' >&2\nexit 8\n",
                encoding="utf-8",
            )
            mock_wget.chmod(0o755)
            project_root = Path(__file__).resolve().parents[1]
            base = root / "data"
            environment = {
                **os.environ,
                "PATH": f"{mock_bin}:{os.environ['PATH']}",
                "BASE": str(base),
                "RUN_ID": "reused-run",
                "MAX_ATTEMPTS": "1",
            }
            for _ in range(2):
                result = subprocess.run(
                    [
                        "bash",
                        str(project_root / "scripts" / "download_inat2021.sh"),
                        "mini",
                    ],
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 69)
            logs = list((base / "metadata" / "reused-run").glob("wget-*.log"))
            self.assertEqual(len(logs), 2)

    def test_sigkill_during_watchdog_launch_is_locked_then_stopped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mock_bin = root / "bin"
            mock_bin.mkdir()
            pid_path = root / "wget.pid"
            bash_env = root / "bash-env"
            bash_env.write_text(
                "set -T\n"
                "trap 'if [[ ${FUNCNAME[0]:-} == supervise_active_child "
                "&& $BASH_COMMAND == wait_for_active_child ]]; then "
                'printf "%s\\\\n" "$BASHPID" > "$MOCK_SUPERVISOR_PID_FILE"; '
                'kill -STOP "$BASHPID"; trap - DEBUG; fi\' DEBUG\n',
                encoding="utf-8",
            )
            supervisor_pid_path = root / "supervisor.pid"
            supervisor_pid = None
            mock_wget = mock_bin / "wget"
            mock_wget.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "output=\n"
                'for argument in "$@"; do\n'
                '  case "$argument" in\n'
                "    --output-document=*) output=${argument#*=} ;;\n"
                "  esac\n"
                "done\n"
                'echo $$ > "$MOCK_PID_FILE"\n'
                "trap '' TERM INT HUP\n"
                'while true; do printf x >> "$output"; sleep 0.05; done\n',
                encoding="utf-8",
            )
            mock_wget.chmod(0o755)
            project_root = Path(__file__).resolve().parents[1]
            base = root / "data"
            environment = {
                **os.environ,
                "PATH": f"{mock_bin}:{os.environ['PATH']}",
                "BASE": str(base),
                "RUN_ID": "lock-owner",
                "MOCK_PID_FILE": str(pid_path),
                "BASH_ENV": str(bash_env),
                "MOCK_SUPERVISOR_PID_FILE": str(supervisor_pid_path),
                "HEARTBEAT_INTERVAL": "1",
            }
            first = subprocess.Popen(
                [
                    "bash",
                    str(project_root / "scripts" / "download_inat2021.sh"),
                    "mini",
                ],
                env=environment,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                for _ in range(100):
                    if pid_path.exists() and supervisor_pid_path.exists():
                        break
                    time.sleep(0.02)
                self.assertTrue(pid_path.exists())
                self.assertTrue(supervisor_pid_path.exists())
                wget_pid = int(pid_path.read_text(encoding="utf-8"))
                supervisor_pid = int(supervisor_pid_path.read_text(encoding="utf-8"))
                direct_children = Path(
                    f"/proc/{supervisor_pid}/task/{supervisor_pid}/children"
                ).read_text(encoding="utf-8")
                self.assertEqual(direct_children.split(), [str(wget_pid)])
                os.kill(first.pid, signal.SIGKILL)
                first.wait(timeout=3)
                os.kill(wget_pid, 0)
                second = subprocess.run(
                    [
                        "bash",
                        str(project_root / "scripts" / "download_inat2021.sh"),
                        "mini",
                    ],
                    env={**environment, "RUN_ID": "lock-contender"},
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(second.returncode, 73)
                self.assertIn("lock-busy", second.stderr)
                os.kill(supervisor_pid, signal.SIGCONT)
                child_alive = True
                for _ in range(200):
                    try:
                        os.kill(wget_pid, 0)
                    except ProcessLookupError:
                        child_alive = False
                        break
                    time.sleep(0.02)
                self.assertFalse(child_alive)
            finally:
                if first.poll() is None:
                    first.kill()
                    first.wait()
                if supervisor_pid is not None:
                    try:
                        os.kill(supervisor_pid, signal.SIGCONT)
                        os.kill(supervisor_pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                if pid_path.exists():
                    wget_pid = int(pid_path.read_text(encoding="utf-8"))
                    try:
                        os.kill(wget_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_catchable_signals_during_child_pid_assignment_kill_child(self) -> None:
        for delivered_signal, expected_rc in (
            (signal.SIGHUP, 129),
            (signal.SIGINT, 130),
            (signal.SIGTERM, 143),
        ):
            with (
                self.subTest(signal=delivered_signal),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                mock_bin = root / "bin"
                mock_bin.mkdir()
                pid_path = root / "wget.pid"
                supervisor_pid_path = root / "supervisor.pid"
                bash_env = root / "bash-env"
                bash_env.write_text(
                    "set -T\n"
                    "trap 'if [[ ${FUNCNAME[0]:-} == run_active_child "
                    "&& $BASH_COMMAND == ACTIVE_CHILD_PID=* ]]; then "
                    'printf "%s\\\\n" "$BASHPID" > '
                    '"$MOCK_SUPERVISOR_PID_FILE"; '
                    'kill -STOP "$BASHPID"; trap - DEBUG; fi\' DEBUG\n',
                    encoding="utf-8",
                )
                mock_wget = mock_bin / "wget"
                mock_wget.write_text(
                    "#!/usr/bin/env bash\n"
                    "set -euo pipefail\n"
                    'echo $$ > "$MOCK_PID_FILE"\n'
                    "trap '' TERM INT HUP\n"
                    "while true; do sleep 0.05; done\n",
                    encoding="utf-8",
                )
                mock_wget.chmod(0o755)
                project_root = Path(__file__).resolve().parents[1]
                base = root / "data"
                process = subprocess.Popen(
                    [
                        "bash",
                        str(project_root / "scripts" / "download_inat2021.sh"),
                        "mini",
                    ],
                    env={
                        **os.environ,
                        "PATH": f"{mock_bin}:{os.environ['PATH']}",
                        "BASE": str(base),
                        "RUN_ID": f"signal-{expected_rc}",
                        "MOCK_PID_FILE": str(pid_path),
                        "MOCK_SUPERVISOR_PID_FILE": str(supervisor_pid_path),
                        "BASH_ENV": str(bash_env),
                        "HEARTBEAT_INTERVAL": "1",
                        "CHILD_STOP_ATTEMPTS": "1",
                    },
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                supervisor_pid = None
                try:
                    for _ in range(100):
                        if pid_path.exists() and supervisor_pid_path.exists():
                            break
                        time.sleep(0.02)
                    self.assertTrue(pid_path.exists())
                    self.assertTrue(supervisor_pid_path.exists())
                    wget_pid = int(pid_path.read_text(encoding="utf-8"))
                    stopped_parent_pid = int(
                        supervisor_pid_path.read_text(encoding="utf-8")
                    )
                    self.assertEqual(stopped_parent_pid, process.pid)
                    direct_children = Path(
                        f"/proc/{process.pid}/task/{process.pid}/children"
                    ).read_text(encoding="utf-8")
                    self.assertEqual(len(direct_children.split()), 1)
                    supervisor_pid = int(direct_children)

                    process.send_signal(delivered_signal)
                    os.kill(process.pid, signal.SIGCONT)
                    self.assertEqual(process.wait(timeout=6), expected_rc)
                    with self.assertRaises(ProcessLookupError):
                        os.kill(wget_pid, 0)
                    lock_result = subprocess.run(
                        [
                            "flock",
                            "-n",
                            str(base / "archive" / ".download-inat2021.lock"),
                            "true",
                        ],
                        check=False,
                    )
                    self.assertEqual(lock_result.returncode, 0)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
                    if supervisor_pid is not None:
                        try:
                            os.kill(supervisor_pid, signal.SIGCONT)
                            os.kill(supervisor_pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    if pid_path.exists():
                        wget_pid = int(pid_path.read_text(encoding="utf-8"))
                        try:
                            os.kill(wget_pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass

    def test_term_waits_for_supervisor_to_kill_term_ignoring_child(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mock_bin = root / "bin"
            mock_bin.mkdir()
            pid_path = root / "wget.pid"
            mock_wget = mock_bin / "wget"
            mock_wget.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "output=\n"
                'for argument in "$@"; do\n'
                '  case "$argument" in\n'
                "    --output-document=*) output=${argument#*=} ;;\n"
                "  esac\n"
                "done\n"
                'echo $$ > "$MOCK_PID_FILE"\n'
                "trap '' TERM INT HUP\n"
                'while true; do printf x >> "$output"; sleep 0.05; done\n',
                encoding="utf-8",
            )
            mock_wget.chmod(0o755)
            project_root = Path(__file__).resolve().parents[1]
            base = root / "data"
            process = subprocess.Popen(
                [
                    "bash",
                    str(project_root / "scripts" / "download_inat2021.sh"),
                    "mini",
                ],
                env={
                    **os.environ,
                    "PATH": f"{mock_bin}:{os.environ['PATH']}",
                    "BASE": str(base),
                    "RUN_ID": "term-test",
                    "MOCK_PID_FILE": str(pid_path),
                    "HEARTBEAT_INTERVAL": "1",
                    "CHILD_STOP_ATTEMPTS": "10",
                },
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                for _ in range(100):
                    if pid_path.exists():
                        break
                    time.sleep(0.02)
                self.assertTrue(pid_path.exists())
                wget_pid = int(pid_path.read_text(encoding="utf-8"))
                process.send_signal(signal.SIGTERM)
                self.assertEqual(process.wait(timeout=6), 143)
                with self.assertRaises(ProcessLookupError):
                    os.kill(wget_pid, 0)
                lock_result = subprocess.run(
                    [
                        "flock",
                        "-n",
                        str(base / "archive" / ".download-inat2021.lock"),
                        "true",
                    ],
                    check=False,
                )
                self.assertEqual(lock_result.returncode, 0)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                if pid_path.exists():
                    wget_pid = int(pid_path.read_text(encoding="utf-8"))
                    try:
                        os.kill(wget_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_md5_verification_refreshes_heartbeat(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mock_bin = root / "bin"
            mock_bin.mkdir()
            mock_wget = mock_bin / "wget"
            mock_wget.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "output=\n"
                'for argument in "$@"; do\n'
                '  case "$argument" in\n'
                "    --output-document=*) output=${argument#*=} ;;\n"
                "  esac\n"
                "done\n"
                'case "${output##*/}" in\n'
                "  train_mini.tar.gz.part) size=44636137542 ;;\n"
                "  val.tar.gz.part) size=8931661582 ;;\n"
                "  *) exit 64 ;;\n"
                "esac\n"
                'truncate -s "$size" "$output"\n',
                encoding="utf-8",
            )
            mock_md5 = mock_bin / "md5sum"
            mock_md5.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "cat >/dev/null\n"
                'while [[ ! -e "$MD5_RELEASE" ]]; do sleep 0.05; done\n'
                "exit 0\n",
                encoding="utf-8",
            )
            mock_wget.chmod(0o755)
            mock_md5.chmod(0o755)
            project_root = Path(__file__).resolve().parents[1]
            base = root / "data"
            status_path = base / "metadata" / "heartbeat-test" / "status"
            md5_release = root / "md5-release"
            process = subprocess.Popen(
                [
                    "bash",
                    str(project_root / "scripts" / "download_inat2021.sh"),
                    "mini",
                ],
                env={
                    **os.environ,
                    "PATH": f"{mock_bin}:{os.environ['PATH']}",
                    "BASE": str(base),
                    "RUN_ID": "heartbeat-test",
                    "MAX_ATTEMPTS": "1",
                    "HEARTBEAT_INTERVAL": "1",
                    "MD5_RELEASE": str(md5_release),
                },
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    if status_path.exists() and "phase=md5" in status_path.read_text(
                        encoding="utf-8"
                    ):
                        break
                    time.sleep(0.02)
                self.assertTrue(status_path.exists())
                self.assertIn("phase=md5", status_path.read_text(encoding="utf-8"))
                first_status = dict(
                    line.split("=", 1) for line in status_path.read_text().splitlines()
                )
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    current_status = dict(
                        line.split("=", 1)
                        for line in status_path.read_text().splitlines()
                    )
                    if current_status["heartbeat_at"] != first_status["heartbeat_at"]:
                        break
                    time.sleep(0.02)
                self.assertNotEqual(
                    current_status["heartbeat_at"], first_status["heartbeat_at"]
                )
                self.assertEqual(current_status["phase"], "md5")
                self.assertEqual(
                    current_status["current_file"], first_status["current_file"]
                )
                self.assertIsNone(process.poll())
                md5_release.touch()
                self.assertEqual(process.wait(timeout=10), 0)
            finally:
                md5_release.touch()
                if process.poll() is None:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    unittest.main()
