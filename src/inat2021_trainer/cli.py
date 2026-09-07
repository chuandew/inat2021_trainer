#!/usr/bin/env python3
"""Observable single-GPU ImageFolder training for DingoFS validation."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Sampler, SequentialSampler, Subset
from torchvision import transforms

from inat2021_trainer.manifest import load_manifest_identity

CHECKPOINT_SCHEMA = 1


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


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
        fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class JsonlLogger:
    def __init__(self, path: Path, run_id: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self._closed = False
        self._handle = path.open("a", encoding="utf-8", buffering=1)

    def write(self, event: dict[str, Any]) -> None:
        event = {**event, "timestamp": utc_now(), "run_id": self.run_id}
        self._handle.write(json.dumps(event, sort_keys=True) + "\n")

    def close(self) -> None:
        if not self._closed:
            self._handle.close()
            self._closed = True


_ACTIVE_LOGGER: JsonlLogger | None = None
_OUTPUT_LOCK: Any = None
_FAILURE_STATE: dict[str, Any] = {}


def deterministic_sample_seed(seed: int, epoch: int, index: int) -> int:
    encoded = f"{seed}:{epoch}:{index}".encode()
    digest = hashlib.blake2b(encoded, digest_size=8, person=b"inat-aug").digest()
    return int.from_bytes(digest, "little") & 0x7FFF_FFFF_FFFF_FFFF


class TimedImageFolder(torchvision.datasets.ImageFolder):
    """ImageFolder that reports decode/transform time and deterministic augmentation."""

    def __init__(self, root: Path, sample_transform: Any, seed: int) -> None:
        super().__init__(str(root), transform=None)
        self.sample_transform = sample_transform
        self.seed = seed
        self.root_path = root.resolve()
        self._root_prefix = str(self.root_path) + os.sep

    def relative_path(self, path: str) -> str:
        if path.startswith(self._root_prefix):
            return path[len(self._root_prefix) :]
        return os.path.relpath(path, self.root_path)

    def __getitem__(
        self, key: int | tuple[int, int]
    ) -> tuple[torch.Tensor, int, float, float]:
        if isinstance(key, tuple):
            index, epoch = key
        else:
            index, epoch = key, 0
        path, target = self.samples[index]
        relative_path = self.relative_path(path)

        decode_started = time.perf_counter()
        try:
            sample = self.loader(path)
        except BaseException as error:
            raise RuntimeError(f"failed to decode {relative_path}: {error}") from error
        decode_ms = (time.perf_counter() - decode_started) * 1000.0

        transform_started = time.perf_counter()
        sample_seed = deterministic_sample_seed(self.seed, epoch, index)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(sample_seed)
            sample = self.sample_transform(sample)
        transform_ms = (time.perf_counter() - transform_started) * 1000.0
        return sample, target, decode_ms, transform_ms


class EpochPermutationSampler(Sampler[tuple[int, int]]):
    """Deterministic epoch permutation with a committed-sample resume cursor."""

    def __init__(self, size: int, seed: int, epoch: int, start_index: int = 0) -> None:
        if size <= 0:
            raise ValueError("dataset must contain at least one sample")
        if not 0 <= start_index <= size:
            raise ValueError(
                f"invalid start_index {start_index} for dataset size {size}"
            )
        self.size = size
        self.seed = seed
        self.epoch = epoch
        self.start_index = start_index

    def __iter__(self) -> Iterator[tuple[int, int]]:
        order = np.random.default_rng(self.seed + self.epoch).permutation(self.size)
        for position in range(self.start_index, self.size):
            yield int(order[position]), self.epoch

    def __len__(self) -> int:
        return self.size - self.start_index


def seed_worker(worker_id: int) -> None:
    del worker_id
    global _OUTPUT_LOCK
    if _OUTPUT_LOCK is not None:
        _OUTPUT_LOCK.close()
        _OUTPUT_LOCK = None
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def cleanup_checkpoint_temporaries(checkpoint_dir: Path) -> list[str]:
    removed: list[str] = []
    if not checkpoint_dir.is_dir():
        return removed
    for path in checkpoint_dir.glob("*.tmp.*"):
        if not path.is_file():
            continue
        suffix = path.name.split(".tmp.", 1)[1]
        owner = suffix.split(".", 1)[0]
        if owner.isdigit() and process_is_running(int(owner)):
            continue
        path.unlink()
        removed.append(str(path))
    return removed


def claim_output_directory(path: Path, resume: str, run_id: str) -> Any:
    path.mkdir(parents=True, exist_ok=True)
    lock = (path / ".inat2021-train.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError(
            f"output directory is owned by another trainer: {path}"
        ) from None
    try:
        lock.seek(0)
        lock.truncate()
        lock.write(f"pid={os.getpid()}\nrun_id={run_id}\nstarted_at={utc_now()}\n")
        lock.flush()
        os.fsync(lock.fileno())
        conflicts = [path / "train.jsonl", path / "config.json"]
        checkpoint_dir = path / "checkpoints"
        if checkpoint_dir.is_dir() and any(checkpoint_dir.iterdir()):
            conflicts.append(checkpoint_dir)
        existing = [candidate for candidate in conflicts if candidate.exists()]
        if not resume and existing:
            names = ", ".join(str(candidate) for candidate in existing)
            raise RuntimeError(
                f"output directory contains an existing training run: {names}"
            )
    except BaseException:
        lock.close()
        raise
    return lock


def configure_determinism(enabled: bool) -> None:
    if enabled:
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace not in {None, ":4096:8", ":16:8"}:
            raise RuntimeError(
                f"unsupported CUBLAS_WORKSPACE_CONFIG for deterministic training: {workspace}"
            )
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = not enabled
    torch.backends.cudnn.deterministic = enabled
    torch.use_deterministic_algorithms(enabled)


def build_transforms(
    train_crop_size: int, val_resize_size: int, val_crop_size: int
) -> tuple[Any, Any]:
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(train_crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(val_resize_size),
            transforms.CenterCrop(val_crop_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, val_transform


def resolve_weights(name: str) -> Any:
    if name.lower() in {"", "none", "null"}:
        return None
    return torchvision.models.get_weight(name)


def cached_weights_sha256(weights: Any) -> str | None:
    if weights is None:
        return None
    filename = Path(urlparse(weights.url).path).name
    cache_path = Path(torch.hub.get_dir()) / "checkpoints" / filename
    return sha256_file(cache_path) if cache_path.is_file() else None


def build_model(
    model_name: str, weights_name: str, num_classes: int
) -> tuple[nn.Module, str | None]:
    weights = resolve_weights(weights_name)
    if weights is None:
        model = torchvision.models.get_model(
            model_name, weights=None, num_classes=num_classes
        )
    else:
        model = torchvision.models.get_model(model_name, weights=weights)
        if not hasattr(model, "fc") or not isinstance(model.fc, nn.Linear):
            raise ValueError(
                f"pretrained classifier replacement is unsupported for model {model_name}"
            )
        input_features = model.fc.in_features
        model.fc = nn.Linear(input_features, num_classes)
        nn.init.normal_(model.fc.weight, mean=0.0, std=0.01)
        nn.init.zeros_(model.fc.bias)
    return model, cached_weights_sha256(weights)


def make_lr_scheduler(
    optimizer: torch.optim.Optimizer, max_steps: int, warmup_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    if not 0 <= warmup_steps < max_steps:
        raise ValueError("warmup_steps must be non-negative and smaller than max_steps")

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def config_identity(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def require_final_checkpoint(global_step: int, last_checkpoint_step: int) -> None:
    if last_checkpoint_step != global_step:
        raise RuntimeError("final step was not checkpointed")


def save_checkpoint_atomic(
    checkpoint_dir: Path,
    checkpoint: dict[str, Any],
    global_step: int,
    expected_config_identity: str,
    expected_dataset_identity: str,
) -> dict[str, float | str]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_path = checkpoint_dir / f"step-{global_step:09d}.pt"
    if final_path.exists():
        raise RuntimeError(
            f"checkpoint already exists and will not be overwritten: {final_path}"
        )
    temporary = (
        checkpoint_dir / f"{final_path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )

    write_started = time.perf_counter()
    with temporary.open("xb") as handle:
        torch.save(checkpoint, handle)
        handle.flush()
        os.fsync(handle.fileno())
    write_fsync_ms = (time.perf_counter() - write_started) * 1000.0

    readback_started = time.perf_counter()
    readback = torch.load(temporary, map_location="cpu", weights_only=False)
    if readback.get("schema_version") != CHECKPOINT_SCHEMA:
        raise RuntimeError("checkpoint readback schema mismatch")
    if readback.get("config_identity") != expected_config_identity:
        raise RuntimeError("checkpoint readback config identity mismatch")
    if readback.get("dataset_identity") != expected_dataset_identity:
        raise RuntimeError("checkpoint readback dataset identity mismatch")
    if readback.get("global_step") != global_step:
        raise RuntimeError("checkpoint readback step mismatch")
    readback_ms = (time.perf_counter() - readback_started) * 1000.0

    checksum_started = time.perf_counter()
    checksum = sha256_file(temporary)
    checksum_ms = (time.perf_counter() - checksum_started) * 1000.0

    publish_started = time.perf_counter()
    os.replace(temporary, final_path)
    fsync_directory(checkpoint_dir)
    _FAILURE_STATE.update(global_step=global_step, checkpoint=str(final_path))
    rename_fsync_ms = (time.perf_counter() - publish_started) * 1000.0

    latest_started = time.perf_counter()
    latest = {
        "schema_version": CHECKPOINT_SCHEMA,
        "checkpoint": final_path.name,
        "sha256": checksum,
        "global_step": global_step,
        "config_identity": expected_config_identity,
        "dataset_identity": expected_dataset_identity,
        "created_at": utc_now(),
    }
    atomic_json_write(checkpoint_dir / "latest.json", latest)
    latest_publish_ms = (time.perf_counter() - latest_started) * 1000.0

    return {
        "checkpoint_path": str(final_path),
        "checkpoint_sha256": checksum,
        "checkpoint_write_fsync_ms": write_fsync_ms,
        "checkpoint_readback_ms": readback_ms,
        "checkpoint_checksum_ms": checksum_ms,
        "checkpoint_rename_fsync_ms": rename_fsync_ms,
        "checkpoint_latest_publish_ms": latest_publish_ms,
    }


def resolve_resume_checkpoint(
    resume: str, checkpoint_dir: Path
) -> tuple[Path, dict[str, Any]]:
    if resume != "latest":
        raise ValueError("only --resume latest is supported")
    latest_path = checkpoint_dir / "latest.json"
    with latest_path.open(encoding="utf-8") as handle:
        latest = json.load(handle)
    if latest.get("schema_version") != CHECKPOINT_SCHEMA:
        raise RuntimeError("latest.json schema mismatch")
    filename = latest.get("checkpoint")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise RuntimeError("latest.json contains an unsafe checkpoint path")
    path = checkpoint_dir / filename
    actual_checksum = sha256_file(path)
    if actual_checksum != latest.get("sha256"):
        raise RuntimeError("latest checkpoint checksum mismatch")
    return path, latest


def checkpoint_step(path: Path) -> int | None:
    prefix = "step-"
    suffix = ".pt"
    if not path.name.startswith(prefix) or not path.name.endswith(suffix):
        return None
    encoded_step = path.name[len(prefix) : -len(suffix)]
    if len(encoded_step) != 9 or not encoded_step.isdigit():
        return None
    return int(encoded_step)


def validate_checkpoint(
    path: Path,
    expected_step: int,
    expected_config_identity: str,
    expected_dataset_identity: str,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA:
        raise RuntimeError("checkpoint schema mismatch")
    if checkpoint.get("config_identity") != expected_config_identity:
        raise RuntimeError("resume config identity mismatch")
    if checkpoint.get("dataset_identity") != expected_dataset_identity:
        raise RuntimeError("resume dataset identity mismatch")
    if checkpoint.get("global_step") != expected_step:
        raise RuntimeError("checkpoint step mismatch")
    checkpoint["resolved_path"] = str(path)
    return checkpoint


def load_checkpoint_strict(
    resume: str,
    checkpoint_dir: Path,
    expected_config_identity: str,
    expected_dataset_identity: str,
) -> dict[str, Any]:
    if resume != "latest":
        raise ValueError("only --resume latest is supported")

    latest_path = checkpoint_dir / "latest.json"
    latest_step = -1
    checkpoint = None
    if latest_path.exists():
        path, latest = resolve_resume_checkpoint(resume, checkpoint_dir)
        latest_step = latest.get("global_step")
        if not isinstance(latest_step, int) or isinstance(latest_step, bool):
            raise RuntimeError("latest.json step is invalid")
        if checkpoint_step(path) != latest_step:
            raise RuntimeError("latest.json checkpoint filename step mismatch")
        checkpoint = validate_checkpoint(
            path,
            latest_step,
            expected_config_identity,
            expected_dataset_identity,
        )

    newer = [
        (step, candidate)
        for candidate in checkpoint_dir.glob("step-*.pt")
        if (step := checkpoint_step(candidate)) is not None and step > latest_step
    ]
    if not newer:
        if checkpoint is None:
            raise RuntimeError("no resumable checkpoint exists")
        return checkpoint

    orphan_step, orphan_path = max(newer)
    orphan = validate_checkpoint(
        orphan_path,
        orphan_step,
        expected_config_identity,
        expected_dataset_identity,
    )
    checksum = sha256_file(orphan_path)
    atomic_json_write(
        latest_path,
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "checkpoint": orphan_path.name,
            "sha256": checksum,
            "global_step": orphan_step,
            "config_identity": expected_config_identity,
            "dataset_identity": expected_dataset_identity,
            "created_at": utc_now(),
        },
    )
    return orphan


def visible_gpu_identifier() -> str:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return visible_devices.split(",", 1)[0].strip() or "0"


def gpu_snapshot() -> dict[str, int | None]:
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
        f"--id={visible_gpu_identifier()}",
    ]
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=5
        )
        utilization, memory_used = [
            int(value.strip()) for value in result.stdout.strip().split(",")
        ]
        return {"gpu_utilization_pct": utilization, "gpu_memory_used_mib": memory_used}
    except (OSError, subprocess.SubprocessError, ValueError):
        return {"gpu_utilization_pct": None, "gpu_memory_used_mib": None}


def accuracy(
    output: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...]
) -> list[float]:
    with torch.no_grad():
        maxk = max(topk)
        _, prediction = output.topk(maxk, 1, True, True)
        prediction = prediction.t()
        correct = prediction.eq(target.reshape(1, -1).expand_as(prediction))
        return [
            float(
                correct[:k].reshape(-1).float().sum().item() * 100.0 / target.shape[0]
            )
            for k in topk
        ]


def make_data_loader(
    dataset: Any,
    sampler: Sampler[Any],
    batch_size: int,
    workers: int,
    seed: int,
    drop_last: bool,
) -> DataLoader[Any]:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "sampler": sampler,
        "num_workers": workers,
        "pin_memory": True,
        "drop_last": drop_last,
        "worker_init_fn": seed_worker,
        "generator": torch.Generator().manual_seed(seed),
    }
    if workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**kwargs)


def bounded_validation_dataset(
    dataset: TimedImageFolder, batch_size: int, max_batches: int
) -> Any:
    if max_batches == 0:
        return dataset
    sample_count = min(len(dataset), batch_size * max_batches)
    return Subset(dataset, range(sample_count))


def step_throughput_metrics(
    batch_size: int, dataloader_wait_ms: float, batch_total_ms: float
) -> dict[str, float]:
    end_to_end_batch_ms = dataloader_wait_ms + batch_total_ms
    return {
        "end_to_end_batch_ms": end_to_end_batch_ms,
        "end_to_end_images_per_second": batch_size
        / max(end_to_end_batch_ms / 1000.0, 1e-9),
        "gpu_path_images_per_second": batch_size / max(batch_total_ms / 1000.0, 1e-9),
    }


def evaluate(
    model: nn.Module,
    dataset: TimedImageFolder,
    device: torch.device,
    batch_size: int,
    workers: int,
    seed: int,
    max_batches: int,
) -> dict[str, float | int]:
    validation_dataset = bounded_validation_dataset(dataset, batch_size, max_batches)
    loader = make_data_loader(
        dataset=validation_dataset,
        sampler=SequentialSampler(validation_dataset),
        batch_size=batch_size,
        workers=workers,
        seed=seed,
        drop_last=False,
    )
    criterion = nn.CrossEntropyLoss()
    model.eval()
    processed = 0
    loss_sum = 0.0
    top1_sum = 0.0
    top5_sum = 0.0
    started = time.perf_counter()
    top5_k = min(5, len(dataset.classes))
    with torch.inference_mode():
        for images, targets, _, _ in loader:
            images = images.to(
                device, non_blocking=True, memory_format=torch.channels_last
            )
            targets = targets.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = model(images)
                loss = criterion(output, targets)
            batch_size_actual = targets.shape[0]
            top1, top5 = accuracy(output, targets, (1, top5_k))
            processed += batch_size_actual
            loss_sum += float(loss.item()) * batch_size_actual
            top1_sum += top1 * batch_size_actual
            top5_sum += top5 * batch_size_actual
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "validation_samples": processed,
        "validation_loss": loss_sum / max(processed, 1),
        "validation_top1_pct": top1_sum / max(processed, 1),
        "validation_top5_pct": top5_sum / max(processed, 1),
        "validation_images_per_second": processed / max(elapsed, 1e-9),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        required=True,
        help=(
            "immutable manifest created by inat2021-manifest; startup checks only "
            "manifest/counts/classes, without reading image content. Trusts a "
            "previously verified, unchanged dataset; same-count/same-size edits "
            "are not detected. Run inat2021-manifest --verify-existing explicitly "
            "for full content verification before training"
        ),
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--weights", default="ResNet50_Weights.IMAGENET1K_V2")
    parser.add_argument("--expected-classes", type=int, default=10_000)
    parser.add_argument("--expected-train-samples", type=int, required=True)
    parser.add_argument("--expected-validation-samples", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument(
        "--validation-batches",
        type=int,
        default=50,
        help="number of validation batches; 0 evaluates the full validation split",
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2_026_090_2)
    parser.add_argument("--train-crop-size", type=int, default=224)
    parser.add_argument("--val-resize-size", type=int, default=256)
    parser.add_argument("--val-crop-size", type=int, default=224)
    parser.add_argument("--resume", choices=("latest",), default="")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="require deterministic algorithms and fail if an operation has no deterministic implementation",
    )
    parser.add_argument(
        "--fault-sigkill-after-step",
        type=int,
        default=0,
        help="send SIGKILL to this process after the committed step; 0 disables fault injection",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.train_split == args.val_split:
        raise ValueError("train_split and val_split must differ")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.workers < 0:
        raise ValueError("workers must be non-negative")
    if args.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if not 0 <= args.warmup_steps < args.max_steps:
        raise ValueError("warmup_steps must be smaller than max_steps")
    if args.checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    if args.validation_batches < 0:
        raise ValueError("validation_batches must be non-negative")
    if args.log_every <= 0:
        raise ValueError("log_every must be positive")
    if not 0 <= args.fault_sigkill_after_step <= args.max_steps:
        raise ValueError("fault_sigkill_after_step must be between 0 and max_steps")
    if args.expected_classes <= 0:
        raise ValueError("expected_classes must be positive")
    if args.expected_train_samples <= 0:
        raise ValueError("expected_train_samples must be positive")
    if args.expected_validation_samples <= 0:
        raise ValueError("expected_validation_samples must be positive")
    if args.train_crop_size <= 0:
        raise ValueError("train_crop_size must be positive")
    if args.val_resize_size <= 0:
        raise ValueError("val_resize_size must be positive")
    if args.val_crop_size <= 0:
        raise ValueError("val_crop_size must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(args.momentum) or not 0 <= args.momentum < 1:
        raise ValueError("momentum must be finite and in [0, 1)")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight_decay must be finite and non-negative")


def _main() -> int:
    args = parse_args()
    validate_args(args)
    configure_determinism(args.deterministic)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    args.data_root = args.data_root.resolve()
    args.dataset_manifest = args.dataset_manifest.resolve()
    args.output_dir = args.output_dir.resolve()
    train_root = args.data_root / args.train_split
    val_root = args.data_root / args.val_split
    checkpoint_dir = args.output_dir / "checkpoints"
    run_id = (
        f"train-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"
    )
    global _OUTPUT_LOCK
    _OUTPUT_LOCK = claim_output_directory(args.output_dir, args.resume, run_id)
    logger = JsonlLogger(args.output_dir / "train.jsonl", run_id)
    global _ACTIVE_LOGGER, _FAILURE_STATE
    _ACTIVE_LOGGER = logger
    _FAILURE_STATE = {"phase": "startup", "global_step": 0, "checkpoint": None}
    removed_checkpoint_temporaries = (
        cleanup_checkpoint_temporaries(checkpoint_dir) if args.resume else []
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    startup_started = time.perf_counter()
    train_transform, val_transform = build_transforms(
        args.train_crop_size, args.val_resize_size, args.val_crop_size
    )
    train_dataset = TimedImageFolder(train_root, train_transform, args.seed)
    val_dataset = TimedImageFolder(val_root, val_transform, args.seed)
    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise RuntimeError("train and validation class mappings differ")
    num_classes = len(train_dataset.classes)
    if num_classes != args.expected_classes:
        raise RuntimeError(
            f"expected {args.expected_classes} classes, found {num_classes}"
        )
    if len(train_dataset) != args.expected_train_samples:
        raise RuntimeError(
            f"expected {args.expected_train_samples} training samples, "
            f"found {len(train_dataset)}"
        )
    if len(val_dataset) != args.expected_validation_samples:
        raise RuntimeError(
            f"expected {args.expected_validation_samples} validation samples, "
            f"found {len(val_dataset)}"
        )
    combined_dataset_identity = load_manifest_identity(
        args.dataset_manifest,
        args.train_split,
        args.val_split,
        args.expected_train_samples,
        args.expected_validation_samples,
        args.expected_classes,
        train_dataset.samples,
        val_dataset.samples,
        train_dataset.class_to_idx,
        val_dataset.class_to_idx,
    )
    dataset_startup_ms = (time.perf_counter() - startup_started) * 1000.0

    device = torch.device("cuda:0")
    model, weights_sha256 = build_model(args.model, args.weights, num_classes)
    model.to(device, memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = make_lr_scheduler(optimizer, args.max_steps, args.warmup_steps)

    immutable_config = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "data_root": str(args.data_root),
        "train_split": args.train_split,
        "val_split": args.val_split,
        "dataset_identity": combined_dataset_identity,
        "dataset_manifest_sha256": combined_dataset_identity,
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "num_classes": num_classes,
        "expected_train_samples": args.expected_train_samples,
        "expected_validation_samples": args.expected_validation_samples,
        "model": args.model,
        "weights": args.weights,
        "weights_sha256": weights_sha256,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "warmup_steps": args.warmup_steps,
        "learning_rate": args.learning_rate,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "train_crop_size": args.train_crop_size,
        "val_resize_size": args.val_resize_size,
        "val_crop_size": args.val_crop_size,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "cublas_workspace_config": (
            os.environ.get("CUBLAS_WORKSPACE_CONFIG") if args.deterministic else None
        ),
        "precision": "bf16-autocast-fp32-parameters",
        "memory_format": "channels_last",
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
    }
    immutable_config_identity = config_identity(immutable_config)

    global_step = 0
    epoch = 0
    consumed_in_epoch = 0
    total_consumed = 0
    parent_checkpoint = None
    if args.resume:
        checkpoint = load_checkpoint_strict(
            args.resume,
            checkpoint_dir,
            immutable_config_identity,
            combined_dataset_identity,
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        global_step = checkpoint["global_step"]
        epoch = checkpoint["epoch"]
        consumed_in_epoch = checkpoint["consumed_in_epoch"]
        total_consumed = checkpoint["total_consumed"]
        parent_checkpoint = checkpoint["resolved_path"]
        restore_rng_state(checkpoint["rng_state"])
        _FAILURE_STATE.update(
            phase="resume",
            global_step=global_step,
            checkpoint=parent_checkpoint,
        )
        logger.write(
            {
                "event": "resume",
                "checkpoint": parent_checkpoint,
                "global_step": global_step,
                "epoch": epoch,
                "consumed_in_epoch": consumed_in_epoch,
                "total_consumed": total_consumed,
            }
        )
    atomic_json_write(
        args.output_dir / "config.json",
        {
            **immutable_config,
            "config_identity": immutable_config_identity,
            "dataset_manifest": str(args.dataset_manifest),
            "workers": args.workers,
            "checkpoint_every": args.checkpoint_every,
            "validation_batches": args.validation_batches,
            "run_id": run_id,
            "created_at": utc_now(),
        },
    )

    logger.write(
        {
            "event": "startup",
            "dataset_startup_ms": dataset_startup_ms,
            "train_samples": len(train_dataset),
            "validation_samples": len(val_dataset),
            "num_classes": num_classes,
            "dataset_identity": combined_dataset_identity,
            "config_identity": immutable_config_identity,
            "workers": args.workers,
            "dataset_manifest": str(args.dataset_manifest),
            "removed_checkpoint_temporaries": removed_checkpoint_temporaries,
        }
    )
    _FAILURE_STATE.update(
        phase="training", global_step=global_step, checkpoint=parent_checkpoint
    )

    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    h2d_start = torch.cuda.Event(enable_timing=True)
    h2d_end = torch.cuda.Event(enable_timing=True)
    forward_end = torch.cuda.Event(enable_timing=True)
    backward_end = torch.cuda.Event(enable_timing=True)
    optimizer_end = torch.cuda.Event(enable_timing=True)
    last_checkpoint_step = global_step if args.resume else -1
    while global_step < args.max_steps:
        sampler = EpochPermutationSampler(
            len(train_dataset), args.seed, epoch, start_index=consumed_in_epoch
        )
        loader = make_data_loader(
            train_dataset,
            sampler,
            args.batch_size,
            args.workers,
            args.seed + epoch,
            drop_last=True,
        )
        iterator = iter(loader)
        epoch_progressed = False
        while global_step < args.max_steps:
            wait_started = time.perf_counter()
            try:
                images, targets, decode_ms, transform_ms = next(iterator)
            except StopIteration:
                break
            dataloader_wait_ms = (time.perf_counter() - wait_started) * 1000.0
            epoch_progressed = True

            optimizer.zero_grad(set_to_none=True)
            learning_rate_used = float(optimizer.param_groups[0]["lr"])

            batch_started = time.perf_counter()
            h2d_start.record()
            images = images.to(
                device, non_blocking=True, memory_format=torch.channels_last
            )
            targets = targets.to(device, non_blocking=True)
            h2d_end.record()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = model(images)
                loss = criterion(output, targets)
            forward_end.record()
            loss.backward()
            backward_end.record()
            optimizer.step()
            optimizer_end.record()
            scheduler.step()
            torch.cuda.synchronize(device)
            batch_total_ms = (time.perf_counter() - batch_started) * 1000.0

            batch_size_actual = targets.shape[0]
            global_step += 1
            consumed_in_epoch += batch_size_actual
            total_consumed += batch_size_actual
            checkpoint_metrics: dict[str, float | str] = {}
            if (
                global_step % args.checkpoint_every == 0
                or global_step == args.max_steps
            ):
                checkpoint = {
                    "schema_version": CHECKPOINT_SCHEMA,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "consumed_in_epoch": consumed_in_epoch,
                    "total_consumed": total_consumed,
                    "rng_state": capture_rng_state(),
                    "sampler_seed": args.seed,
                    "dataset_identity": combined_dataset_identity,
                    "config_identity": immutable_config_identity,
                    "immutable_config": immutable_config,
                    "parent_checkpoint": parent_checkpoint,
                    "created_at": utc_now(),
                }
                checkpoint_metrics = save_checkpoint_atomic(
                    checkpoint_dir,
                    checkpoint,
                    global_step,
                    immutable_config_identity,
                    combined_dataset_identity,
                )
                parent_checkpoint = str(checkpoint_metrics["checkpoint_path"])
                _FAILURE_STATE.update(
                    global_step=global_step,
                    checkpoint=parent_checkpoint,
                )
                last_checkpoint_step = global_step

            telemetry = (
                gpu_snapshot()
                if global_step % args.log_every == 0
                else {
                    "gpu_utilization_pct": None,
                    "gpu_memory_used_mib": None,
                }
            )
            step_event: dict[str, Any] = {
                "event": "train_step",
                "global_step": global_step,
                "epoch": epoch,
                "consumed_in_epoch": consumed_in_epoch,
                "total_consumed": total_consumed,
                "batch_size": batch_size_actual,
                "loss": float(loss.item()),
                "learning_rate": learning_rate_used,
                "dataloader_wait_ms": dataloader_wait_ms,
                "decode_ms_mean": float(decode_ms.float().mean().item()),
                "decode_ms_max": float(decode_ms.float().max().item()),
                "transform_ms_mean": float(transform_ms.float().mean().item()),
                "h2d_ms": h2d_start.elapsed_time(h2d_end),
                "forward_ms": h2d_end.elapsed_time(forward_end),
                "backward_ms": forward_end.elapsed_time(backward_end),
                "optimizer_ms": backward_end.elapsed_time(optimizer_end),
                "batch_total_ms": batch_total_ms,
                **step_throughput_metrics(
                    batch_size_actual, dataloader_wait_ms, batch_total_ms
                ),
                "gpu_allocated_mib": torch.cuda.memory_allocated(device) / (1024**2),
                "gpu_reserved_mib": torch.cuda.memory_reserved(device) / (1024**2),
                "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated(device)
                / (1024**2),
                **telemetry,
                **checkpoint_metrics,
            }
            logger.write(step_event)
            if global_step % args.log_every == 0 or checkpoint_metrics:
                print(
                    f"step={global_step}/{args.max_steps} epoch={epoch} "
                    f"loss={loss.item():.6f} "
                    f"end_to_end_images/s="
                    f"{step_event['end_to_end_images_per_second']:.2f} "
                    f"gpu_path_images/s="
                    f"{step_event['gpu_path_images_per_second']:.2f} "
                    f"wait_ms={dataloader_wait_ms:.2f}",
                    flush=True,
                )
            if global_step == args.fault_sigkill_after_step:
                logger.write(
                    {
                        "event": "fault_injection",
                        "signal": "SIGKILL",
                        "global_step": global_step,
                    }
                )
                print(f"fault-injection=SIGKILL step={global_step}", flush=True)
                os.kill(os.getpid(), signal.SIGKILL)

        if not epoch_progressed and consumed_in_epoch == 0:
            raise RuntimeError("training loader produced no complete batch")
        if global_step < args.max_steps:
            epoch += 1
            consumed_in_epoch = 0

    require_final_checkpoint(global_step, last_checkpoint_step)

    _FAILURE_STATE.update(
        phase="validation", global_step=global_step, checkpoint=parent_checkpoint
    )
    validation = evaluate(
        model,
        val_dataset,
        device,
        args.batch_size,
        args.workers,
        args.seed,
        args.validation_batches,
    )
    logger.write({"event": "validation", "global_step": global_step, **validation})
    logger.write(
        {
            "event": "complete",
            "global_step": global_step,
            "epoch": epoch,
            "total_consumed": total_consumed,
            "checkpoint": parent_checkpoint,
        }
    )
    logger.close()
    _ACTIVE_LOGGER = None
    print(
        json.dumps(
            {"status": "completed", "global_step": global_step, **validation},
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    global _ACTIVE_LOGGER, _OUTPUT_LOCK, _FAILURE_STATE
    try:
        return _main()
    except BaseException as error:
        if _ACTIVE_LOGGER is not None:
            try:
                _ACTIVE_LOGGER.write(
                    {
                        "event": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        **_FAILURE_STATE,
                    }
                )
                _ACTIVE_LOGGER.close()
            except OSError as logging_error:
                print(
                    f"failed to record terminal training error: {logging_error}",
                    file=sys.stderr,
                    flush=True,
                )
        raise
    finally:
        _ACTIVE_LOGGER = None
        _FAILURE_STATE = {}
        if _OUTPUT_LOCK is not None:
            _OUTPUT_LOCK.close()
            _OUTPUT_LOCK = None


if __name__ == "__main__":
    raise SystemExit(main())
