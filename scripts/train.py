"""Training script for Faster R-CNN vehicle detector.

Supports single-GPU and DDP multi-GPU training. Reads data from local disk or MinIO.

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 5 --batch-size 4 --lr 0.005
    python scripts/train.py --from-minio
    torchrun --nproc_per_node=1 scripts/train.py
"""

import argparse
import logging
import os
import tempfile
import time
from pathlib import Path

import mlflow
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from vision_demo.data.dataset import CocoVehicleDataset, get_transforms
from vision_demo.model.detector import create_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "coco"


def collate_fn(batch):
    """Custom collate since Faster R-CNN expects list of images and list of targets."""
    return tuple(zip(*batch, strict=True))


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """Train for one epoch, logging loss periodically.

    Args:
        model: The detection model.
        optimizer: The optimizer.
        data_loader: Training data loader.
        device: Device to train on.
        epoch: Current epoch number (for logging).

    Returns:
        Average total loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    pbar = tqdm(
        data_loader,
        desc=f"Epoch {epoch}",
        unit="batch",
        leave=True,
        bar_format="{l_bar}{bar:30}{r_bar}",
    )
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{running_loss / num_batches:.4f}")

    epoch_loss = running_loss / max(num_batches, 1)
    return epoch_loss


def is_distributed() -> bool:
    """Check if the script was launched with torchrun/DDP."""
    return "RANK" in os.environ


def get_rank() -> int:
    """Return the global rank of this process (0 if not distributed)."""
    return int(os.environ.get("RANK", 0))


def is_main_process() -> bool:
    """Return True if this is rank 0 (or non-distributed)."""
    return get_rank() == 0


def setup_distributed():
    """Initialize the DDP process group."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    """Destroy the DDP process group."""
    dist.destroy_process_group()


def main():
    """Run training."""
    parser = argparse.ArgumentParser(description="Train Faster R-CNN vehicle detector.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Root COCO data directory.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum.")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Weight decay.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--sample", type=float, default=1.0, help="Fraction of training data to use (0.0-1.0).")
    parser.add_argument("--from-minio", action="store_true", help="Pull training data from MinIO.")
    parser.add_argument("--minio-endpoint", type=str, default="localhost:9000", help="MinIO endpoint.")
    parser.add_argument("--minio-bucket", type=str, default="vision-demo", help="MinIO bucket.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to save model checkpoint.")
    parser.add_argument("--mlflow-uri", type=str, default="http://localhost:5000", help="MLflow tracking URI.")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging.")
    args = parser.parse_args()

    # DDP setup
    distributed = is_distributed()
    local_rank = 0
    if distributed:
        local_rank = setup_distributed()
        if is_main_process():
            logger.info("DDP initialized: %d processes.", dist.get_world_size())

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_main_process():
        logger.info("Using device: %s", device)

    # Resolve data directory
    data_dir = args.data_dir
    if args.from_minio:
        from vision_demo.data.storage import download_directory, get_client

        client = get_client(endpoint=args.minio_endpoint)
        data_dir = Path(tempfile.mkdtemp(prefix="vision_demo_"))
        if is_main_process():
            logger.info("Downloading training data from MinIO to %s...", data_dir)
        download_directory(client, args.minio_bucket, "data/train/images", data_dir / "train" / "images")
        download_directory(client, args.minio_bucket, "data/train", data_dir / "train")
        if is_main_process():
            logger.info("MinIO download complete.")

    # Dataset
    train_dataset = CocoVehicleDataset(data_dir / "train", transforms=get_transforms(train=True))
    if args.sample < 1.0:
        n = int(len(train_dataset) * args.sample)
        indices = torch.randperm(len(train_dataset))[:n].tolist()
        train_dataset = Subset(train_dataset, indices)
    if is_main_process():
        logger.info("Training samples: %d", len(train_dataset))

    # Sampler and DataLoader
    sampler = DistributedSampler(train_dataset) if distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Model
    model = create_model()
    model.to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank])
    if is_main_process():
        logger.info("Model loaded on %s.", device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # MLflow setup (rank 0 only)
    use_mlflow = not args.no_mlflow and is_main_process()
    if use_mlflow:
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment("vehicle-detection")
        mlflow.start_run()
        mlflow.log_params(
            {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "momentum": args.momentum,
                "weight_decay": args.weight_decay,
                "sample": args.sample,
                "from_minio": args.from_minio,
                "num_training_samples": len(train_dataset),
            }
        )

    # Training loop
    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        start = time.time()
        epoch_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        elapsed = time.time() - start
        if is_main_process():
            logger.info("Epoch %d complete — loss: %.4f, time: %.1fs", epoch, epoch_loss, elapsed)
        if use_mlflow:
            mlflow.log_metrics({"loss": epoch_loss, "epoch_time_s": elapsed}, step=epoch)

    # Save checkpoint (rank 0 only)
    if is_main_process():
        output_dir = args.output_dir or (args.data_dir.parent.parent / "models")
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / "detector.pth"
        state = model.module.state_dict() if distributed else model.state_dict()
        torch.save(state, checkpoint_path)
        logger.info("Saved checkpoint to %s", checkpoint_path)
        if use_mlflow:
            mlflow.log_artifact(str(checkpoint_path))
            mlflow.end_run()

    if distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()
