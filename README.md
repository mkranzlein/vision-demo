# Vision Demo

![CI](https://github.com/mkranzlein/vision-demo/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/mkranzlein/vision-demo/graph/badge.svg)](https://codecov.io/gh/mkranzlein/vision-demo)

## Dataset

This project uses the vehicle classes from [COCO 2017](https://cocodataset.org/) for object detection.

**Classes (8):** bicycle, car, motorcycle, airplane, bus, train, truck, boat

### Downloading the data

```bash
python scripts/download_coco.py
```

This will:

1. Download COCO 2017 annotation files (~252 MB)
2. Filter to vehicle classes only
3. Download the corresponding images (~6-9 GB for train, ~500 MB for val)

Data is saved to `data/coco/` (gitignored) with this structure:

```
data/coco/
├── annotations/          # Raw COCO annotation files
├── train/
│   ├── images/           # Training images
│   └── annotations.json  # Filtered vehicle annotations
├── val/
│   ├── images/           # Validation images
│   └── annotations.json  # Filtered vehicle annotations
└── test/
    ├── images/           # Test images (split from train)
    └── annotations.json  # Filtered vehicle annotations
```

## Infrastructure

The project uses Docker Compose for local infrastructure. MinIO provides S3-compatible
object storage, simulating a cloud training workflow locally.

### Starting services

```bash
docker compose up -d
```

This starts:

- **MinIO** — S3-compatible storage on `localhost:9000` (API) and `localhost:9001` (console UI)
- **MLflow** — Experiment tracking on `localhost:5000`, artifacts stored in MinIO

Default credentials: `minioadmin` / `minioadmin`

### Uploading data to MinIO

After downloading the COCO data, upload it to MinIO:

```bash
python scripts/upload_to_minio.py
```

### Training

The training script supports both single-GPU and distributed (DDP) modes, reading
data from local disk or MinIO.

Single GPU:

```bash
python scripts/train.py
```

Distributed (DDP via torchrun):

```bash
torchrun --nproc_per_node=1 scripts/train.py
```

Train from MinIO (simulates cloud training):

```bash
torchrun --nproc_per_node=1 scripts/train.py --from-minio
```

Training automatically logs to MLflow when the server is running. Use `--no-mlflow` to disable.

Options: `--epochs`, `--batch-size`, `--lr`, `--sample` (fraction of data to use)

### Inference API

Start the API server:

```bash
uvicorn vision_demo.app:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health` — health check
- `POST /detect` — upload an image, get back bounding box detections

The API loads the model checkpoint from `models/detector.pth` on first request.
