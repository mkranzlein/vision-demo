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
└── val/
    ├── images/           # Validation images
    └── annotations.json  # Filtered vehicle annotations
```
