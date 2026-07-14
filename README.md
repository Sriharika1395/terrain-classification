# Terrain Mapping from LiDAR Data Using Hybrid 3D-2D Deep Learning

Autonomous navigation requires more than object detection — hazards like slopes, pitfalls, and unstable surfaces come from point distributions and elevation geometry rather than distinct object classes. Existing approaches tend to trade off computational speed against semantic detail: 2D projection methods are fast but lose vertical geometry, 3D point-based methods keep the geometry but are expensive.

This repo contains three models evaluated on the KITTI Odometry / SemanticKITTI datasets:

1. **BEV U-Net** — a 2D model operating on bird's-eye-view projections of each LiDAR scan, built for real-time terrain classification.
2. **KPConv** — a 3D model performing semantic segmentation directly on raw point clouds, preserving full spatial geometry.
3. **Hybrid (KPConv + BEV U-Net, FiLM fusion)** — fuses 3D geometric features from KPConv into the BEV U-Net bottleneck via feature-wise linear modulation.

Each model has a distinct capability profile: the 2D pipeline is structurally aligned with navigation tasks at real-time latency, the 3D pipeline provides semantic depth and geometric richness at higher computational cost, and the hybrid combines both — surpassing either baseline on mIoU while staying close to real-time.

## Results

| Model | Pixel/Point Accuracy | mIoU | Inference | GPU memory |
|---|---|---|---|---|
| BEV U-Net | 98.70% | 0.749 (excl. Void) | ~10 ms | ~2 GB |
| KPConv | 91.7% (@500 epochs) | 0.561 (@500 epochs) | ~100 ms | ~8 GB |
| Hybrid | 98.67% | 0.761 | ~25-35 ms | ~4 GB |

Per-class IoU, hybrid model:

| Class | IoU | Notes |
|---|---|---|
| Flat/Traversable | 0.917 | strong, core navigation class |
| Slope | 0.823 | most significant improvement over BEV U-Net alone — KPConv's 3D geometry corrects ambiguity introduced by top-down projection |
| Obstacle | 0.885 | strong detection of non-traversable regions |
| Void | 0.181 | low due to extreme class rarity |

KPConv was trained in two stages: first on a subsampled ~15% slice of SemanticKITTI (44.7% mIoU at 200 epochs), then scaled to the full dataset for 500 epochs, reaching 56.1% mIoU — close to the official 58.8% baseline reported at 800 epochs on full data.

## Repo structure

```
.
├── HybridModel.ipynb        # hybrid KPConv + BEV U-Net, FiLM fusion
├── BEV_UNet.ipynb           # standalone 2D baseline
├── KPConv.ipynb             # standalone 3D baseline
└── README.md
```

## Setup

Requires the SemanticKITTI velodyne + labels data: [semantic-kitti dataset on Kaggle](https://www.kaggle.com/datasets/luischavarriazamora/semantic-kitti).

Originally trained on Kaggle (T4 GPU), with the dataset added via Kaggle's "Add Input" and paths following Kaggle's standard layout (`/kaggle/input/...`, `/kaggle/working`, `/kaggle/temp`). The notebooks now read these paths from environment variables instead, so they aren't tied to Kaggle specifically:

```bash
export SEMANTIC_KITTI_ROOT=/path/to/semantic-kitti
export WORK_DIR=./working     # checkpoints, plots
export TEMP_DIR=./temp        # cached BEV rasters / point hierarchies
```

To reproduce the original Kaggle setup:
1. Create a new Kaggle notebook, open **Settings** and turn on a GPU accelerator (T4).
2. Under **Add Input**, search for and add the `luischavarriazamora/semantic-kitti` dataset.
3. Kaggle mounts it at `/kaggle/input/datasets/luischavarriazamora/semantic-kitti`, with your working directory at `/kaggle/working` and scratch space at `/kaggle/temp`. Set the environment variables above to these paths before running the notebook.

The caching step writes one BEV raster, label map, and subsampled point cloud per scan to `TEMP_DIR/cache/`, plus the KPConv multi-scale point/neighbor hierarchy to `TEMP_DIR/scales/`. Already-cached scans are skipped, so it's safe to re-run. Checkpoints and plots go to `WORK_DIR/`.

Dependencies: PyTorch, NumPy, SciPy, tqdm, matplotlib.

## Methodology summary

**BEV U-Net.** Each scan is projected onto a 256x256 top-down grid at 0.2 m/pixel resolution, with five channels per cell: max height, min height, height range, log-normalized point density, and mean intensity. Flat traversable surfaces appear as dense low-height regions, obstacles as tall compact blobs, and slopes as gradual elevation gradients. The model has 7.76M parameters and outputs five terrain classes: Flat/Traversable, Slope, Obstacle, Unknown, and Void. Class-weighted cross-entropy handles the imbalance caused by the Unknown class covering ~70% of pixels. Trained for 50 epochs with AdamW (OneCycleLR to CosineAnnealingLR), FP16 mixed precision.

**KPConv.** Processes each scan as a raw, unstructured point cloud — no voxelization or projection — using learnable kernel points in continuous 3D space to extract local geometric features. This preserves fine and sparse structures (pedestrians, cyclists, poles) that BEV projection loses. Predicts across the 16 SemanticKITTI semantic categories. Trained on the full dataset with batch size 8, up to 500 epochs.

**Hybrid.** Each scan is processed through both branches in parallel: the 2D branch builds the same 256x256 five-channel BEV image used by the standalone U-Net, while the 3D branch subsamples 6,000 points from the original ~120,000. The KPConv encoder is modified to skip per-point classification and instead global-average-pools to a single 256-dimensional scene descriptor. That descriptor is passed through a linear layer producing two 512-dimensional vectors (gamma, beta), applied to the BEV bottleneck:

```
output = bottleneck * (1 + gamma) + beta
```

This conditions the BEV spatial representation with 3D geometric features before decoding. 8.78M parameters, trained on all 23,201 labeled scans (sequences 00-10, validated on 08), AdamW lr=3e-4, batch size 16, CosineAnnealingLR, FP16, early stopping patience 8. Converged at epoch 23, stopped at epoch 31.


