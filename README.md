# Open-VocGSSLAM Evaluation

Minimal, evaluation-only package for Open-VocGSSLAM Gaussian maps. It contains
the custom RGB/depth/semantic CUDA rasterizer, Gaussian checkpoint loader, camera
utilities, and geometric and Replica semantic evaluation scripts. It does not
contain SLAM training/mapping code, datasets, model checkpoints, or results.

## Requirements

- Linux and an NVIDIA CUDA GPU
- CUDA toolkit compatible with the installed PyTorch build
- A C++ compiler and `nvcc`
- Python 3.9 or 3.10

The reference environment uses PyTorch 2.5.1, torchvision 0.20.1, and CUDA 12.1.
Install the matching PyTorch/torchvision wheels first by following the official
PyTorch selector. Then install this repository:

```bash
git clone <repository-url>
cd Open-VocGSSLAM-Eval
PYTHON_BIN=python ./install.sh
python scripts/smoke_test.py
```

Do not install another `diff-gaussian-rasterization` implementation into the
same environment. This repository includes the customized rasterizer required
by `render_3`.

## Expected input layout

Copy or mount data and results outside Git:

```text
Replica/
  cam_params.json
  room0/
    images/
    depth_images/
    semantic_class/
    traj.txt
    info_semantic.json
saved_results/room0/
  estimated_poses.npy
  geometry_refinement/iter_001000/
    scene_final.ply
    scene_final.pth
```

## Geometric evaluation

```bash
python scripts/evaluate_pruning_sweep.py \
  --model-dir saved_results/room0/geometry_refinement/iter_001000 \
  --dataset-path Replica \
  --scene-name room0 \
  --estimated-poses-path saved_results/room0/estimated_poses.npy
```

## Rendered semantic evaluation

```bash
python scripts/evaluate_semantic_pruning_sweep.py \
  --model-dir saved_results/room0/geometry_refinement/iter_001000 \
  --dataset-path Replica \
  --scene-name room0 \
  --estimated-poses-path saved_results/room0/estimated_poses.npy
```

## Single-query render

```bash
python eval.py \
  --model_path saved_results/room0/geometry_refinement/iter_001000 \
  --dataset_path Replica \
  --scene_name room0 \
  --include_feature \
  --img_label sofa \
  --mask_thresh 0.6
```

Use `--dr-splat-pq-index PATH` when the checkpoint uses Dr-Splat PQ codes.
