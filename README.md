# Open-VocGSSLAM Evaluation

Portable Open-VocGSSLAM runtime and evaluation package. It contains the ICP
tracker, Gaussian mapper, semantic keyframe extraction, shared multiprocessing
objects, custom RGB/depth/semantic CUDA rasterizer, checkpoint loader, and
evaluation scripts. Datasets, model checkpoints, ROS drivers, and results are
not included.

## Requirements

- Linux and an NVIDIA CUDA GPU
- CUDA toolkit compatible with the installed PyTorch build
- A C++ compiler and `nvcc`
- Python 3.9 or 3.10

The reference environment uses PyTorch 2.5.1, torchvision 0.20.1, and CUDA 12.1.
The machine must have a compatible NVIDIA driver. CUDA 12.1 and `nvcc` are
installed inside the Conda environment, so the system CUDA installation is not
modified.

Create the complete Conda environment and install the extensions with:

```bash
git clone <repository-url>
cd Open-VocGSSLAM-Eval
conda env create -f environment.yml
conda activate openvocgsslam-eval
source env_vars.sh
./install.sh
python scripts/smoke_test.py
```

Before running `install.sh`, verify the isolated toolchain:

```bash
nvidia-smi
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

Run `source env_vars.sh` after each new shell activation, before compiling the
extensions. It sets `CUDA_HOME` to the active Conda environment rather than the
computer's system CUDA directory.

Do not install another `diff-gaussian-rasterization` implementation into the
same environment. This repository includes the customized rasterizer required
by `render_3`.

If evaluation reports that a CUDA function expected a different number of
arguments, an incompatible precompiled rasterizer is being imported. Re-run
`./install.sh`, then use `python scripts/smoke_test.py` to verify that both the
Python module and `_C` binary paths belong to the active Conda environment.

## Run live tracking and mapping

The runtime consumes synchronized ZED RGB-D messages directly and does not
preload a dataset into memory. Source ROS 2 and your ZED workspace, start the
ZED driver, then run:

```bash
python main.py \
  --ros-rgb-topic /zed/zed_node/rgb/color/rect/image \
  --ros-depth-topic /zed/zed_node/depth/depth_registered \
  --ros-camera-info-topic /zed/zed_node/rgb/color/rect/camera_info \
  --save-path saved_results/zed_live \
  --no-include-feature \
  --pruning-mode simple
```

The runtime orchestration is in `src/OSGSSLAM.py`: `Tracker` and `Mapper` run
as separate processes and exchange cameras and Gaussian targets through the
objects in `scene/shared_objs.py`.

Rerun visualization is enabled by default and shows the live camera, tracked
point clouds, and mapper renders. Pass `--no-rerun-viewer` for a headless run.

The RGB and registered-depth streams are approximately synchronized (30 ms by
default), and intrinsics are read from `CameraInfo`. Use `--max-frames N` for a
bounded run; the default of zero runs until ROS shutdown or Ctrl-C. The tracker
publishes accepted insertion keyframes to the mapper; semantics remain in the
mapper so feature extraction does not delay pose tracking.

Runtime assets are intentionally excluded from Git. Supply the paths required
by the selected mode, including the language codebook and segmentation model
checkpoint when semantic mapping is enabled.

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
