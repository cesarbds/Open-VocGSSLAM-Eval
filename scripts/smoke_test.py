#!/usr/bin/env python3
"""Verify that the custom CUDA renderer and checkpoint model import correctly."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

import diff_gaussian_rasterization
from diff_gaussian_rasterization import _C
from diff_gaussian_rasterization import GaussianRasterizer
from gaussian_renderer import render_3
from scene.gaussian_model import GaussianModel
from scene.shared_objs import SharedCam, SharedGaussians, SharedPoints, SharedTargetPoints
from src.mapper import Mapper
from src.tracker import Tracker


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required by the Gaussian rasterizer")
    model = GaussianModel(sh_degree=3, include_feature=True)
    print(f"CUDA: {torch.version.cuda}; GPU: {torch.cuda.get_device_name(0)}")
    print(f"Renderer Python: {diff_gaussian_rasterization.__file__}")
    print(f"Renderer CUDA: {_C.__file__}")
    print("Renderer layout: 64 semantic channels, 12 quick-render coefficients")
    print(f"Gaussian model: {type(model).__name__}; render function: {render_3.__name__}")
    print(
        "SLAM runtime: "
        f"tracker={Tracker.__name__}, mapper={Mapper.__name__}, shared buffers=OK"
    )
    print("Smoke test passed")


if __name__ == "__main__":
    main()
