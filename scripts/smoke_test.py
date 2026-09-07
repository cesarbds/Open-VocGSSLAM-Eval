#!/usr/bin/env python3
"""Verify that the custom CUDA renderer and checkpoint model import correctly."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from diff_gaussian_rasterization import GaussianRasterizer
from gaussian_renderer import render_3
from scene.gaussian_model import GaussianModel


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required by the Gaussian rasterizer")
    model = GaussianModel(sh_degree=3, include_feature=True)
    print(f"CUDA: {torch.version.cuda}; GPU: {torch.cuda.get_device_name(0)}")
    print(f"Renderer: {GaussianRasterizer.__module__}")
    print(f"Gaussian model: {type(model).__name__}; render function: {render_3.__name__}")
    print("Smoke test passed")


if __name__ == "__main__":
    main()
