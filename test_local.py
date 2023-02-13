"""
"""

import sys
from pathlib import Path

sys.path.insert(0, "/home/wenhao/Jupyter/wenhao/workspace/torch_ecg/")
sys.path.insert(0, "/home/wenhao/Jupyter/wenhao/workspace/bib_lookup/")
tmp_data_dir = Path("/home/wenhao/Jupyter/wenhao/data/CinC2023/training_subset/")

import numpy as np
import torch

from cfg import TrainCfg, ModelCfg, _BASE_DIR  # noqa: F401


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if ModelCfg.torch_dtype == torch.float64:
    torch.set_default_tensor_type(torch.DoubleTensor)
    DTYPE = np.float64
else:
    DTYPE = np.float32


if __name__ == "__main__":
    pass
