"""
"""

import sys
from pathlib import Path

sys.path.insert(0, "/home/wenhao/Jupyter/wenhao/workspace/torch_ecg/")
sys.path.insert(0, "/home/wenhao/Jupyter/wenhao/workspace/bib_lookup/")
tmp_data_dir = Path("/home/wenh06/Jupyter/wenhao/data/CinC2023/")

import numpy as np
import torch

from cfg import TrainCfg, ModelCfg, _BASE_DIR  # noqa: F401
from train_model import train_challenge_model
from run_model import run_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if ModelCfg.torch_dtype == torch.float64:
    torch.set_default_tensor_type(torch.DoubleTensor)
    DTYPE = np.float64
else:
    DTYPE = np.float32

TASK = "classification"  # "multi_task"


def test_entry():

    data_folder = str(tmp_data_dir / "training_subset")  # subset
    # data_folder = str(tmp_data_dir / "training")  # full set
    train_challenge_model(data_folder, str(TrainCfg.model_dir), verbose=2)

    output_dir = _BASE_DIR / "tmp" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_model(
        TrainCfg.model_dir,
        data_folder,
        str(output_dir),
        allow_failures=False,
        verbose=2,
    )

    print("entry test passed")


if __name__ == "__main__":
    pass
