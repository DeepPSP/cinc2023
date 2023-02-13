"""
"""

import numpy as np
import torch

from cfg import TrainCfg, ModelCfg, _BASE_DIR  # noqa: F401
from data_reader import CINC2023Reader


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if ModelCfg.torch_dtype == torch.float64:
    torch.set_default_tensor_type(torch.DoubleTensor)
    DTYPE = np.float64
else:
    DTYPE = np.float32

tmp_data_dir = _BASE_DIR / "tmp" / "CINC2022"
tmp_data_dir.mkdir(parents=True, exist_ok=True)
dr = CINC2023Reader(tmp_data_dir)
dr.download(full=False)
dr._ls_rec()
del dr


if __name__ == "__main__":
    pass
