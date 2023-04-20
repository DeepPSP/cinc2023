"""
"""

from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader  # noqa: F401
from torch.nn.parallel import (  # noqa: F401
    DistributedDataParallel as DDP,
    DataParallel as DP,
)  # noqa: F401
from torch_ecg.utils.utils_nn import default_collate_fn as collate_fn

from cfg import TrainCfg, ModelCfg, _BASE_DIR  # noqa: F401
from data_reader import CINC2023Reader
from dataset import CinC2023Dataset
from models import CRNN_CINC2023


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if ModelCfg.torch_dtype == torch.float64:
    torch.set_default_tensor_type(torch.DoubleTensor)
    DTYPE = np.float64
else:
    DTYPE = np.float32

tmp_data_dir = _BASE_DIR / "tmp" / "CINC2023"
tmp_data_dir.mkdir(parents=True, exist_ok=True)
dr = CINC2023Reader(tmp_data_dir)
dr.download(full=False)
dr._ls_rec()
# let's remove cached metadata files
# to test the CinC2023Dataset at the very first run
# when metadata files are not computed and cached yet,
# which is exactly the case for the challenge
dr.clear_cached_metadata_files()

del dr


TASK = "classification"


def test_dataset() -> None:
    """ """
    ds_config = deepcopy(TrainCfg)
    ds_config.db_dir = tmp_data_dir

    ds_train = CinC2023Dataset(ds_config, TASK, training=True, lazy=True)
    ds_val = CinC2023Dataset(ds_config, TASK, training=False, lazy=True)

    ds_train._load_all_data()
    ds_val._load_all_data()

    print("dataset test passed")


def test_models() -> None:
    """ """
    model = CRNN_CINC2023(ModelCfg[TASK])
    model.to(DEVICE)
    ds_config = deepcopy(TrainCfg)
    ds_config.db_dir = tmp_data_dir
    ds_val = CinC2023Dataset(ds_config, TASK, training=False, lazy=True)
    ds_val._load_all_data()
    dl = DataLoader(
        dataset=ds_val,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    for idx, input_tensors in enumerate(dl):
        waveforms = input_tensors.pop("waveforms").to(DEVICE)
        # input_tensors = {k: v.to(DEVICE) for k, v in input_tensors.items()}
        # out_tensors = model(waveforms, input_tensors)
        print(model.inference(waveforms))
        if idx > 10:
            break

    print("models test passed")


def test_challenge_metrics() -> None:
    """ """
    pass

    # print("challenge metrics test passed")


def test_trainer() -> None:
    """ """
    train_config = deepcopy(TrainCfg)
    train_config.db_dir = tmp_data_dir
    # train_config.model_dir = model_folder
    # train_config.final_model_filename = "final_model.pth.tar"
    train_config.debug = True

    train_config.n_epochs = 20
    train_config.batch_size = 24  # 16G (Tesla T4)
    # train_config.log_step = 20
    # # train_config.max_lr = 1.5e-3
    # train_config.early_stopping.patience = 20

    # train_config[TASK].cnn_name = "resnet_nature_comm_bottle_neck_se"
    # train_config[TASK].rnn_name = "none"  # "none", "lstm"
    # train_config[TASK].attn_name = "se"  # "none", "se", "gc", "nl"

    # NOT finished

    # print("trainer test passed")


# from train_model import train_challenge_model
# from run_model import run_model


def test_entry() -> None:
    """ """

    pass

    # print("entry test passed")


test_team_code = test_entry  # alias


if __name__ == "__main__":
    test_dataset()
    test_models()
