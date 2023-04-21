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
from torch_ecg.utils.misc import dict_to_str
from torch_ecg.components.outputs import (
    ClassificationOutput,
)

from utils.scoring_metrics import compute_challenge_metrics
from cfg import TrainCfg, ModelCfg, BaseCfg, _BASE_DIR
from data_reader import CINC2023Reader
from dataset import CinC2023Dataset
from outputs import CINC2023Outputs, cpc2outcome_map
from models import CRNN_CINC2023
from trainer import CINC2023Trainer, _set_task


CINC2023Trainer.__DEBUG__ = False
CRNN_CINC2023.__DEBUG__ = False
CinC2023Dataset.__DEBUG__ = False


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
    # random prediction
    cpc_probs = np.random.rand(100, len(BaseCfg.cpc))
    cpc_probs = np.exp(cpc_probs) / np.sum(np.exp(cpc_probs), axis=1, keepdims=True)
    cpc_preds = np.argmax(cpc_probs, axis=1)

    outputs = CINC2023Outputs(
        cpc_output=ClassificationOutput(
            classes=BaseCfg.cpc,
            prob=cpc_probs,
            pred=cpc_preds,
        ),
    )

    # random ground truth
    cpc_gt = np.random.randint(1, len(BaseCfg.cpc) + 1, size=100)
    outcome_gt = np.array(
        [cpc2outcome_map[BaseCfg.cpc_map[str(cpc)]] for cpc in cpc_gt]
    )
    labels = {
        "cpc": cpc_gt,
        "outcome": outcome_gt,
    }

    metrics = compute_challenge_metrics([labels], [outputs])

    print(dict_to_str(metrics))

    print("challenge metrics test passed")


def test_trainer() -> None:
    """ """
    train_config = deepcopy(TrainCfg)
    train_config.db_dir = tmp_data_dir
    # train_config.model_dir = model_folder
    # train_config.final_model_filename = "final_model.pth.tar"
    train_config.debug = True

    train_config.n_epochs = 5
    train_config.batch_size = 8  # 16G (Tesla T4)
    # train_config.log_step = 20
    # # train_config.max_lr = 1.5e-3
    # train_config.early_stopping.patience = 20

    train_config[TASK].cnn_name = "resnet_nature_comm_bottle_neck_se"
    # train_config[TASK].rnn_name = "none"  # "none", "lstm"
    # train_config[TASK].attn_name = "se"  # "none", "se", "gc", "nl"

    _set_task(TASK, train_config)

    model_config = deepcopy(ModelCfg[TASK])

    # adjust model choices if needed
    model_name = model_config.model_name = train_config[TASK].model_name
    model_config[model_name].cnn_name = train_config[TASK].cnn_name
    model_config[model_name].rnn_name = train_config[TASK].rnn_name
    model_config[model_name].attn_name = train_config[TASK].attn_name

    model_cls = CRNN_CINC2023
    model = model_cls(config=model_config)
    if torch.cuda.device_count() > 1:
        model = DP(model)
        # model = DDP(model)
    model.to(device=DEVICE)

    trainer = CINC2023Trainer(
        model=model,
        model_config=model_config,
        train_config=train_config,
        device=DEVICE,
        lazy=False,
    )

    best_state_dict = trainer.train()

    print("trainer test passed")


# from train_model import train_challenge_model
from team_code import train_challenge_model
from run_model import run_model


def test_entry() -> None:
    """ """

    data_folder = str(tmp_data_dir)
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


test_team_code = test_entry  # alias


if __name__ == "__main__":
    # test_dataset()
    # test_models()
    test_challenge_metrics()
    test_trainer()  # directly run test_entry
    # test_entry()
