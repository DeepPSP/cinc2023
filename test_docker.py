"""
"""

import os
from copy import deepcopy
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch.nn.parallel import DataParallel as DP  # DistributedDataParallel as DDP,
from torch.utils.data import DataLoader
from torch_ecg.cfg import CFG
from torch_ecg.components.outputs import ClassificationOutput
from torch_ecg.utils.misc import dict_to_str, str2bool
from torch_ecg.utils.utils_nn import default_collate_fn as collate_fn

from cfg import _BASE_DIR, BaseCfg, ModelCfg, TrainCfg
from data_reader import CINC2023Reader
from dataset import CinC2023Dataset
from models import CRNN_CINC2023
from outputs import CINC2023Outputs, cpc2outcome_map
from trainer import CINC2023Trainer, _set_task
from truncate_data import run as truncate_data_run
from utils.misc import func_indicator
from utils.scoring_metrics import compute_challenge_metrics

# set_entry_test_flag(True)


CINC2023Trainer.__DEBUG__ = False
CRNN_CINC2023.__DEBUG__ = False
CinC2023Dataset.__DEBUG__ = False


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if ModelCfg.torch_dtype == torch.float64:
    torch.set_default_tensor_type(torch.DoubleTensor)
    DTYPE = np.float64
else:
    DTYPE = np.float32


tmp_data_dir = Path(os.environ.get("revenger_data_dir", _BASE_DIR / "tmp" / "CINC2023")).resolve()
print(f"tmp_data_dir: {str(tmp_data_dir)}")
tmp_data_dir.mkdir(parents=True, exist_ok=True)

dr = CINC2023Reader(tmp_data_dir)
# downloading is done outside the docker container
# and the data folder is mounted to the docker container as read-only
# dr.download(full=False)
# dr._ls_rec()

# let's remove cached metadata files
# to test the CinC2023Dataset at the very first run
# when metadata files are not computed and cached yet,
# which is exactly the case for the challenge
dr.clear_cached_metadata_files()

# adjusts tmp_data_dir to the true data folder that
# contains the data directly
# rather than the possible parent folder
tmp_data_dir = dr._df_records.path.iloc[0].parents[1]

# truncate data
truncate_cfg = CFG(
    input_folder=str(dr._df_records.path.iloc[0].parents[1]),
)


del dr


tmp_model_dir = Path(os.environ.get("revenger_model_dir", TrainCfg.model_dir)).resolve()

truncated_data_dir = {}
for limit in [12, 24, 48, 72]:
    truncate_cfg.time_limit = limit
    truncate_cfg.patient_ids = []
    truncated_data_dir[limit] = str(tmp_model_dir / "truncated_data" / f"CINC2023_{limit}h")
    truncate_cfg.output_folder = truncated_data_dir[limit]
    truncate_data_run(truncate_cfg)

tmp_output_dir = Path(os.environ.get("revenger_output_dir", _BASE_DIR / "tmp" / "output")).resolve()


TASK = "classification"


# make tmp_data_dir read-only recursively
# as the data folder of CinC2023 is read-only
# os.system(f"chmod -R 555 {str(tmp_data_dir)}")

# os.chmod(str(tmp_data_dir), 0o555)


def echo_write_permission(folder: Union[str, Path]) -> None:
    """ """
    is_writeable = "is writable" if os.access(str(folder), os.W_OK) else "is not writable"
    print(f"{str(folder)} {is_writeable}")


echo_write_permission(tmp_data_dir)


@func_indicator("testing dataset")
def test_dataset() -> None:
    """ """
    ds_config = deepcopy(TrainCfg)
    ds_config.db_dir = tmp_data_dir
    ds_config.working_dir = tmp_model_dir / "working_dir"
    ds_config.working_dir.mkdir(parents=True, exist_ok=True)

    ds_train = CinC2023Dataset(ds_config, TASK, training=True, lazy=True)
    ds_val = CinC2023Dataset(ds_config, TASK, training=False, lazy=True)

    ds_train._load_all_data()
    ds_val._load_all_data()

    print("dataset test passed")


@func_indicator("testing models")
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


@func_indicator("testing challenge metrics")
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
    outcome_gt = np.array([cpc2outcome_map[BaseCfg.cpc_map[str(cpc)]] for cpc in cpc_gt])
    labels = {
        "cpc": cpc_gt,
        "outcome": outcome_gt,
    }

    hospitals = np.random.choice(["A", "B", "C", "D", "E", "F", "G"], size=100)
    metrics = compute_challenge_metrics([labels], [outputs], [hospitals])
    print(dict_to_str(metrics))

    hospitals = np.random.choice(range(7), size=100)
    metrics = compute_challenge_metrics([labels], [outputs], [hospitals])
    print(dict_to_str(metrics))

    print("challenge metrics test passed")


@func_indicator("testing trainer")
def test_trainer() -> None:
    """ """
    echo_write_permission(tmp_data_dir)
    echo_write_permission(tmp_model_dir)

    train_config = deepcopy(TrainCfg)
    train_config.db_dir = tmp_data_dir
    # train_config.model_dir = model_folder
    # train_config.final_model_filename = "final_model.pth.tar"
    train_config.debug = True
    train_config.working_dir = tmp_model_dir / "working_dir"
    train_config.working_dir.mkdir(parents=True, exist_ok=True)

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


from evaluate_model import evaluate_model
from run_model import run_model

# from train_model import train_challenge_model
from team_code import train_challenge_model


@func_indicator("testing challenge entry")
def test_entry() -> None:
    """ """
    echo_write_permission(tmp_data_dir)
    echo_write_permission(tmp_model_dir)

    # run the model training function (script)
    print("run model training function")
    data_folder = tmp_data_dir
    train_challenge_model(str(data_folder), str(tmp_model_dir), verbose=2)

    # run the model inference function (script)
    output_dir = tmp_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("run model for the original data")

    run_model(
        str(tmp_model_dir),
        str(data_folder),
        str(output_dir),
        allow_failures=False,
        verbose=2,
    )

    print("evaluate model for the original data")

    (
        challenge_score,
        auroc_outcomes,
        auprc_outcomes,
        accuracy_outcomes,
        f_measure_outcomes,
        mse_cpcs,
        mae_cpcs,
    ) = evaluate_model(str(data_folder), str(output_dir))
    eval_res = {
        "challenge_score": challenge_score,
        "auroc_outcomes": auroc_outcomes,
        "auprc_outcomes": auprc_outcomes,
        "accuracy_outcomes": accuracy_outcomes,
        "f_measure_outcomes": f_measure_outcomes,
        "mse_cpcs": mse_cpcs,
        "mae_cpcs": mae_cpcs,
    }

    print(f"original data evaluation results: {dict_to_str(eval_res)}")

    for limit in [12, 24, 48, 72]:
        print(f"run model for the {limit}h data")
        run_model(
            str(tmp_model_dir),
            str(truncated_data_dir[limit]),
            str(output_dir),
            allow_failures=False,
            verbose=2,
        )

        print(f"evaluate model for the {limit}h data")

        (
            challenge_score,
            auroc_outcomes,
            auprc_outcomes,
            accuracy_outcomes,
            f_measure_outcomes,
            mse_cpcs,
            mae_cpcs,
        ) = evaluate_model(str(truncated_data_dir[limit]), str(output_dir))
        eval_res = {
            "challenge_score": challenge_score,
            "auroc_outcomes": auroc_outcomes,
            "auprc_outcomes": auprc_outcomes,
            "accuracy_outcomes": accuracy_outcomes,
            "f_measure_outcomes": f_measure_outcomes,
            "mse_cpcs": mse_cpcs,
            "mae_cpcs": mae_cpcs,
        }

        print(f"{limit}h data evaluation results: {dict_to_str(eval_res)}")

    print("entry test passed")


test_team_code = test_entry  # alias


if __name__ == "__main__":
    TEST_FLAG = os.environ.get("CINC2023_REVENGER_TEST", False)
    TEST_FLAG = str2bool(TEST_FLAG)
    if not TEST_FLAG:
        # raise RuntimeError(
        #     "please set CINC2023_REVENGER_TEST to true (1, y, yes, true, etc.) to run the test"
        # )
        print("Test is skipped.")
        print("Please set CINC2023_REVENGER_TEST to true (1, y, yes, true, etc.) to run the test")
        exit(0)

    print("#" * 80)
    print("testing team code")
    print("#" * 80)
    print(f"tmp_data_dir: {str(tmp_data_dir)}")
    print(f"tmp_model_dir: {str(tmp_model_dir)}")
    print(f"tmp_output_dir: {str(tmp_output_dir)}")
    print("#" * 80)

    # test_dataset()
    # test_models()
    test_challenge_metrics()
    # test_trainer()  # directly run test_entry
    test_entry()
    # set_entry_test_flag(False)
