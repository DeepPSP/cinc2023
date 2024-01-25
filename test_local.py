"""
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, "/home/wenh06/Jupyter/wenhao/workspace/torch_ecg/")
sys.path.insert(0, "/home/wenh06/Jupyter/wenhao/workspace/bib_lookup/")
tmp_data_dir = Path("/home/wenh06/Jupyter/wenhao/data/CinC2023/")

import numpy as np
import torch
from torch_ecg.utils.misc import dict_to_str, str2bool

from cfg import _BASE_DIR, ModelCfg, TrainCfg
from evaluate_model import evaluate_model
from run_model import run_model

# from train_model import train_challenge_model
from team_code import train_challenge_model
from utils.misc import func_indicator

# set_entry_test_flag(True)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if ModelCfg.torch_dtype == torch.float64:
    torch.set_default_tensor_type(torch.DoubleTensor)
    DTYPE = np.float64
else:
    DTYPE = np.float32


TASK = "classification"  # "classification" or "regression", etc.


trunc_data_folder = {limit: tmp_data_dir / f"trunc_subset_{limit}" for limit in [12, 24, 48, 72]}


@func_indicator("testing challenge entry")
def test_entry():
    # data_folder = str(tmp_data_dir / "training_subset")  # subset
    data_folder = tmp_data_dir / "training"  # full set
    train_challenge_model(str(data_folder), str(TrainCfg.model_dir), verbose=2)

    output_dir = _BASE_DIR / "tmp" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("run model for the original data")

    run_model(
        str(TrainCfg.model_dir),
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
            str(TrainCfg.model_dir),
            str(trunc_data_folder[limit]),
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
        ) = evaluate_model(str(trunc_data_folder[limit]), str(output_dir))
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


if __name__ == "__main__":
    TEST_FLAG = os.environ.get("CINC2023_REVENGER_TEST", False)
    TEST_FLAG = str2bool(TEST_FLAG)
    if not TEST_FLAG:
        raise RuntimeError(
            "please set CINC2023_REVENGER_TEST to true (1, y, yes, true, etc.) to run the test, e.g."
            "\n        CINC2023_REVENGER_TEST=1 python test_local.py   "
        )
    # set_entry_test_flag(True)
    test_entry()
    # set_entry_test_flag(False)
