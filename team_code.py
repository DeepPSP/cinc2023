#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries and functions. You can change or remove them.
#
################################################################################

import pickle
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
import torch
from torch.nn.parallel import (  # noqa: F401
    DistributedDataParallel as DDP,
    DataParallel as DP,
)  # noqa: F401
from torch_ecg.cfg import CFG
from torch_ecg._preprocessors import PreprocManager

from cfg import TrainCfg, ModelCfg
from dataset import CinC2023Dataset
from models import (
    CRNN_CINC2023,
)
from trainer import (  # noqa: F401
    CINC2023Trainer,
    _set_task,
)  # noqa: F401
from helper_code import (
    find_data_folders,
    load_challenge_data,
)
from utils.features import get_features, get_labels, load_challenge_metadata


################################################################################
# NOTE: configurable options

TASK = "classification"  # "classification", "regression"

# choices of the models
TrainCfg[TASK].model_name = "crnn"

# "tresnetS"  # "resnet_nature_comm", "tresnetF", etc.
TrainCfg[TASK].cnn_name = "resnet_nature_comm_bottle_neck_se"

# TrainCfg[TASK].rnn_name = "none"  # "none", "lstm"
# TrainCfg[TASK].attn_name = "se"  # "none", "se", "gc", "nl"
################################################################################


################################################################################
# NOTE: constants

FS = 100

_ModelFilename = "final_model_main.pth.tar"
_ModelFilename_ml = "final_model_ml.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if ModelCfg.torch_dtype == torch.float64:
    torch.set_default_tensor_type(torch.DoubleTensor)
    DTYPE = np.float64
else:
    DTYPE = np.float32

CinC2023Dataset.__DEBUG__ = False
CRNN_CINC2023.__DEBUG__ = False
CINC2023Trainer.__DEBUG__ = False
################################################################################


################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################


# Train your model.
def train_challenge_model(data_folder: str, model_folder: str, verbose: int) -> None:
    """

    Parameters
    ----------
    data_folder: str,
        path to the folder containing the training data
    model_folder: str,
        path to the folder to save the trained model
    verbose: int,
        verbosity level

    """
    print("\n" + "*" * 100)
    msg = "   CinC2023 challenge training entry starts   ".center(100, "#")
    print(msg)
    print("*" * 100 + "\n")

    # Find data files.
    if verbose >= 1:
        print("Finding the Challenge data...")

    patient_ids = find_data_folders(data_folder)
    num_patients = len(patient_ids)

    if num_patients == 0:
        raise FileNotFoundError("No data was provided.")
    else:
        if verbose >= 1:
            print(f"Found {num_patients} patients.")

    # Create a folder for the model if it does not already exist.
    # os.makedirs(model_folder, exist_ok=True)
    Path(model_folder).mkdir(parents=True, exist_ok=True)

    # Train the models.
    if verbose >= 1:
        print("Training the Challenge models on the Challenge data...")

    ###############################################################################
    # Train the model.
    ###############################################################################
    # general configs and logger
    train_config = deepcopy(TrainCfg)
    train_config.db_dir = Path(data_folder).resolve().absolute()
    train_config.model_dir = Path(model_folder).resolve().absolute()
    train_config.debug = False

    if train_config.get("entry_test_flag", False):
        # to test in the file test_docker.py or in test_local.py
        train_config.n_epochs = 1
        train_config.batch_size = 4
        train_config.log_step = 4
        # train_config.max_lr = 1.5e-3
        train_config.early_stopping.patience = 20
    else:
        train_config.n_epochs = 60
        train_config.freeze_backbone_at = 40
        train_config.batch_size = 32  # 16G (Tesla T4)
        train_config.log_step = 50
        # train_config.max_lr = 1.5e-3
        train_config.early_stopping.patience = int(train_config.n_epochs * 0.6)

    train_config.final_model_name = _ModelFilename
    train_config[TASK].final_model_name = _ModelFilename
    _set_task(TASK, train_config)

    model_config = deepcopy(ModelCfg[TASK])

    # adjust model choices if needed
    model_name = model_config.model_name = train_config[TASK].model_name
    if "cnn" in model_config[model_name]:
        model_config[model_name].cnn.name = train_config[TASK].cnn_name
    if "rnn" in model_config[model_name]:
        model_config[model_name].rnn.name = train_config[TASK].rnn_name
    if "attn" in model_config[model_name]:
        model_config[model_name].attn.name = train_config[TASK].attn_name

    start_time = time.time()

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

    best_state_dict = trainer.train()  # including saving model

    del trainer
    del model
    del best_state_dict

    torch.cuda.empty_cache()

    ###############################################################################
    # Train ML model using patient metadata.
    ###############################################################################
    # Extract the features and labels.
    if verbose >= 1:
        print("Extracting features and labels from the Challenge data...")

    features = list()
    outcomes = list()
    cpcs = list()

    for i in range(num_patients):
        if verbose >= 2:
            print("    {}/{}...".format(i + 1, num_patients))

        # Load data.
        patient_id = patient_ids[i]
        patient_metadata = load_challenge_metadata(data_folder, patient_id)

        # Extract features.
        current_features = get_features(patient_metadata)
        features.append(current_features)

        # Extract labels.
        current_labels = get_labels(patient_metadata)
        outcomes.append(current_labels["outcome"])
        cpcs.append(current_labels["cpc"])

    features = np.vstack(features)
    outcomes = np.vstack(outcomes)
    cpcs = np.vstack(cpcs)

    # Train the models.
    if verbose >= 1:
        print("Training the Challenge models on the Challenge data...")

    # Define parameters for random forest classifier and regressor.
    n_estimators = 42  # Number of trees in the forest.
    max_leaf_nodes = 456  # Maximum number of leaf nodes in each tree.
    random_state = 789  # Random state; set for reproducibility.

    # Impute any missing features; use the mean value by default.
    imputer = SimpleImputer().fit(features)

    # Train the models.
    features = imputer.transform(features)
    outcome_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_leaf_nodes=max_leaf_nodes,
        random_state=random_state,
    ).fit(features, outcomes.ravel())
    cpc_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_leaf_nodes=max_leaf_nodes,
        random_state=random_state,
    ).fit(features, cpcs.ravel())

    d = {"imputer": imputer, "outcome_model": outcome_model, "cpc_model": cpc_model}
    model_path = Path(model_folder).resolve().absolute() / _ModelFilename_ml
    with open(model_path, "wb") as f:
        pickle.dump(d, f)

    if verbose >= 1:
        print("Done.")


# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_model(
    model_folder: str, verbose: int
) -> Dict[str, Union[CFG, torch.nn.Module, BaseEstimator]]:
    """

    Parameters
    ----------
    model_folder: str,
        path to the folder containing the trained model
    verbose: int,
        verbosity level

    Returns
    -------
    dict
        with items:
        - main_model: torch.nn.Module,
            the loaded model, for murmur predictions,
            or for both murmur and outcome predictions
        - train_cfg: CFG,
            the training configuration,
            including the list of classes (the ordering is important),
            and the preprocessing configurations
        - outcome_model: BaseEstimator,
            the loaded model, for outcome predictions
        - cpc_model: BaseEstimator,
            the loaded model, for cpc predictions

    """
    print("\n" + "*" * 100)
    msg = "   loading CinC2023 challenge model   ".center(100, "#")
    print(msg)

    model_cls = CRNN_CINC2023
    main_model, train_cfg = model_cls.from_checkpoint(
        path=Path(model_folder).resolve().absolute() / _ModelFilename,
        device=DEVICE,
    )
    main_model.eval()

    ml_model_path = Path(model_folder).resolve().absolute() / _ModelFilename_ml
    with open(ml_model_path, "rb") as f:
        ml_models = pickle.load(f)

    msg = "   CinC2023 challenge model loaded   ".center(100, "#")
    print(msg)
    print("*" * 100 + "\n")

    models = dict(main_model=main_model, train_cfg=train_cfg, **ml_models)

    return models


# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(
    models: Dict[str, Union[CFG, torch.nn.Module, BaseEstimator]],
    data_folder: str,
    patient_id: str,
    verbose: int,
):
    imputer = models["imputer"]
    outcome_model = models["outcome_model"]
    cpc_model = models["cpc_model"]

    main_model = models["main_model"]
    main_model.to(device=DEVICE)
    train_cfg = models["train_cfg"]
    ppm_config = CFG(random=False)
    ppm_config.update(deepcopy(train_cfg[TASK]))
    ppm = PreprocManager.from_config(ppm_config)

    # Load data.
    patient_metadata, recording_metadata, recording_data = load_challenge_data(
        data_folder, patient_id
    )

    # TODO

    # return outcome, outcome_probability, cpc
