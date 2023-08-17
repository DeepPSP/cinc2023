#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries and functions. You can change or remove them.
#
################################################################################

import os
import pickle
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Union, Tuple, Sequence

import numpy as np
import torch
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from torch.nn.parallel import (  # noqa: F401
    DistributedDataParallel as DDP,
    DataParallel as DP,
)  # noqa: F401
from torch_ecg.cfg import CFG
from torch_ecg.utils.misc import str2bool
from torch_ecg._preprocessors import PreprocManager
from tqdm.auto import tqdm

from cfg import TrainCfg, ModelCfg
from dataset import CinC2023Dataset
from models import CRNN_CINC2023
from trainer import CINC2023Trainer, _set_task
from helper_code import (  # noqa: F401
    find_data_folders,
    reduce_channels,
    expand_channels,
)  # noqa: F401
from utils.features import get_features, get_labels
from utils.misc import (
    load_challenge_metadata,
    load_challenge_eeg_data,
    find_eeg_recording_files,
)


################################################################################
# environment variables

try:
    TEST_FLAG = os.environ.get("CINC2023_REVENGER_TEST", False)
    TEST_FLAG = str2bool(TEST_FLAG)
except Exception:
    TEST_FLAG = False

################################################################################


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

EEG_CHANNEL_PAIRS = [
    [pair.split("-")[0] for pair in TrainCfg.eeg_channel_pairs],
    [pair.split("-")[1] for pair in TrainCfg.eeg_channel_pairs],
]
################################################################################


################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################


# Train your model.
def train_challenge_model(data_folder: str, model_folder: str, verbose: int) -> None:
    """Train models for the CinC2023 challenge.

    Parameters
    ----------
    data_folder : str
        path to the folder containing the training data
    model_folder : str
        path to the folder to save the trained model
    verbose : int
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
        print("Training the neural network model on the Challenge data...")

    ###############################################################################
    # Train the model.
    ###############################################################################
    # general configs and logger
    train_config = deepcopy(TrainCfg)
    train_config.db_dir = Path(data_folder).resolve().absolute()
    train_config.model_dir = Path(model_folder).resolve().absolute()
    train_config.debug = False

    # NOTE: for CinC2023, the data folder (db_dir) is read-only
    # the workaround is writing to the model folder
    # which is set to be the working directory (working_dir)
    train_config.working_dir = train_config.model_dir / "working_dir"
    train_config.working_dir.mkdir(parents=True, exist_ok=True)

    # if train_config.get("entry_test_flag", False):
    if TEST_FLAG:
        # to test in the file test_docker.py or in test_local.py
        train_config.n_epochs = 2
        train_config.batch_size = 8
        train_config.reload_data_every = 1
        train_config[TASK].input_len = 100 * train_config[TASK].fs
        train_config[TASK].siglen = train_config[TASK].input_len
        train_config.log_step = 20
        # train_config.max_lr = 1.5e-3
        train_config.early_stopping.patience = 20
    else:
        train_config.n_epochs = 60
        train_config.batch_size = 24  # 16G (Tesla T4)
        train_config.reload_data_every = 6
        # train_config[TASK].input_len = 180 * train_config[TASK].fs
        # train_config[TASK].siglen = train_config[TASK].input_len
        train_config.log_step = 50
        # train_config.max_lr = 1.5e-3
        train_config.early_stopping.patience = int(train_config.n_epochs * 0.35)

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
    # TODO: replace with grid searched model.
    ###############################################################################
    # Extract the features and labels.
    if verbose >= 1:
        print("Extracting features and labels from the Challenge data...")

    features = list()
    outcomes = list()
    cpcs = list()

    # for i in range(num_patients):
    #     if verbose >= 2:
    #         print("    {}/{}...".format(i + 1, num_patients))
    for i in tqdm(
        range(num_patients),
        desc="Extracting features and labels",
        total=num_patients,
        dynamic_ncols=True,
        mininterval=1.0,
        disable=verbose < 2,
    ):
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
        print("Training the RF models on the Challenge data...")

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
def load_challenge_models(
    model_folder: str, verbose: int
) -> Dict[str, Union[CFG, torch.nn.Module, BaseEstimator]]:
    """Load trained models.

    Parameters
    ----------
    model_folder : str
        path to the folder containing the trained model
    verbose : int
        verbosity level

    Returns
    -------
    dict
        with items:
            - main_model: torch.nn.Module,
              the loaded model, for cpc predictions,
              or for both cpc and outcome predictions.
            - train_cfg: CFG,
              the training configuration,
              including the list of classes (the ordering is important),
              and the preprocessing configurations.
            - outcome_model: BaseEstimator,
              the loaded model, for outcome predictions.
            - cpc_model: BaseEstimator,
              the loaded model, for cpc predictions.

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
) -> Tuple[int, float, float]:
    """Run trained models.

    Parameters
    ----------
    models : dict
        with items:
            - main_model: torch.nn.Module,
              the loaded model, for cpc predictions,
              or for both cpc and outcome predictions.
            - train_cfg: CFG,
              the training configuration,
              including the list of classes (the ordering is important),
              and the preprocessing configurations.
            - outcome_model: BaseEstimator,
              the loaded model, for outcome predictions.
            - cpc_model: BaseEstimator,
              the loaded model, for cpc predictions.
    data_folder : str
        path to the folder containing the Challenge data
    patient_id : str
        the patient ID
    verbose : int
        verbosity level

    Returns
    -------
    outcome : int
        the predicted outcome
    outcome_probability : float
        the predicted outcome probability
    cpc : float
        the predicted cpc

    """
    imputer = models["imputer"]
    outcome_model = models["outcome_model"]
    cpc_model = models["cpc_model"]

    main_model = models["main_model"]
    main_model = main_model.to(device=DEVICE).eval()
    train_cfg = models["train_cfg"]
    ppm_config = CFG(random=False)
    ppm_config.update(deepcopy(train_cfg[TASK]))
    ppm = PreprocManager.from_config(ppm_config)

    # Load data.
    # patient_metadata: str
    patient_metadata = load_challenge_metadata(data_folder, patient_id)
    recording_files = find_eeg_recording_files(data_folder, patient_id)
    num_recordings = len(recording_files)

    if verbose >= 1:
        print(f"Found {num_recordings} recordings for patient {patient_id}.")

    if num_recordings == 0:
        # No available recordings.
        # use the outcome_model and cpc_model to predict the outcome and cpc
        features = get_features(patient_metadata).reshape(1, -1)
        features = imputer.transform(features)
        outcome = outcome_model.predict(features)[0].item()
        # outcome_probability is the probability of the "Poor" (1) class
        outcome_probability = outcome_model.predict_proba(features)[0, 1].item()
        cpc = cpc_model.predict(features)[0].item()

        return outcome, outcome_probability, cpc

    # There are available recordings.

    # recording_data: list of 3-tuples (signal, sampling_frequency, channel_names)
    recording_data = load_challenge_eeg_data(data_folder, patient_id)
    # find recordings whose channels are a superset of the common channels
    valid_indices = [
        idx
        for idx, (_, _, channel_names) in enumerate(recording_data)
        if set(TrainCfg.common_eeg_channels).issubset(channel_names)
    ]
    # use only the valid recordings if there are any
    # otherwise use all recordings
    if len(valid_indices) > 0:
        recording_data = [recording_data[idx] for idx in valid_indices]
    main_model_outputs = []
    for signal, sampling_frequency, channel_names in recording_data:
        # Preprocess the recordings.
        # to correct dtype if necessary
        signal = _to_dtype(signal, DTYPE)
        # to channel first format if necessary
        if signal.shape[0] != len(channel_names):
            signal = signal.T
        signal = format_input_signal(signal, channel_names)
        # preprocess the signal
        signal, _ = ppm(signal, sampling_frequency)

        # Run the model.
        with torch.no_grad():
            main_model_output = main_model.inference(signal.copy().astype(DTYPE))
            main_model_outputs.append(main_model_output)

    # Aggregate the outputs.
    # main_model_outputs is a list of instances of CINC2023Outputs, with attributes:
    # - cpc_output, outcome_output: ClassificationOutput, with items:
    #     - classes: list of str,
    #         list of the class names
    #     - prob: ndarray or DataFrame,
    #         scalar (probability) predictions,
    #         (and binary predictions if `class_names` is True)
    #     - pred: ndarray,
    #         the array of class number predictions
    #     - bin_pred: ndarray,
    #         the array of binary predictions
    #     - forward_output: ndarray,
    #         the array of output of the model's forward function,
    #         useful for producing challenge result using
    #         multiple recordings
    # - cpc_value: List[float],
    #     the list of cpc values
    # - outcome: List[str],
    #     the list of outcome classes

    # Aggregate the CPC predictions.
    cpc_outputs = np.array(
        [output.cpc_value for output in main_model_outputs]
    ).flatten()
    cpc = cpc_outputs.mean().item()

    # Aggregate the outcome predictions.
    outcome_outputs = np.concatenate(
        [output.outcome_output.prob for output in main_model_outputs]
    )
    outcome_outputs = outcome_outputs.mean(axis=0, keepdims=True)
    # apply softmax to get the probabilities
    outcome_outputs = np.exp(outcome_outputs) / np.exp(outcome_outputs).sum(
        axis=1, keepdims=True
    )
    # get the predicted outcome
    outcome = outcome_outputs.argmax(axis=1)[0].item()
    # outcome_probability is the probability of the "Poor" class
    outcome_probability = outcome_outputs[0, train_cfg.outcome.index("Poor")].item()

    return outcome, outcome_probability, cpc


def _to_dtype(data: np.ndarray, dtype: np.dtype = np.float32) -> np.ndarray:
    """ """
    if data.dtype == dtype:
        return data
    if data.dtype in (np.int8, np.uint8, np.int16, np.int32, np.int64):
        data = data.astype(dtype) / (np.iinfo(data.dtype).max + 1)
    return data


def format_input_signal(
    raw_signal: np.ndarray, input_channels: Sequence[str]
) -> np.ndarray:
    """Format the input signal via forming the single-channel raw
    signal to bipolar signal.

    Parameters
    ----------
    raw_signal : np.ndarray
        the raw signal, of shape ``(n_channels, n_samples)``.
    input_channels : Sequence[str]
        the list of input channels.

    Returns
    -------
    np.ndarray
        the formatted signal, of shape ``(n_channels, n_samples)``

    """
    missing_channels = list(set(TrainCfg.common_eeg_channels) - set(input_channels))
    if len(missing_channels) > 0:
        # add missing channels
        raw_signal = np.concatenate(
            [raw_signal, np.zeros((len(missing_channels), raw_signal.shape[1]))]
        )
        input_channels = list(input_channels) + missing_channels

    diff_inds = [
        [input_channels.index(item) for item in lst] for lst in EEG_CHANNEL_PAIRS
    ]
    signal = raw_signal[diff_inds[0]] - raw_signal[diff_inds[1]]
    return signal
