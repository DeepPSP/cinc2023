"""
"""

import pathlib
from copy import deepcopy

import numpy as np
import torch
from sklearn.model_selection import ParameterGrid
from torch_ecg.cfg import CFG
from torch_ecg.utils.utils_nn import adjust_cnn_filter_lengths
from torch_ecg.components.inputs import InputConfig

from cfg_models import ModelArchCfg


__all__ = [
    "BaseCfg",
    "TrainCfg",
    "ModelCfg",
]


_BASE_DIR = pathlib.Path(__file__).absolute().parent


###############################################################################
# Base Configs,
# including path, data type, classes, etc.
###############################################################################

BaseCfg = CFG()
BaseCfg.db_dir = None
BaseCfg.project_dir = _BASE_DIR
BaseCfg.log_dir = _BASE_DIR / "log"
BaseCfg.model_dir = _BASE_DIR / "saved_models"
BaseCfg.log_dir.mkdir(exist_ok=True)
BaseCfg.model_dir.mkdir(exist_ok=True)
BaseCfg.fs = 100
BaseCfg.n_channels = 18
BaseCfg.torch_dtype = torch.float32  # "double"
BaseCfg.np_dtype = np.float32
BaseCfg.ignore_index = -100
BaseCfg.outcome = ["Good", "Poor"]
BaseCfg.outcome_map = {
    "Good": 0,
    "Poor": 1,
}
BaseCfg.cpc = [str(cpc_level) for cpc_level in range(1, 6)]
BaseCfg.cpc_map = {str(cpc_level): cpc_level - 1 for cpc_level in range(1, 6)}
BaseCfg.cpc2outcome_map = {
    "1": "Good",
    "2": "Good",
    "3": "Poor",
    "4": "Poor",
    "5": "Poor",
}
BaseCfg.output_target = "cpc"  # "cpc", "outcome"


###############################################################################
# training configurations for machine learning and deep learning
###############################################################################

TrainCfg = deepcopy(BaseCfg)

###########################################
# common configurations for all tasks
###########################################

TrainCfg.checkpoints = _BASE_DIR / "checkpoints"
TrainCfg.checkpoints.mkdir(exist_ok=True)
# TODO: add "contrastive_learning", "regression", "multi_task", etc.
TrainCfg.tasks = ["classification"]

TrainCfg.train_ratio = 0.8

# configs of training epochs, batch, etc.
TrainCfg.n_epochs = 60
# TODO: automatic adjust batch size according to GPU capacity
# https://stackoverflow.com/questions/45132809/how-to-select-batch-size-automatically-to-fit-gpu
TrainCfg.batch_size = 24

# configs of optimizers and lr_schedulers
TrainCfg.optimizer = "adamw_amsgrad"  # "sgd", "adam", "adamw"
TrainCfg.momentum = 0.949  # default values for corresponding PyTorch optimizers
TrainCfg.betas = (0.9, 0.999)  # default values for corresponding PyTorch optimizers
TrainCfg.decay = 1e-2  # default values for corresponding PyTorch optimizers

TrainCfg.learning_rate = 5e-4  # 1e-3
TrainCfg.lr = TrainCfg.learning_rate

TrainCfg.lr_scheduler = "one_cycle"  # "one_cycle", "plateau", "burn_in", "step", None
TrainCfg.lr_step_size = 50
TrainCfg.lr_gamma = 0.1
TrainCfg.max_lr = 2e-3  # for "one_cycle" scheduler, to adjust via expriments

# configs of callbacks, including early stopping, checkpoint, etc.
TrainCfg.early_stopping = CFG()  # early stopping according to challenge metric
TrainCfg.early_stopping.min_delta = 0.001  # should be non-negative
TrainCfg.early_stopping.patience = TrainCfg.n_epochs // 2
TrainCfg.keep_checkpoint_max = 10

# configs of loss function
# TrainCfg.loss = "AsymmetricLoss"  # "FocalLoss", "BCEWithLogitsLoss"
# TrainCfg.loss_kw = CFG(gamma_pos=0, gamma_neg=0.2, implementation="deep-psp")
TrainCfg.flooding_level = 0.0  # flooding performed if positive,

# configs of logging
TrainCfg.log_step = 20
# TrainCfg.eval_every = 20

for t in TrainCfg.tasks:
    TrainCfg[t] = CFG()

###########################################
# classification configurations
###########################################

TrainCfg.classification.fs = BaseCfg.fs
TrainCfg.classification.n_channels = BaseCfg.n_channels
TrainCfg.classification.final_model_name = None
TrainCfg.classification.output_target = BaseCfg.output_target

# input format configurations
TrainCfg.classification.data_format = "channel_first"
TrainCfg.classification.input_config = InputConfig(
    input_type="waveform",  # "waveform", "spectrogram", "mel", "mfcc", "spectral"
    n_channels=TrainCfg.classification.n_channels,
    fs=TrainCfg.classification.fs,
)
TrainCfg.classification.num_channels = TrainCfg.classification.input_config.n_channels
TrainCfg.classification.input_len = int(
    100 * TrainCfg.classification.fs
)  # 100 seconds, to adjust
TrainCfg.classification.siglen = TrainCfg.classification.input_len  # alias
TrainCfg.classification.sig_slice_tol = None  # None, do no slicing

if TrainCfg.classification.output_target == "cpc":
    TrainCfg.classification.classes = deepcopy(BaseCfg.cpc)
    TrainCfg.classification.class_map = deepcopy(BaseCfg.cpc_map)
elif TrainCfg.classification.output_target == "outcome":
    TrainCfg.classification.classes = deepcopy(BaseCfg.outcome)
    TrainCfg.classification.class_map = deepcopy(BaseCfg.outcome_map)

# preprocess configurations
# NOTE that all EEG data was pre-processed with bandpass filtering (0.5-20Hz) and resampled to 100 Hz.
TrainCfg.classification.resample = CFG(fs=TrainCfg.classification.fs)
TrainCfg.classification.bandpass = False
TrainCfg.classification.normalize = CFG(  # None or False for no normalization
    method="z-score",
    mean=0.0,
    std=1.0,
)

# augmentations configurations via `from_dict` of `torch-audiomentations`
TrainCfg.classification.augmentations = [
    # currently empty
]
TrainCfg.classification.augmentations_kw = CFG(
    p=0.7,
    p_mode="per_batch",
)

# model choices
TrainCfg.classification.model_name = "crnn"  # "wav2vec", "crnn"
TrainCfg.classification.cnn_name = "resnet_nature_comm_bottle_neck_se"
TrainCfg.classification.rnn_name = "lstm"  # "none", "lstm"
TrainCfg.classification.attn_name = "se"  # "none", "se", "gc", "nl"

# loss function choices
TrainCfg.classification.loss = CFG(
    cpc="AsymmetricLoss",  # "FocalLoss", "BCEWithLogitsWithClassWeightLoss"
    outcome="AsymmetricLoss",  # "FocalLoss", "BCEWithLogitsWithClassWeightLoss"
)
TrainCfg.classification.loss_kw = CFG(
    cpc=CFG(gamma_pos=0, gamma_neg=0.2, implementation="deep-psp"),
    outcome=CFG(gamma_pos=0, gamma_neg=0.2, implementation="deep-psp"),
)

# monitor choices
# "outcome_score", "outcome_accuracy", "outcome_f_measure", "cpc_mae", "cpc_mse"
TrainCfg.classification.monitor = "outcome_score"

# TODO: consider a regression task for cpc


def set_entry_test_flag(test_flag: bool):
    TrainCfg.entry_test_flag = test_flag


###############################################################################
# configurations for building deep learning models
# terminologies of stanford ecg repo. will be adopted
###############################################################################


_BASE_MODEL_CONFIG = CFG()
_BASE_MODEL_CONFIG.torch_dtype = BaseCfg.torch_dtype


ModelCfg = deepcopy(_BASE_MODEL_CONFIG)

for t in TrainCfg.tasks:
    ModelCfg[t] = deepcopy(_BASE_MODEL_CONFIG)
    ModelCfg[t].task = t
    ModelCfg[t].output_target = TrainCfg[t].output_target
    ModelCfg[t].classes = TrainCfg[t].classes
    ModelCfg[t].fs = TrainCfg[t].fs

    ModelCfg[t].update(deepcopy(ModelArchCfg[t]))

    ModelCfg[t].num_channels = TrainCfg[t].num_channels
    ModelCfg[t].input_len = TrainCfg[t].input_len
    ModelCfg[t].model_name = TrainCfg[t].model_name
    ModelCfg[t].cnn_name = TrainCfg[t].cnn_name
    ModelCfg[t].rnn_name = TrainCfg[t].rnn_name
    ModelCfg[t].attn_name = TrainCfg[t].attn_name

    # adjust filter length; cnn, rnn, attn choices in model configs
    for mn in [
        "crnn",
        # "seq_lab",
        # "unet",
    ]:
        if mn not in ModelCfg[t]:
            continue
        ModelCfg[t][mn] = adjust_cnn_filter_lengths(ModelCfg[t][mn], ModelCfg[t].fs)
        ModelCfg[t][mn].cnn.name = ModelCfg[t].cnn_name
        ModelCfg[t][mn].rnn.name = ModelCfg[t].rnn_name
        ModelCfg[t][mn].attn.name = ModelCfg[t].attn_name


# machine learning model configurations

MLCfg = CFG()
MLCfg.db_dir = None
MLCfg.log_dir = BaseCfg.log_dir
MLCfg.model_dir = BaseCfg.model_dir
MLCfg.y_col = "cpc"
MLCfg.task = "classification"  # "classification", "regression"
MLCfg.classes = deepcopy(BaseCfg.cpc)
MLCfg.class_map = deepcopy(BaseCfg.cpc_map)
MLCfg.x_cols_cate = [  # categorical features
    "Sex",
    "OHCA",
    "VFib",
    "TTM",
]
MLCfg.x_cols_cont = [  # continuous features
    "Age",
    "ROSC",
]
MLCfg.cont_scaler = "standard"  # "minmax", "standard"
MLCfg.x_cols = MLCfg.x_cols_cate + MLCfg.x_cols_cont
MLCfg.feature_list = deepcopy(MLCfg.x_cols)
MLCfg.grids = CFG()
MLCfg.grids.rf = ParameterGrid(
    {
        "n_estimators": [10, 15, 20, 50, 100],
        "criterion": ["gini", "entropy"],
        "min_samples_split": [2, 3, 4],
        "max_features": ["auto", "sqrt", "log2"],
        "bootstrap": [True, False],
        "oob_score": [True, False],
        "warm_start": [True, False],
        "class_weight": ["balanced", "balanced_subsample", None],
    }
)
MLCfg.grids.xgb = ParameterGrid(
    {
        "n_estimators": [10, 15, 20, 50],
        "learning_rate": [0.01, 0.05, 0.1],
        "reg_alpha": [0.0, 0.1, 0.5, 1.0],
        "reg_lambda": [0.0, 0.1, 0.5, 1.0],
        "max_depth": [3, 5, 8],
        "verbosity": [0],
    }
)
MLCfg.grids.gdbt = ParameterGrid(
    {
        "n_estimators": [10, 15, 20, 50, 100],
        "loss": ["deviance", "exponential"],
        "learning_rate": [0.01, 0.05, 0.1],
        "criterion": ["friedman_mse", "mse"],
        "min_samples_split": [2, 3, 4],
        "max_features": ["auto", "sqrt", "log2"],
        "warm_start": [True, False],
        "ccp_alpha": [0.0, 0.1, 0.5, 1.0],
    }
)
MLCfg.grids.svc = ParameterGrid(
    {
        "C": [0.1, 0.5, 1, 10],
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "degree": [2, 3, 5],  # for "poly" kernel
        "gamma": [
            "scale",
            "auto",
        ],  # Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
        "coef0": [0.0, 0.2, 0.5, 1.0],  # for 'poly' and 'sigmoid'
        "class_weight": ["balanced", None],
        "probability": [True],
        "shrinking": [True, False],
    }
)
MLCfg.grids.bagging = ParameterGrid(
    {
        "n_estimators": [10, 15, 20, 50, 100],
        "max_features": [0.1, 0.2, 0.5, 0.9, 1.0],
        "bootstrap": [True, False],
        "bootstrap_features": [True, False],
        "oob_score": [True, False],
        "warm_start": [True, False],
    }
)
MLCfg.monitor = "outcome_score"
