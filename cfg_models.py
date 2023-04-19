"""
"""

from copy import deepcopy

from torch_ecg.cfg import CFG
from torch_ecg.model_configs import (  # noqa: F401
    # cnn bankbone
    vgg_block_basic,
    vgg_block_mish,
    vgg_block_swish,
    vgg16,
    vgg16_leadwise,
    resnet_block_basic,
    resnet_bottle_neck_B,
    resnet_bottle_neck_D,
    resnet_block_basic_se,
    resnet_block_basic_gc,
    resnet_bottle_neck_se,
    resnet_bottle_neck_gc,
    resnet_nature_comm,
    resnet_nature_comm_se,
    resnet_nature_comm_gc,
    resnet_nature_comm_bottle_neck,
    resnet_nature_comm_bottle_neck_se,
    resnetN,
    resnetNB,
    resnetNS,
    resnetNBS,
    tresnetF,
    tresnetP,
    tresnetN,
    tresnetS,
    tresnetM,
    multi_scopic_block,
    multi_scopic,
    multi_scopic_leadwise,
    densenet_leadwise,
    xception_leadwise,
    # lstm
    lstm,
    attention,
    # mlp
    linear,
    # attn
    non_local,
    squeeze_excitation,
    global_context,
    # the whole model config
    ECG_CRNN_CONFIG,
)


__all__ = [
    "ModelArchCfg",
]


# mostly follow from torch_ecg.torch_ecg.model_configs.ecg_crnn
ModelArchCfg = CFG()

ModelArchCfg.classification = CFG()
ModelArchCfg.classification.crnn = deepcopy(ECG_CRNN_CONFIG)
