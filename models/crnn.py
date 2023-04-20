"""
"""

from copy import deepcopy
from typing import Union, Optional, Any, Dict

import numpy as np
import torch
from torch import Tensor
from torch_ecg.cfg import CFG
from torch_ecg.models.ecg_crnn import ECG_CRNN
from torch_ecg.components.outputs import (
    ClassificationOutput,
)
from torch_ecg.utils import add_docstring

from cfg import ModelCfg
from outputs import CINC2023Outputs


__all__ = ["CRNN_CINC2023"]


class CRNN_CINC2023(ECG_CRNN):
    """ """

    __DEBUG__ = True
    __name__ = "CRNN_CINC2023"

    def __init__(self, config: Optional[CFG] = None, **kwargs: Any) -> None:
        """

        Parameters
        ----------
        config : dict
            hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file

        Usage
        -----
        ```python
        from cfg import ModelCfg
        task = "classification"
        model_cfg = deepcopy(ModelCfg[task])
        model = ECG_CRNN_CINC2023(model_cfg)
        ````

        """
        if config is None:
            _config = deepcopy(ModelCfg.classification)
        else:
            _config = deepcopy(config)
        super().__init__(
            _config.classes,
            _config.num_channels,
            _config[_config.model_name],
        )
        self.output_target = _config.output_target

    def forward(
        self,
        waveforms: Tensor,
        labels: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """

        Parameters
        ----------
        waveforms : torch.Tensor
            Waveforms tensor, of shape ``(batch_size, channels, seq_len)``.
        labels : Dict[str, torch.Tensor], optional
            the labels of the waveforms data, including:
            - "cpc" (optional): the cpc labels,
              of shape ``(batch_size, n_classes)`` or ``(batch_size,)``
            - "outcome" (optional): the outcome labels,
              of shape ``(batch_size, n_classes)`` or ``(batch_size,)``

        Returns
        -------
        Dict[str, torch.Tensor]
            with items:
            - "cpc" (optional): the cpc predictions,
              of shape ``(batch_size, n_classes)`` or ``(batch_size,)``
            - "outcome" (optional): the outcome predictions,
              of shape ``(batch_size, n_classes)``

        """
        pred = super().forward(waveforms)  # batch_size, n_classes
        out = {self.output_target: pred}

        return out

    @torch.no_grad()
    def inference(self, waveforms: Union[np.ndarray, Tensor]) -> CINC2023Outputs:
        """auxiliary function to `forward`, for CINC2023,

        Parameters
        ----------
        waveforms : numpy.ndarray or torch.Tensor,
            Waveforms tensor, of shape ``(batch_size, channels, seq_len)``

        Returns
        -------
        CINC2023Outputs, with attributes:
            - cpc_output, outcome_output: ClassificationOutput, with items:
                - classes: list of str,
                  list of the class names
                - prob: ndarray or DataFrame,
                  scalar (probability) predictions,
                  (and binary predictions if `class_names` is True)
                - pred: ndarray,
                  the array of class number predictions
                - bin_pred: ndarray,
                  the array of binary predictions
                - forward_output: ndarray,
                  the array of output of the model's forward function,
                  useful for producing challenge result using
                  multiple recordings

        """
        self.eval()
        _input = torch.as_tensor(waveforms, dtype=self.dtype, device=self.device)
        if _input.ndim == 2:
            _input = _input.unsqueeze(0)  # add a batch dimension
        # batch_size, channels, seq_len = _input.shape
        forward_output = self.forward(_input)

        prob = self.softmax(forward_output[self.output_target])
        pred = torch.argmax(prob, dim=-1)
        bin_pred = (prob == prob.max(dim=-1, keepdim=True).values).to(int)
        prob = prob.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()
        bin_pred = bin_pred.cpu().detach().numpy()

        output = ClassificationOutput(
            classes=self.classes,
            prob=prob,
            pred=pred,
            bin_pred=bin_pred,
            forward_output=forward_output[self.output_target].cpu().detach().numpy(),
        )

        if self.output_target == "cpc":
            return CINC2023Outputs(cpc_output=output)
        elif self.output_target == "outcome":
            return CINC2023Outputs(outcome_output=output)

    @add_docstring(inference.__doc__)
    def inference_CINC2023(
        self,
        waveforms: Union[np.ndarray, Tensor],
        seg_thr: float = 0.5,
    ) -> CINC2023Outputs:
        """
        alias for `self.inference`
        """
        return self.inference(waveforms, seg_thr)
