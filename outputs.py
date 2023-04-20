"""
"""

from typing import Optional
from dataclasses import dataclass

from torch_ecg.components.outputs import (
    ClassificationOutput,
)


__all__ = ["CINC2023Outputs"]


@dataclass
class CINC2023Outputs:
    """Output class for CinC2023"""

    cpc_output: Optional[ClassificationOutput] = None
    cpc_loss: Optional[float] = None
    outcome_output: Optional[ClassificationOutput] = None
    outcome_loss: Optional[float] = None
    outcome: Optional[str] = None

    def __post_init__(self):
        assert any(
            [
                self.cpc_output is not None,
                self.outcome_output is not None,
            ]
        ), "At least one output should be provided"
