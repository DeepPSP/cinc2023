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

    cpc_output: ClassificationOutput
    cpc_loss: Optional[float] = None
    outcome: Optional[str] = None
