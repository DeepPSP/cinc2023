"""
Miscellaneous functions.
"""

import os
from functools import wraps
from pathlib import Path
from typing import Callable, Any

import pandas as pd

from helper_code import load_text_file


__all__ = [
    "load_challenge_metadata",
    "load_unofficial_phase_metadata",
    "func_indicator",
]


def load_challenge_metadata(data_folder: str, patient_id: str) -> str:
    """Load the patient metadata.

    Adapted the load_challenge_data function from the official repo.

    Parameters
    ----------
    data_folder : str
        The data folder.
    patient_id : str
        The patient ID.

    Returns
    -------
    str
        The patient metadata.

    """
    # Define file location.
    patient_metadata_file = os.path.join(data_folder, patient_id, patient_id + ".txt")

    # Load non-recording data.
    patient_metadata = load_text_file(patient_metadata_file)

    return patient_metadata


def load_unofficial_phase_metadata() -> pd.DataFrame:
    """Load the unofficial phase metadata."""
    path = Path(__file__).parent / "unofficial_phase_metadata.csv.gz"
    df = pd.read_csv(path, index_col=0)
    df["subject"] = df["subject"].apply(lambda x: f"{x:04d}")
    df["start_sec"] = df["time"].apply(lambda x: 60 * int(x.split(":")[1]))
    df["end_sec"] = df["start_sec"] + 60 * 5
    return df


def func_indicator(name: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            print("\n" + "-" * 100)
            print(f"  Start {name}  ".center(100, "-"))
            print("-" * 100 + "\n")
            func(*args, **kwargs)
            print("\n" + "-" * 100)
            print(f"  End {name}  ".center(100, "-"))
            print("-" * 100 + "\n")

        return wrapper

    return decorator
