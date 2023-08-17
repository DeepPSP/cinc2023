"""
Miscellaneous functions.
"""

import os
from functools import wraps
from pathlib import Path
from typing import Callable, Any, List, Tuple

import numpy as np
import pandas as pd
from torch_ecg.utils.misc import get_record_list_recursive3
from tqdm.auto import tqdm

from cfg import BaseCfg
from helper_code import load_text_file, load_recording_data


__all__ = [
    "load_challenge_eeg_data",
    "find_eeg_recording_files",
    "load_challenge_metadata",
    "load_unofficial_phase_metadata",
    "func_indicator",
]


def load_challenge_eeg_data(
    data_folder: str, patient_id: str
) -> List[Tuple[np.ndarray, int, List[str]]]:
    """Load challenge EEG data given the data folder and patient ID.

    Adapted from the ``load_challenge_data`` function of the unofficial phase.

    Parameters
    ----------
    data_folder : str
        The data folder.
    patient_id : str
        The patient ID.

    Returns
    -------
    List[Tuple[numpy.ndarray, int, List[str]]]
        A list of tuples, each tuple contains
        the recording data, sampling frequency, and channel names.

    """
    patient_folder = Path(data_folder) / patient_id
    # Load recordings.
    recording_files = find_eeg_recording_files(data_folder, patient_id)
    recordings = list()
    with tqdm(
        recording_files, desc=f"Loading {patient_id} recordings", mininterval=1
    ) as pbar:
        for recording_location in pbar:
            if os.path.exists(recording_location + ".hea"):
                recording_data, channels, sampling_frequency = load_recording_data(
                    recording_location
                )
                # utility_frequency = get_utility_frequency(recording_location + ".hea")
                recordings.append((recording_data, int(sampling_frequency), channels))
    return recordings


def find_eeg_recording_files(data_folder: str, patient_id: str) -> List[str]:
    """Find the EEG recording files.

    Parameters
    ----------
    data_folder : str
        The data folder.
    patient_id : str
        The patient ID.

    Returns
    -------
    List[str]
        Absolute paths of the EEG recording files, without file extension.

    """
    patient_folder = Path(data_folder) / patient_id
    recording_files = get_record_list_recursive3(
        patient_folder, f"{BaseCfg.recording_pattern}\\.hea", relative=False
    )
    recording_files = [
        fp
        for fp in recording_files
        if fp.endswith("EEG") and Path(fp).parent == patient_folder
    ]
    return recording_files


def load_challenge_metadata(data_folder: str, patient_id: str) -> str:
    """Load the patient metadata.

    Adapted from the ``load_challenge_data`` function of the unofficial phase.
    Now deprecated by the ``load_challenge_data`` function of the official phase.

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
