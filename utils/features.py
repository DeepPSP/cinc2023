"""
features from metadata
"""

import os

import numpy as np

from helper_code import (
    get_age,
    get_sex,
    get_rosc,
    get_ohca,
    get_vfib,
    get_ttm,
    get_outcome,
    get_cpc,
    load_text_file,
)


__all__ = [
    "get_features",
    "get_labels",
    "load_challenge_metadata",
]


def get_features(patient_metadata: str) -> np.ndarray:
    """Extract features from the patient metadata.

    Adapted from the official repo.

    Parameters
    ----------
    patient_metadata : str
        The patient metadata.

    Returns
    -------
    np.ndarray
        The patient features.

    """
    age = get_age(patient_metadata)
    sex = get_sex(patient_metadata)
    rosc = get_rosc(patient_metadata)
    ohca = get_ohca(patient_metadata)
    vfib = get_vfib(patient_metadata)
    ttm = get_ttm(patient_metadata)

    # Use one-hot encoding for sex; add more variables
    sex_features = np.zeros(2, dtype=int)
    if sex == "Female":
        female = 1
        male = 0
        other = 0
    elif sex == "Male":
        female = 0
        male = 1
        other = 0
    else:
        female = 0
        male = 0
        other = 1

    # Combine the patient features.
    patient_features = np.array([age, female, male, other, rosc, ohca, vfib, ttm])

    return patient_features


def get_labels(patient_metadata: str) -> dict:
    """Extract labels from the patient metadata.

    Adapted from the official repo.

    Parameters
    ----------
    patient_metadata : str
        The patient metadata.

    Returns
    -------
    dict
        The patient labels, including
        - "outcome" (int)
        - "cpc" (float)

    """
    labels = {}
    labels["outcome"] = get_outcome(patient_metadata)
    labels["cpc"] = get_cpc(patient_metadata)

    return labels


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
