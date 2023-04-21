#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries and functions. You can change or remove them.
#
################################################################################

import os

import mne
import numpy as np

from helper_code import (  # noqa: F401
    find_data_folders,
    load_challenge_data,
    get_outcome,
    get_cpc,
    get_age,
    get_sex,
    get_rosc,
    get_ohca,
    get_vfib,
    get_ttm,
    reorder_recording_channels,
    get_quality_scores,
)  # noqa: F401


################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################


# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose >= 1:
        print("Finding the Challenge data...")

    patient_ids = find_data_folders(data_folder)
    num_patients = len(patient_ids)

    if num_patients == 0:
        raise FileNotFoundError("No data was provided.")

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Train the models.
    if verbose >= 1:
        print("Training the Challenge models on the Challenge data...")

    # TODO

    if verbose >= 1:
        print("Done.")


# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_models(model_folder, verbose):
    # TODO
    pass


# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):

    # Load data.
    patient_metadata, recording_metadata, recording_data = load_challenge_data(
        data_folder, patient_id
    )

    # TODO

    # return outcome, outcome_probability, cpc


################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################


# Extract features from the data.
def get_features(patient_metadata, recording_metadata, recording_data):
    # Extract features from the patient metadata.
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

    # Extract features from the recording data and metadata.
    channels = [
        "Fp1-F7",
        "F7-T3",
        "T3-T5",
        "T5-O1",
        "Fp2-F8",
        "F8-T4",
        "T4-T6",
        "T6-O2",
        "Fp1-F3",
        "F3-C3",
        "C3-P3",
        "P3-O1",
        "Fp2-F4",
        "F4-C4",
        "C4-P4",
        "P4-O2",
        "Fz-Cz",
        "Cz-Pz",
    ]
    num_channels = len(channels)
    num_recordings = len(recording_data)

    # Compute mean and standard deviation for each channel for each recording.
    available_signal_data = list()
    for i in range(num_recordings):
        signal_data, sampling_frequency, signal_channels = recording_data[i]
        if signal_data is not None:
            signal_data = reorder_recording_channels(
                signal_data, signal_channels, channels
            )  # Reorder the channels in the signal data, as needed, for consistency across different recordings.
            available_signal_data.append(signal_data)

    if len(available_signal_data) > 0:
        available_signal_data = np.hstack(available_signal_data)
        signal_mean = np.nanmean(available_signal_data, axis=1)
        signal_std = np.nanstd(available_signal_data, axis=1)
    else:
        signal_mean = float("nan") * np.ones(num_channels)
        signal_std = float("nan") * np.ones(num_channels)

    # Compute the power spectral density for the delta, theta, alpha, and beta frequency bands for each channel of the most
    # recent recording.
    index = None
    for i in reversed(range(num_recordings)):
        signal_data, sampling_frequency, signal_channels = recording_data[i]
        if signal_data is not None:
            index = i
            break

    if index is not None:
        signal_data, sampling_frequency, signal_channels = recording_data[index]
        signal_data = reorder_recording_channels(
            signal_data, signal_channels, channels
        )  # Reorder the channels in the signal data, as needed, for consistency across different recordings.

        delta_psd, _ = mne.time_frequency.psd_array_welch(
            signal_data, sfreq=sampling_frequency, fmin=0.5, fmax=8.0, verbose=False
        )
        theta_psd, _ = mne.time_frequency.psd_array_welch(
            signal_data, sfreq=sampling_frequency, fmin=4.0, fmax=8.0, verbose=False
        )
        alpha_psd, _ = mne.time_frequency.psd_array_welch(
            signal_data, sfreq=sampling_frequency, fmin=8.0, fmax=12.0, verbose=False
        )
        beta_psd, _ = mne.time_frequency.psd_array_welch(
            signal_data, sfreq=sampling_frequency, fmin=12.0, fmax=30.0, verbose=False
        )

        delta_psd_mean = np.nanmean(delta_psd, axis=1)
        theta_psd_mean = np.nanmean(theta_psd, axis=1)
        alpha_psd_mean = np.nanmean(alpha_psd, axis=1)
        beta_psd_mean = np.nanmean(beta_psd, axis=1)

        quality_score = get_quality_scores(recording_metadata)[index]
    else:
        delta_psd_mean = theta_psd_mean = alpha_psd_mean = beta_psd_mean = float(
            "nan"
        ) * np.ones(num_channels)
        quality_score = float("nan")

    recording_features = np.hstack(
        (
            signal_mean,
            signal_std,
            delta_psd_mean,
            theta_psd_mean,
            alpha_psd_mean,
            beta_psd_mean,
            quality_score,
        )
    )

    # Combine the features from the patient metadata and the recording data and metadata.
    features = np.hstack((patient_features, recording_features))

    return features
