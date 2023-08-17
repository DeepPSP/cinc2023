from copy import deepcopy
from typing import Sequence, Optional

import numpy as np
from torch_ecg.cfg import CFG
from torch_ecg.utils import mask_to_intervals

from .artifact_pipeline.segment_EEG import (
    segment_EEG as eeg_segment_func,
    seg_mask_explanation,
)


__all__ = [
    "compute_sqi",
    "find_normal_intervals",
    "DEFAULT_SEGMENT_CONFIG",
]


# config item names follow ./artifact_pipeline/step1_process_each_file_bipolar.py
DEFAULT_SEGMENT_CONFIG = CFG(
    # resamp_fs=100.0,  # Hz
    window_time=5,  # s
    window_step=5,  # s
    start_end_remove_window_num=0,
    to_remove_mean=False,
    amplitude_thres=500,  # uV
    line_freq=60.0,  # Hz
    bandpass_freq=[0.5, 30.0],  # Hz
    # fmt: off
    available_channels=[
        "Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz",
        "Pz", "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2",
    ],
    bipolar_channels=[
        "Fp1-F7", "F7-T3", "T3-T5", "T5-O1", "Fp2-F8", "F8-T4", "T4-T6",
        "T6-O2", "Fp1-F3", "F3-C3", "C3-P3", "P3-O1", "Fp2-F4", "F4-C4",
        "C4-P4", "P4-O2", "Fz-Cz", "Cz-Pz",
    ],
    # fmt: on
)


def compute_sqi(
    signal: np.ndarray,
    channels: Sequence[str],
    fs: float,
    is_bipolar: bool,
    segment_config: Optional[dict] = None,
    sqi_window_time: float = 5.0,  # min
    sqi_window_step: float = 1.0,  # min
) -> np.ndarray:
    """Compute SQI for the signal.

    SQI is computed as the proportion of normal segments
    in the window of ``sqi_window_time`` minutes.

    Parameters
    ----------
    signal : np.ndarray
        The signal to compute SQI for. Shape: ``(n_channels, n_samples)``.
    channels : Sequence[str]
        The channel names in the signal.
    fs : float
        The sampling frequency of the signal.
    is_bipolar : bool
        Whether the signal is bipolar.
        If True, the signal is used directly.
        Otherwise, the bipolar signal is computed from the original signal.
    segment_config : dict, optional
        The configuration for segmenting the signal.
        If not provided, use the default configuration,
        i.e. ``DEFAULT_SEGMENT_CONFIG``.
    sqi_window_time : float, default 5.0
        The window time for computing SQI, in minutes.
    sqi_window_step : float, default 1.0
        The window step for computing SQI, in minutes.

    Returns
    -------
    sqi : np.ndarray
        The SQI for the signal. Shape: ``(n_windows, 3)``.
        The first column is the start index of the window,
        the second column is the end index of the window,
        and the third column is the SQI value.

    """
    # use default config if not provided
    _config = deepcopy(DEFAULT_SEGMENT_CONFIG)
    _config.update(segment_config or {})

    if is_bipolar:
        bipolar_signal = signal.astype(np.float64)
        _config.bipolar_channels = channels
    else:
        # obtain the bipolar signal from the original signal
        channel_pairs = [
            [pair.split("-")[0] for pair in _config.bipolar_channels],
            [pair.split("-")[1] for pair in _config.bipolar_channels],
        ]
        diff_inds = [[channels.index(item) for item in lst] for lst in channel_pairs]
        bipolar_signal = signal[diff_inds[0]] - signal[diff_inds[1]]
        bipolar_signal = bipolar_signal.astype(np.float64)

    # compute segments and information of the segments
    segs_, bs_, seg_start_ids_, seg_mask, specs_, freqs_ = eeg_segment_func(
        EEG=bipolar_signal,
        Ch_names=_config.bipolar_channels,
        window_time=_config.window_time,
        step_time=_config.window_step,
        Fs=fs,
        notch_freq=_config.line_freq,
        bandpass_freq=_config.bandpass_freq,
        to_remove_mean=_config.to_remove_mean,
        amplitude_thres=_config.amplitude_thres,
        start_end_remove_window_num=_config.start_end_remove_window_num,
        n_jobs=-1,
    )
    seg_end_ids_ = seg_start_ids_ + int(_config.window_time * fs)

    # no segments found
    if len(segs_) <= 0:
        return np.array([]).reshape(0, 3)

    seg_mask = map(lambda x: x.split("_")[0], seg_mask)
    # convert to int type according to index in `seg_mask_explanation`
    seg_mask = np.array([seg_mask_explanation.index(item) for item in seg_mask])
    normal_index = seg_mask_explanation.index("normal")
    # convert to boolean mask, where True means normal
    # and False means other (abnormal) types of segments
    normal_mask = seg_mask == normal_index

    # compute SQI for each window
    window_size = int(sqi_window_time * 60 * fs)
    window_step = int(sqi_window_step * 60 * fs)
    siglen = signal.shape[1]
    num_windows = (siglen - window_size) // window_step + 1
    sqi = np.zeros((num_windows, 3), dtype=np.float64)
    for i in range(num_windows):
        start = i * window_step
        end = min(start + window_size, siglen)
        sqi[i, 0] = start
        sqi[i, 1] = end
        # count the number of segments that is totally in the window
        # and the number of normal segments
        num_total = np.sum((seg_start_ids_ >= start) & (seg_end_ids_ <= end))
        num_normal = np.sum(
            (seg_start_ids_ >= start) & (seg_end_ids_ <= end) & normal_mask
        )
        sqi[i, 2] = num_normal / num_total if num_total > 0 else 0.0

    return sqi


def find_normal_intervals(
    signal: np.ndarray,
    channels: Sequence[str],
    fs: float,
    is_bipolar: bool,
    segment_config: Optional[dict] = None,
) -> np.ndarray:
    """Find normal intervals in the signal.

    NOTE that if there are gaps in the segments,
    i.e. ``window_step`` is greater than ``window_time``,
    the gaps will be filled with the type of their previous windows.

    Parameters
    ----------
    signal : np.ndarray
        The input signal. Shape: ``(n_channels, n_samples)``.
    channels : Sequence[str]
        The channel names in the signal.
    fs : float
        The sampling frequency of the signal.
    is_bipolar : bool, optional
        Whether the signal is bipolar.
        If True, the signal is used directly.
        Otherwise, the bipolar signal is computed from the original signal.
    segment_config : dict, optional
        The configuration for segmenting the signal.
        If not provided, use the default configuration.

    Returns
    -------
    normal_intervals: np.ndarray
        The normal intervals in the signal. Shape: ``(n_intervals, 2)``.
        The first column is the start index of the interval,
        and the second column is the end index of the interval.
        The intervals are half-open: ``[start, end)``.

    """
    # use default config if not provided
    _config = deepcopy(DEFAULT_SEGMENT_CONFIG)
    _config.update(segment_config or {})

    if is_bipolar:
        bipolar_signal = signal.astype(np.float64)
        _config.bipolar_channels = channels
    else:
        # obtain the bipolar signal from the original signal
        channel_pairs = [
            [pair.split("-")[0] for pair in _config.bipolar_channels],
            [pair.split("-")[1] for pair in _config.bipolar_channels],
        ]
        diff_inds = [[channels.index(item) for item in lst] for lst in channel_pairs]
        bipolar_signal = signal[diff_inds[0]] - signal[diff_inds[1]]
        bipolar_signal = bipolar_signal.astype(np.float64)

    # compute segments and information of the segments
    segs_, bs_, seg_start_ids_, seg_mask, specs_, freqs_ = eeg_segment_func(
        EEG=bipolar_signal,
        Ch_names=_config.bipolar_channels,
        window_time=_config.window_time,
        step_time=_config.window_step,
        Fs=fs,
        notch_freq=_config.line_freq,
        bandpass_freq=_config.bandpass_freq,
        to_remove_mean=_config.to_remove_mean,
        amplitude_thres=_config.amplitude_thres,
        start_end_remove_window_num=_config.start_end_remove_window_num,
        n_jobs=-1,
    )
    seg_end_ids_ = seg_start_ids_ + int(_config.window_time * fs)

    # no segments found
    if len(segs_) <= 0:
        return np.array([]).reshape(0, 2)

    seg_mask = map(lambda x: x.split("_")[0], seg_mask)
    # convert to int type according to index in `seg_mask_explanation`
    seg_mask = np.array([seg_mask_explanation.index(item) for item in seg_mask])
    # convert to intervals
    normal_index = seg_mask_explanation.index("normal")
    normal_window_intervals = mask_to_intervals(
        seg_mask, vals=normal_index, right_inclusive=False
    )
    # convert to samples according to seg_start_ids_ and seg_end_ids_
    normal_intervals = [
        [seg_start_ids_[start], seg_end_ids_[end - 1]]
        for start, end in normal_window_intervals
    ]
    normal_intervals = np.array(normal_intervals).astype(np.int64)

    return normal_intervals
