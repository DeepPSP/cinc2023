"""
"""

from pathlib import Path
from typing import Union, Optional, Any, List, Dict, Sequence

import numpy as np
import pandas as pd  # noqa: F401
import wfdb  # noqa: F401
import scipy.signal as ss  # noqa: F401
import scipy.io as sio  # noqa: F401
from torch_ecg.databases.base import PhysioNetDataBase, DataBaseInfo
from torch_ecg.utils.misc import (  # noqa: F401
    get_record_list_recursive,
    get_record_list_recursive3,
    list_sum,
    add_docstring,
)

from cfg import BaseCfg


__all__ = [
    "CINC2023Reader",
]


_CINC2023_INFO = DataBaseInfo(
    title="""
    Predicting Neurological Recovery from Coma After Cardiac Arrest
    """,
    about="""
    1. The goal of the Challenge is to use longitudinal EEG recordings to predict good and poor patient outcomes after cardiac arrest.
    2. The data originates from seven academic hospitals.
    3. All EEG data was pre-processed with bandpass filtering (0.5-20Hz) and resampled to 100 Hz.
    4. Each recording contains an array with EEG signals from 18 bipolar channel pairs.
    5. The EEG recordings continue for several hours to days, so the EEG signals are prone to quality deterioration from non-physiological artifacts. Only the **cleanest 5 minutes** of EEG data per hour are provided.
    6. There might be gaps in the EEG data, since patients may have EEG started several hours after the arrest or need to have brain monitoring interrupted transiently while in the ICU.
    7. In addition to EEG data, one additional .tsv file includes artifact scores for each hour, containing
        - **Time**: the timestamp for the start of each EEG signal file in relation to the time of cardiac arrest (under the column “Time”).
        - **Quality**: a measure of quality of the EEG signal for the 5-minute epochs, based on how many 10-second epochs within a 5-minute EEG window are contaminated by artifacts, ranging from 0 (all artifacts) to 1 (no artifacts).
    8. Each patient has one .txt file containing patient information (ref. 9) and clinical outcome (ref. 10).
    9. Patient information includes information recorded at the time of admission (age, sex), location of arrest (out or in-hospital), type of cardiac rhythm recorded at the time of resuscitation (shockable rhythms include ventricular fibrillation or ventricular tachycardia and non-shockable rhythms include asystole and pulseless electrical activity), and the time between cardiac arrest and ROSC (return of spontaneous circulation).
    10. Clinical outcome was determined prospectively in two centers by phone interview (at 6 months from ROSC), and at the remaining hospitals retrospectively through chart review (at 3-6 months from ROSC). Neurological function was determined using the Cerebral Performance Category (CPC) scale. CPC is an ordinal scale ranging from 1 to 5:
        - CPC = 1: good neurological function and independent for activities of daily living.
        - CPC = 2: moderate neurological disability but independent for activities of daily living.
        - CPC = 3: severe neurological disability.
        - CPC = 4: unresponsive wakefulness syndrome [previously known as vegetative state].
        - CPC = 5: dead.
    The CPC scores are grouped into two categories:
        - Good: CPC = 1 or 2.
        - Poor: CPC = 3, 4, or 5.
    """,
    usage=[
        "neurological recovery prediction",
    ],
    note="""
    """,
    issues="""
    """,
    references=[
        "https://moody-challenge.physionet.org/2023/",
    ],
    doi=[],
)


@add_docstring(_CINC2023_INFO.format_database_docstring())
class CINC2023Reader(PhysioNetDataBase):
    """ """

    __name__ = "CINC2023Reader"

    def __init__(
        self,
        db_dir: str,
        fs: int = 100,
        backend: str = "wfdb",
        working_dir: Optional[str] = None,
        verbose: int = 2,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        db_dir: str,
            storage path of the database
        fs: int, default 100,
            (re-)sampling frequency of the recordings
        backend: str, default "wfdb",
            backend to use, can be one of
            "scipy",  "wfdb",
            case insensitive.
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
            log verbosity
        kwargs: auxilliary key word arguments

        """
        super().__init__(
            db_name="cinc2023",  # to update
            db_dir=db_dir,
            fs=fs,
            backend=backend,
            working_dir=working_dir,
            verbose=verbose,
            **kwargs,
        )
        self.dtype = kwargs.get("dtype", BaseCfg.np_dtype)

        self._rec_pattern = "ICARE\\_(?P<sid>[\\d]+)\\_(?P<loc>[\\d]+)"
        self.data_ext = "wav"
        self.header_ext = "hea"
        self.quality_ext = "tsv"
        self.ann_ext = "txt"

        self._all_records = None
        self._all_subjects = None
        self._subject_records = None
        self._ls_rec()

    def _auto_infer_units(self) -> None:
        """
        auto infer the units of the signals
        """
        raise NotImplementedError

    def _reset_fs(self, new_fs: int) -> None:
        """ """
        self.fs = new_fs

    def _ls_rec(self) -> None:
        """
        list all records in the database
        """
        raise NotImplementedError

    def get_absolute_path(
        self, rec: Union[str, int], extension: Optional[str] = None
    ) -> Path:
        """
        get the absolute path of the record `rec`

        Parameters
        ----------
        rec: str or int,
            record name or index of the record in `self.all_records`
        extension: str, optional,
            extension of the file

        Returns
        -------
        Path,
            absolute path of the file

        """
        pass

    def load_data(
        self,
        rec: Union[str, int],
        channels: Optional[Union[str, int, Sequence[Union[str, int]]]] = None,
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        data_format: str = "channel_first",
        units: Union[str, type(None)] = "mV",
        fs: Optional[int] = None,
    ) -> np.ndarray:
        """
        load data from the record `rec`

        Parameters
        ----------
        rec: str or int,
            the record name or the index of the record in `self.all_records`
        channels: str or int or sequence of str or int, optional,
            the channel(s) to load, if None, load all channels
        sampfrom: int, optional,
            the starting sample index, if None, load from the beginning
        sampto: int, optional,
            the ending sample index, if None, load to the end
        data_format: str, default "channel_first",
            the format of the data, can be one of
            "channel_first", "channel_last"
            case insensitive.
        units: str or None, default "mV",
            the units of the data, can be one of
            "mV", "uV" (with alias "muV", "μV"),
            case insensitive.
        fs: int, optional,
            the sampling frequency of the record, defaults to `self.fs`,

        Returns
        -------
        data: np.ndarray,
            the data of the record

        """
        pass

    def load_ann(
        self, rec_or_sid: Union[str, int], class_map: Optional[Dict[str, int]] = None
    ) -> Union[str, int]:
        """
        load classification annotation of the record `rec` or the subject `sid`

        Parameters
        ----------
        rec_or_sid: str or int,
            the record name or the index of the record in `self.all_records`
            or the subject id
        class_map: dict, optional,
            the mapping of the annotation classes

        Returns
        -------
        ann: str or int,
            the class of the record,
            or the number of the class if `class_map` is provided

        """
        pass

    @property
    def all_subjects(self) -> List[str]:
        return self._all_subjects

    @property
    def subject_records(self) -> Dict[str, List[str]]:
        return self._subject_records

    def plot(self, rec: Union[str, int], **kwargs) -> None:
        """
        plot the record `rec`, with metadata and segmentation

        Parameters
        ----------
        rec: str or int,
            the record name or the index of the record in `self.all_records`
        kwargs: dict,
            not used currently

        Returns
        -------
        fig: matplotlib.figure.Figure,
            the figure of the record
        ax: matplotlib.axes.Axes,
            the axes of the figure

        """
        pass

    @property
    def database_info(self) -> DataBaseInfo:
        return _CINC2023_INFO
