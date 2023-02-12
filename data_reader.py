"""
"""

import re
from pathlib import Path
from typing import Union, Optional, Any, List, Dict, Sequence

import gdown
import numpy as np
import pandas as pd
import wfdb
import scipy.signal as ss  # noqa: F401
from tqdm.auto import tqdm
from torch_ecg.cfg import DEFAULTS
from torch_ecg.databases.base import PhysioNetDataBase, DataBaseInfo
from torch_ecg.utils.misc import get_record_list_recursive3, add_docstring
from torch_ecg.utils.download import _untar_file

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

        self._url_compressed = {
            "full": "https://drive.google.com/u/0/uc?id=1MgIsfknRpRgR2jpVfzcR1Qwj0i6PC0-A",
            "subset": "https://drive.google.com/u/0/uc?id=1YGa1tFC0TzqBj8Uw32B47EwZ2KeiGUr2",
        }

        self._rec_pattern = "ICARE\\_(?P<sid>[\\d]+)\\_(?P<loc>[\\d]+)"
        self.data_ext = "mat"
        self.header_ext = "hea"
        self.quality_ext = "tsv"
        self.ann_ext = "txt"

        self._df_subjects = None
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
        write_file = False
        self._df_records = pd.DataFrame(columns=["record", "subject", "path"])
        records_file = self.db_dir / "RECORDS-NEW"
        if records_file.exists():
            self._df_records["record"] = records_file.read_text().splitlines()
            self._df_records["path"] = self._df_records["record"].apply(
                lambda x: self.db_dir / x
            )
        elif self._subsample is None:
            write_file = True

        if len(self._df_records) == 0:
            if self._subsample is None:
                write_file = True
            self._df_records["path"] = get_record_list_recursive3(
                self.db_dir, f"{self._rec_pattern}\\.{self.data_ext}", relative=False
            )
            self._df_records["path"] = self._df_records["path"].apply(lambda x: Path(x))

        self._df_records["record"] = self._df_records["path"].apply(lambda x: x.stem)
        self._df_records["subject"] = self._df_records["record"].apply(
            lambda x: re.match(self._rec_pattern, x).group("sid")
        )

        self._df_records["fs"] = 100
        self._df_records["siglen"] = 5 * 60 * self._df_records["fs"]
        self._df_records["n_sig"] = 18

        self._df_records = self._df_records.sort_values(by="record")
        self._df_records.set_index("record", inplace=True)

        if len(self._df_records) > 0:
            if self._subsample is not None:
                size = min(
                    len(self._df_records),
                    max(1, int(round(self._subsample * len(self._df_records)))),
                )
                self.logger.debug(
                    f"subsample `{size}` records from `{len(self._df_records)}`"
                )
                self._df_records = self._df_records.sample(
                    n=size, random_state=DEFAULTS.SEED, replace=False
                )

        self._all_records = self._df_records.index.tolist()
        self._all_subjects = self._df_records["subject"].unique().tolist()
        self._subject_records = {
            sid: self._df_records.loc[self._df_records["subject"] == sid].index.tolist()
            for sid in self._all_subjects
        }

        # collect subject metadata from the .txt files
        # self._df_subjects = pd.DataFrame(columns=["Age", "Sex", "ROSC", "OHCA", "VFib", "TTM", "Outcome", "CPC"])
        metadata_list = []
        with tqdm(
            self._all_subjects,
            total=len(self._all_subjects),
            dynamic_ncols=True,
            mininterval=1.0,
            desc="Collecting subject metadata",
        ) as pbar:
            for sid in pbar:
                file_path = (
                    self._df_records.loc[self._df_records["subject"] == sid]
                    .iloc[0]["path"]
                    .parent
                    / f"ICARE_{sid}.txt"
                )
                metadata = {
                    k.strip(): v.strip()
                    for k, v in [
                        line.split(":") for line in file_path.read_text().splitlines()
                    ]
                }
                metadata["subject"] = sid
                metadata_list.append(metadata)
        self._df_subjects = pd.DataFrame(metadata_list)
        self._df_subjects.set_index("subject", inplace=True)
        cols = ["Age", "Sex", "ROSC", "OHCA", "VFib", "TTM", "Outcome", "CPC"]
        self._df_subjects = self._df_subjects[cols]

        if write_file:
            records_file.write_text(
                "\n".join(
                    self._df_records["path"]
                    .apply(lambda x: x.relative_to(self.db_dir).as_posix())
                    .tolist()
                )
            )

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
        if isinstance(rec, int):
            rec = self[rec]
        path = self._df_records.loc[rec, "path"]
        if extension is not None and not extension.startswith("."):
            extension = f".{extension}"
        return path.with_suffix(extension or "").resolve()

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
    def webpage(self) -> str:
        # to update to the PhysioNet webpage
        "https://moody-challenge.physionet.org/2023/"

    @property
    def url(self) -> str:
        # return posixpath.join(
        #     wfdb.io.download.PN_INDEX_URL, f"{self.db_name}/{self.version}"
        # )
        return ""  # currently not available

    def download(self, full: bool = True) -> None:
        """
        download the database from Google Drive
        """
        url = self._url_compressed["full" if full else "subset"]
        dl_file = "training.tar.gz" if full else "training_subset.tar.gz"
        dl_file = str(self.db_dir / dl_file)
        gdown.download(url, dl_file, quiet=False)
        _untar_file(dl_file, self.db_dir)
        self._ls_rec()

    @property
    def database_info(self) -> DataBaseInfo:
        return _CINC2023_INFO
