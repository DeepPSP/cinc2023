"""
"""

import re
import warnings
from ast import literal_eval
from pathlib import Path
from typing import Union, Optional, Any, List, Dict, Sequence, Tuple

import gdown
import numpy as np
import pandas as pd
import wfdb
import scipy.signal as SS
from tqdm.auto import tqdm
from torch_ecg.cfg import DEFAULTS
from torch_ecg.databases.base import PhysioNetDataBase, DataBaseInfo
from torch_ecg.utils.misc import get_record_list_recursive3, add_docstring
from torch_ecg.utils.download import _untar_file

from cfg import BaseCfg
from utils.misc import load_unofficial_phase_metadata


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
    3. The database consists of clinical, EEG, and ECG data from adult patients with out-of-hospital or in-hospital cardiac arrest who had return of heart function (i.e., return of spontaneous circulation [ROSC]) but remained comatose - defined as the inability to follow verbal commands and a Glasgow Coma Score <= 8.
    4. EEG data (and also other types of data) have varying sampling rates (500, 512, 256, 2048, 250, 200, 1024).
    5. The recordings are segmented every hour, and each segment can start at any time of the hour and ends at the end of the hour or when the EEG recording ends, whichever comes first.
    6. Each EEG recording contains an array of varying length EEG signals from 19-21 channels. The public training data share a common set of 19 channels.
    7. The voltage values of the each EEG signals are relative to a **unknown** reference potential. Therefore, one has to use the differences between pairs of channels. For a system (surface potential field) of N channels, the minimum number of channels required to reconstruct the EEG signals is N-1, hence a deep learning model for CinC2023 which accepts raw EEG signals as input should have at least 18 input channels. One can use the 18 pairs from the unofficial phase or choose a common reference channel from the 19 common channels (e.g. Pz) and use the 18 pairs of differences between the reference channel and the other 18 channels.
    8. The EEG recordings for one patient continue for several hours to days, so the EEG signals are prone to quality deterioration from non-physiological artifacts. ~~Only the **cleanest 5 minutes** of EEG data per hour are provided.~~
    9. There might be gaps in the EEG data, since patients may have EEG started several hours after the arrest or need to have brain monitoring interrupted transiently while in the ICU.
    10. Pattern for the data files: <patient_id>_<segment_id>_<hour>_<signal_type>.mat
    11. 4 types (groups) of signals were collected. In addition to EEG data, there are 3 (optional) other groups: ECG, REF, OTHER. The signals have the following channels:

        - EEG: Fp1, Fp2, F7, F8, F3, F4, T3, T4, C3, C4, T5, T6, P3, P4, O1, O2, Fz, Cz, Pz, Fpz, Oz, F9
        - ECG: ECG, ECG1, ECG2, ECGL, ECGR
        - REF: RAT1, RAT2, REF, C2, A1, A2, BIP1, BIP2, BIP3, BIP4, Cb2, M1, M2, In1-Ref2, In1-Ref3
        - OTHER: SpO2, EMG1, EMG2, EMG3, LAT1, LAT2, LOC, ROC, LEG1, LEG2

    Note that the following channels do not appear in the public training set:

        - EEG: Oz
        - REF: A1, A2, BIP1, BIP2, BIP3, BIP4, C2, Cb2, In1-Ref2

    12. Each patient has one .txt file containing patient information (ref. 13) and clinical outcome (ref. 14).
    13. Patient information includes information recorded at the time of admission (age, sex), identifier of the hospital where the data was collected (hospital), location of arrest (out or in-hospital), type of cardiac rhythm recorded at the time of resuscitation (shockable rhythms include ventricular fibrillation or ventricular tachycardia and non-shockable rhythms include asystole and pulseless electrical activity), and the time between cardiac arrest and ROSC (return of spontaneous circulation). The following table summarizes the patient information:

        +----------------+-----------------------------------------------+-----------------------------------------+
        |  info          |   meaning                                     |   type and values                       |
        +================+===============================================+=========================================+
        |  Hospital      |   identifier of the hospital where the data   |   categorical                           |
        |                |   is collected                                |   A, B, C, D, E, F, G                   |
        +----------------+-----------------------------------------------+-----------------------------------------+
        |  Age           |   Age (in years)                              |   continuous                            |
        +----------------+-----------------------------------------------+-----------------------------------------+
        |  Sex           |   Sex                                         |   categorical                           |
        +----------------+-----------------------------------------------+-----------------------------------------+
        |  ROSC          |   time from cardiac arrest to return of       |   continuous                            |
        |                |   spontaneous circulation, in minutes         |                                         |
        +----------------+-----------------------------------------------+-----------------------------------------+
        |  OHCA          |   out-of-hospital cardiac arrest              |   categorical (boolean)                 |
        |                |                                               |   True = out of hospital cardiac arrest |
        |                |                                               |   False = in-hospital cardiac arrest    |
        +----------------+-----------------------------------------------+-----------------------------------------+
        |  Shockable     |   ventricular fibrillation                    |   categorical (boolean)                 |
        |  Rhythm        |                                               |   True = shockable rhythm               |
        |                |                                               |   False = non-shockable rhythm          |
        +----------------+-----------------------------------------------+-----------------------------------------+
        |  TTM           |   targeted temperature management,            |   continuous (indeed, categorical)      |
        |                |   in Celsius                                  |   33, 36, or NaN for no TTM             |
        +----------------+-----------------------------------------------+-----------------------------------------+

    14. Clinical outcome was determined prospectively in two centers by phone interview (at 6 months from ROSC), and at the remaining hospitals retrospectively through chart review (at 3-6 months from ROSC). Neurological function was determined using the Cerebral Performance Category (CPC) scale. CPC is an ordinal scale ranging from 1 to 5:

        - CPC = 1: good neurological function and independent for activities of daily living.
        - CPC = 2: moderate neurological disability but independent for activities of daily living.
        - CPC = 3: severe neurological disability.
        - CPC = 4: unresponsive wakefulness syndrome [previously known as vegetative state].
        - CPC = 5: dead.

    15. The CPC scores are grouped into two categories:

        - Good: CPC = 1 or 2.
        - Poor: CPC = 3, 4, or 5.

    """,
    usage=[
        "Neurological recovery prediction",
    ],
    note="""
    """,
    issues="""
    """,
    references=[
        "https://moody-challenge.physionet.org/2023/",
        "https://physionet.org/content/i-care/",
    ],
    # doi=["https://doi.org/10.13026/rjbz-cq89"],
    doi=["https://doi.org/10.13026/avek-0p97"],
)


@add_docstring(_CINC2023_INFO.format_database_docstring(), mode="prepend")
class CINC2023Reader(PhysioNetDataBase):
    """
    Parameters
    ----------
    db_dir : str or pathlib.Path
        Local storage path of the database.
    fs : int, default 100
        (Re-)sampling frequency of the recordings.
    backend : {"scipy",  "wfdb"}, optional
        Backend to use, by default "wfdb", case insensitive.
    eeg_channel_pairs : list of str, optional
        List of EEG channel pairs for bipolar referencing.
        Each element is a string of two channel names separated by a hyphen.
    eeg_reference_channel: str, optional
        Name of the channel to use as reference for EEG channels.
        Valid if `eeg_channel_pairs` is None.
        If both `eeg_channel_pairs` and `eeg_reference_channel` are None,
        `self.default_eeg_channel_pairs` will be used.
    working_dir : str, optional
        Working directory, to store intermediate files and log files.
    hour_limit : int, optional
        If not None, only the recordings recorded within the first
        `hour_limit` hours will be visiable to the reader.
    verbose: int, default 2
        Verbosity level for logging.
    kwargs : dict, optional
        Auxilliary key word arguments.

    """

    __name__ = "CINC2023Reader"

    # fmt: off
    channel_names = {
        "EEG": [
            "Fp1", "Fp2", "F7", "F8", "F3", "F4", "T3", "T4", "C3", "C4",
            "T5", "T6", "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz", "Fpz", "Oz", "F9",
        ],
        "ECG": [
            "ECG", "ECG1", "ECG2", "ECGL", "ECGR",
        ],
        "REF": [
            "RAT1", "RAT2", "REF", "C2", "A1", "A2", "BIP1", "BIP2", "BIP3", "BIP4",
            "Cb2", "M1", "M2", "In1-Ref2", "In1-Ref3",
        ],
        "OTHER": [
            "SpO2", "EMG1", "EMG2", "EMG3", "LAT1", "LAT2", "LOC", "ROC", "LEG1", "LEG2",
        ],
    }
    common_eeg_channels = BaseCfg.common_eeg_channels
    default_eeg_channel_pairs = BaseCfg.eeg_channel_pairs
    default_eeg_reference_channel = None
    # fmt: on

    _channel_names_to_signal_types = {
        item: name for name, items in channel_names.items() for item in items
    }

    _rec_pattern = BaseCfg.recording_pattern

    _url_compressed = {
        "full": (
            "https://physionet.org/static/published-projects/i-care/"
            "i-care-international-cardiac-arrest-research-consortium-database-2.0.zip"
        ),
        "subset": "https://drive.google.com/u/0/uc?id=13IAz0mZIyT4X18izeSClj2veE9E09vop",
    }

    def __init__(
        self,
        db_dir: str,
        fs: int = 100,
        backend: str = "wfdb",
        eeg_channel_pairs: Optional[List[str]] = None,
        eeg_reference_channel: Optional[str] = None,
        working_dir: Optional[str] = None,
        hour_limit: Optional[int] = None,
        verbose: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            db_name="i-care",
            db_dir=db_dir,
            fs=fs,
            backend=backend,
            working_dir=working_dir,
            verbose=verbose,
            **kwargs,
        )
        self.fs = fs
        self.backend = backend
        self.hour_limit = hour_limit
        self.eeg_channel_pairs = eeg_channel_pairs
        if self.eeg_channel_pairs is not None:
            if eeg_reference_channel is not None:
                warnings.warn(
                    "Both `eeg_channel_pairs` and `eeg_reference_channel` are provided, "
                    "the latter will be ignored.",
                    RuntimeWarning,
                )
            self.eeg_reference_channel = None
        elif eeg_reference_channel is not None:
            self.eeg_reference_channel = eeg_reference_channel
            self.eeg_channel_pairs = [
                f"{ch}-{eeg_reference_channel}"
                for ch in self.common_eeg_channels
                if ch != eeg_reference_channel
            ]
        else:
            self.eeg_reference_channel = self.default_eeg_reference_channel
            self.eeg_channel_pairs = self.default_eeg_channel_pairs
        self.dtype = kwargs.get("dtype", BaseCfg.np_dtype)

        self.data_ext = "mat"
        self.header_ext = "hea"
        # self.quality_ext = "tsv"
        self.ann_ext = "txt"

        self.records_file = self.db_dir / "RECORDS-NEW"
        self.records_metadata_file = self.db_dir / "RECORDS.csv"
        self.subjects_metadata_file = self.db_dir / "SUBJECTS.csv"

        self._df_records_all_bak = None
        self._df_records_all = None
        self._df_records_bak = None
        self._df_records = None
        self._df_subjects = None
        self._all_records_all = None
        self._all_records = None
        self._all_subjects = None
        self._subject_records_all = None
        self._subject_records = None
        self._df_unofficial_phase_metadata = None
        self._ls_rec()

    def _auto_infer_units(self) -> None:
        """Auto infer the units of the signals."""
        raise NotImplementedError

    def _reset_fs(self, new_fs: int) -> None:
        """Reset the default sampling frequency of the database."""
        self.fs = new_fs

    def _reset_hour_limit(self, new_hour_limit: Union[int, None]) -> None:
        """Reset the hour limit of the database."""
        self.hour_limit = new_hour_limit

        if self.hour_limit is not None:
            self._df_records_all = self._df_records_all_bak[
                self._df_records_all_bak.hour <= self.hour_limit
            ]
            self._df_records = self._df_records_bak[
                self._df_records_bak.hour <= self.hour_limit
            ]
        else:
            self._df_records_all = self._df_records_all_bak.copy()
            self._df_records = self._df_records_bak.copy()

        self._all_records_all = {
            sig_type: self._df_records_all[
                self._df_records_all.sig_type == sig_type
            ].index.tolist()
            for sig_type in self._df_records_all.sig_type.unique().tolist()
        }
        self._subject_records_all = {
            sbj: self._df_records_all.loc[
                self._df_records_all["subject"] == sbj
            ].index.tolist()
            for sbj in self._all_subjects
        }
        self._subject_records = {
            sbj: self._df_records.loc[self._df_records["subject"] == sbj].index.tolist()
            for sbj in self._all_subjects
        }
        self._all_records = self._df_records.index.tolist()

    def _ls_rec(self) -> None:
        """Find all records in the database directory
        and store them (path, metadata, etc.) in a dataframe.
        """
        # fmt: off
        records_index = "record"
        records_cols = [
            "subject", "path", "sig_type",
            # "hour", "time", "quality",
            "hour", "start_sec", "end_sec", "utility_freq",
            "fs", "sig_len", "n_sig", "sig_name",
            "diff_inds",
        ]
        subjects_index = "subject"
        subjects_cols = [
            "Directory",
            "Hospital", "Age", "Sex", "ROSC", "OHCA", "Shockable Rhythm", "TTM",
            "Outcome", "CPC",
        ]
        # fmt: on
        eeg_channel_pairs = [
            [pair.split("-")[0] for pair in self.eeg_channel_pairs],
            [pair.split("-")[1] for pair in self.eeg_channel_pairs],
        ]

        self._df_records_all = pd.DataFrame(columns=[records_index] + records_cols)
        self._df_subjects = pd.DataFrame(columns=[subjects_index] + subjects_cols)

        cache_exists = (
            self.records_file.exists()
            and self.records_metadata_file.exists()
            and self.subjects_metadata_file.exists()
        )
        write_files = False

        # load from cache if exists
        if cache_exists:
            self._df_records_all = pd.read_csv(
                self.records_metadata_file, index_col="record"
            )
            self._df_records_all["subject"] = self._df_records_all["subject"].apply(
                lambda x: f"{x:04d}"
            )
            self._df_records_all["path"] = self._df_records_all["path"].apply(
                lambda x: Path(x).resolve()
            )
            self._df_records_all["sig_name"] = self._df_records_all["sig_name"].apply(
                literal_eval
            )  # cells from str to list
            self._df_records_all["diff_inds"] = self._df_records_all["diff_inds"].apply(
                literal_eval
            )  # cells from str to list
            self._df_subjects = pd.read_csv(
                self.subjects_metadata_file, index_col="subject"
            )
            self._df_subjects.index = self._df_subjects.index.map(lambda x: f"{x:04d}")
            self._df_subjects["CPC"] = self._df_subjects["CPC"].apply(str)
            self._df_subjects["Directory"] = self._df_subjects["Directory"].apply(
                lambda x: Path(x).resolve()
            )
        elif self._subsample is None:
            write_files = True

        if not self._df_records_all.empty:
            # filter out records that do not have data files
            data_suffix = f".{self.data_ext}"
            self._df_records_all = self._df_records_all[
                self._df_records_all["path"].apply(
                    lambda x: Path(x).with_suffix(data_suffix).exists()
                )
            ]

        # collect all records in the database directory recursively
        # if cache does not exist
        if len(self._df_records_all) == 0:
            if self._subsample is None:
                write_files = True
            self._df_records_all["path"] = get_record_list_recursive3(
                self.db_dir, f"{self._rec_pattern}\\.{self.data_ext}", relative=False
            )
            self._df_records_all["path"] = self._df_records_all["path"].apply(
                lambda x: Path(x)
            )

            self._df_records_all["record"] = self._df_records_all["path"].apply(
                lambda x: x.stem
            )
            self._df_records_all["subject"] = self._df_records_all["record"].apply(
                lambda x: re.match(self._rec_pattern, x).group("sbj")
            )
            self._df_records_all["sig_type"] = self._df_records_all["record"].apply(
                lambda x: re.match(self._rec_pattern, x).group("sig")
            )
            self._df_records_all["hour"] = (
                self._df_records_all["record"]
                .apply(lambda x: re.match(self._rec_pattern, x).group("hour"))
                .astype(int)
            )

            self._df_records_all = self._df_records_all.sort_values(by="record")
            self._df_records_all.set_index("record", inplace=True)

            # collect metadata for each record from header files
            for extra_col in [
                "fs",
                "sig_len",
                "n_sig",
                "sig_name",
                "start_sec",
                "end_sec",
                "utility_freq",
            ]:
                self._df_records_all[extra_col] = None
            pattern = (  # not start with "#"
                "Utility frequency: (?P<utility_frequency>\\d+)\\n"
                "Start time: (?P<start_hour>\\d+):(?P<start_minute>\\d+):(?P<start_second>\\d+)\\n"
                "End time: (?P<end_hour>\\d+):(?P<end_minute>\\d+):(?P<end_second>\\d+)"
            )
            if not self._df_records_all.empty:
                with tqdm(
                    self._df_records_all.iterrows(),
                    total=len(self._df_records_all),
                    dynamic_ncols=True,
                    mininterval=1.0,
                    desc="Collecting recording metadata",
                ) as pbar:
                    for idx, row in pbar:
                        header = wfdb.rdheader(str(row.path))
                        for extra_col in ["fs", "sig_len", "n_sig", "sig_name"]:
                            self._df_records_all.at[idx, extra_col] = getattr(
                                header, extra_col
                            )
                        # assign "diff-inds" column for EEG records
                        if row.sig_type != "EEG":
                            diff_inds = []
                        else:
                            diff_inds = [
                                [header.sig_name.index(item) for item in lst]
                                for lst in eeg_channel_pairs
                            ]
                        self._df_records_all.at[idx, "diff_inds"] = diff_inds
                        # assign "start_sec", "end_sec" and "utility_freq" columns
                        # which are comments in the header file
                        d = re.search(pattern, "\n".join(header.comments)).groupdict()
                        self._df_records_all.at[idx, "start_sec"] = int(
                            d["start_minute"]
                        ) * 60 + int(d["start_second"])
                        # plus 1 to end_sec to make it exclusive
                        # i.e. [start_sec, end_sec)
                        self._df_records_all.at[idx, "end_sec"] = (
                            int(d["end_minute"]) * 60 + int(d["end_second"]) + 1
                        )
                        self._df_records_all.at[idx, "utility_freq"] = int(
                            d["utility_frequency"]
                        )
                for extra_col in ["fs", "sig_len", "n_sig"]:
                    self._df_records_all[extra_col] = self._df_records_all[
                        extra_col
                    ].astype(int)

        if len(self._df_records_all) > 0 and self._subsample is not None:
            all_subjects = self._df_records_all["subject"].unique().tolist()
            size = min(
                len(all_subjects),
                max(1, int(round(self._subsample * len(all_subjects)))),
            )
            self.logger.debug(f"subsample `{size}` subjects from `{len(all_subjects)}`")
            all_subjects = DEFAULTS.RNG.choice(all_subjects, size=size, replace=False)
            self._df_records_all = self._df_records_all.loc[
                self._df_records_all["subject"].isin(all_subjects)
            ].sort_values(by="record")

        self._all_subjects = self._df_records_all["subject"].unique().tolist()

        # collect subject metadata from the .txt files
        if self._df_subjects.empty and len(self._all_subjects) > 0:
            metadata_list = []
            with tqdm(
                self._all_subjects,
                total=len(self._all_subjects),
                dynamic_ncols=True,
                mininterval=1.0,
                desc="Collecting subject metadata",
            ) as pbar:
                for sbj in pbar:
                    file_path = (
                        self._df_records_all.loc[self._df_records_all["subject"] == sbj]
                        .iloc[0]["path"]
                        .parent
                        / f"{sbj}.txt"
                    )
                    metadata = {
                        k.strip(): v.strip()
                        for k, v in [
                            line.split(":")
                            for line in file_path.read_text().splitlines()
                        ]
                    }
                    metadata["subject"] = sbj
                    metadata["Directory"] = file_path.parent
                    metadata_list.append(metadata)
            self._df_subjects = pd.DataFrame(
                metadata_list, columns=["subject"] + subjects_cols
            )
            self._df_subjects.set_index("subject", inplace=True)
            self._df_subjects = self._df_subjects[subjects_cols]
        else:
            self._df_subjects = self._df_subjects[
                self._df_subjects.index.isin(self._all_subjects)
            ]

        if self._df_records_all.empty or self._df_subjects.empty:
            write_files = False

        if write_files:
            self.records_file.write_text(
                "\n".join(
                    self._df_records_all["path"]
                    .apply(lambda x: x.relative_to(self.db_dir).as_posix())
                    .tolist()
                )
            )
            self._df_records_all.to_csv(self.records_metadata_file)
            self._df_subjects.to_csv(self.subjects_metadata_file)

        self._df_records = self._df_records_all[
            self._df_records_all["sig_type"] == "EEG"
        ]
        for aux_sig in ["ECG", "REF", "OTHER"]:
            df_tmp = self._df_records_all[self._df_records_all["sig_type"] == aux_sig]
            df_tmp.index = df_tmp.index.map(lambda x: x.replace(aux_sig, "EEG"))
            df_tmp = df_tmp.assign(aux_sig=True)
            df_tmp = df_tmp[["aux_sig"]]
            df_tmp.columns = [aux_sig]
            # merge self._df_records and df_tmp
            self._df_records = self._df_records.join(df_tmp, how="outer")
            # fill NaNs with False
            self._df_records[aux_sig].fillna(False, inplace=True)
            del df_tmp

        self._df_records_all_bak = self._df_records_all.copy()
        self._df_records_bak = self._df_records.copy()
        # restrict to the records with "hour" column <= self.hour_limit
        self._reset_hour_limit(self.hour_limit)

        self._df_unofficial_phase_metadata = load_unofficial_phase_metadata()

    def clear_cached_metadata_files(self) -> None:
        """Remove the cached metadata files if they exist."""
        if self.records_file.exists():
            # `Path.unlink` in Python 3.6 does NOT have the `missing_ok` parameter
            self.records_file.unlink()
        if self.records_metadata_file.exists():
            self.records_metadata_file.unlink()
        if self.subjects_metadata_file.exists():
            self.subjects_metadata_file.unlink()

    def get_subject_id(self, rec_or_sbj: Union[str, int]) -> str:
        """Attach a unique subject ID for the record.

        Parameters
        ----------
        rec_or_sbj : str or int
            Record name or index of the record in :attr:`all_records`
            or subject name

        Returns
        -------
        str
            Subject ID associated with the record.

        """
        if isinstance(rec_or_sbj, int):
            rec_or_sbj = self[rec_or_sbj]
        if rec_or_sbj in self.all_records:
            return self._df_records.loc[rec_or_sbj, "subject"]
        elif rec_or_sbj in self.all_subjects:
            return rec_or_sbj
        else:
            raise ValueError(f"record or subject `{rec_or_sbj}` not found")

    def get_absolute_path(
        self,
        rec_or_sbj: Union[str, int],
        signal_type: Optional[str] = None,
        extension: Optional[str] = None,
    ) -> Path:
        """Get the absolute path of the record.

        Parameters
        ----------
        rec_or_sbj : str or int
            Record name or index of the record in :attr:`all_records`
            or subject name.
        signal_type : {"EEG", "ECG", "REF", "OTHER"}, optional
            Type of the signal.
            Can be directly passed as a part of the record name.
        extension : str, optional
            Extension of the file.

        Returns
        -------
        pathlib.Path
            Absolute path of the file or directory.

        """
        if isinstance(rec_or_sbj, int):
            rec_or_sbj = self[rec_or_sbj]
            if signal_type is not None:
                rec_or_sbj = rec_or_sbj.replace("EEG", signal_type)
        if rec_or_sbj in self._df_records_all.index:
            path = self._df_records_all.loc[rec_or_sbj, "path"]
        elif rec_or_sbj in self.all_subjects:
            path = self._df_subjects.loc[rec_or_sbj, "Directory"]
            if extension is not None:
                path = path / f"{rec_or_sbj}"
        else:
            raise FileNotFoundError(f"record or subject `{rec_or_sbj}` not found")
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
        units: Union[str, type(None)] = "uV",
        fs: Optional[int] = None,
    ) -> np.ndarray:
        """Load EEG data from the record.

        Parameters
        ----------
        rec : str or int
            Record name or the index of the record in :attr:`all_records`.
        channels : str or int or Sequence[str] or Sequence[int], optional
            Names or indices of the channel(s) to load.
            If is None, all channels will be loaded.
        sampfrom : int, optional
            Start index of the data to be loaded.
        sampto : int, optional
            End index of the data to be loaded.
        data_format : {"channel_first", "channel_last", "flat", "plain"}
            Format of the data, default "channel_first".
            Can be "flat" (alias "plain") if `channels` is a single channel.
            Case insensitive.
        units : str or None, default "uV"
            Units of the data, can be one of
            "mV", "uV" (with alias "muV", "μV"), case insensitive.
            None for digital data, without digital-to-physical conversion.
            NOTE: non-null `units` are treated identically to get the physical values,
            since the physical units for the data are missing
            because some of the data had already been scaled by the data sources
        fs : int, optional
            Sampling frequency of the record,
            defaults to `self.fs` if `self.fs` is set
            else defaults to the raw sampling frequency of the record.

        Returns
        -------
        data : numpy.ndarray
            The loaded EEG data.

        """
        if isinstance(rec, int):
            rec = self[rec]
        fp = str(self.get_absolute_path(rec))
        rdrecord_kwargs = dict()
        # normalize channels
        if channels is not None:
            if isinstance(channels, (str, int)):
                channels = [channels]
            channels = [
                self._df_records.loc[rec, "sig_name"].index(chn)
                if isinstance(chn, str)
                else chn
                for chn in channels
            ]
            rdrecord_kwargs["channels"] = channels
            n_channels = len(channels)
        else:
            n_channels = self._df_records.loc[rec, "n_sig"]
        allowed_data_format = ["channel_first", "channel_last", "flat", "plain"]
        assert (
            data_format.lower() in allowed_data_format
        ), f"`data_format` should be one of `{allowed_data_format}`, but got `{data_format}`"
        if n_channels > 1:
            assert data_format.lower() in ["channel_first", "channel_last"], (
                "`data_format` should be one of `['channel_first', 'channel_last']` "
                f"when the passed number of `channels` is larger than 1, but got `{data_format}`"
            )

        allowed_units = ["mv", "uv", "μv", "muv"]
        assert (
            units is None or units.lower() in allowed_units
        ), f"`units` should be one of `{allowed_units}` or None, but got `{units}`"

        rdrecord_kwargs.update(
            dict(
                sampfrom=sampfrom or 0,
                sampto=sampto,
                physical=units is not None,
                return_res=DEFAULTS.DTYPE.INT,
            )
        )
        wfdb_rec = wfdb.rdrecord(fp, **rdrecord_kwargs)

        # p_signal or d_signal is in the format of "channel_last", and with units in "μV"
        if units is None:
            data = wfdb_rec.d_signal
        else:
            data = wfdb_rec.p_signal
        # elif units.lower() in ["μv", "uv", "muv"]:
        #     data = wfdb_rec.p_signal
        # elif units.lower() == "mv":
        #     data = wfdb_rec.p_signal / 1000

        fs = fs or self.fs
        if fs is not None and fs != wfdb_rec.fs:
            data = SS.resample_poly(data, fs, wfdb_rec.fs, axis=0).astype(data.dtype)
        else:
            fs = wfdb_rec.fs

        if data_format.lower() == "channel_first":
            data = data.T
        elif data_format.lower() in ["flat", "plain"]:
            data = data.flatten()

        return data

    def load_bipolar_data(
        self,
        rec: Union[str, int],
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        data_format: str = "channel_first",
        units: Union[str, type(None)] = "uV",
        fs: Optional[int] = None,
    ) -> np.ndarray:
        """Load bipolar EEG data from the record.

        Bipolar EEG is the difference between two channels.
        Ref. `self.eeg_channel_pairs`.

        Parameters
        ----------
        rec : str or int
            Record name or the index of the record in :attr:`all_records`.
        sampfrom : int, optional
            Start index of the data to be loaded.
        sampto : int, optional
            End index of the data to be loaded.
        data_format : {"channel_first", "channel_last"}
            Format of the data, default "channel_first".
            Case insensitive.
        units : str or None, default "uV"
            Units of the data, can be one of
            "mV", "uV" (with alias "muV", "μV"), case insensitive.
            None for digital data, without digital-to-physical conversion.
            NOTE: non-null `units` are treated identically to get the physical values,
            since the physical units for the data are missing
            because some of the data had already been scaled by the data sources
        fs : int, optional
            Sampling frequency of the record,
            defaults to `self.fs` if `self.fs` is set
            else defaults to the raw sampling frequency of the record.

        Returns
        -------
        data : numpy.ndarray
            The loaded EEG data.

        """
        if isinstance(rec, int):
            rec = self[rec]
        fp = str(self.get_absolute_path(rec))
        metadata_row = self._df_records.loc[rec]
        allowed_data_format = ["channel_first", "channel_last"]
        assert (
            data_format.lower() in allowed_data_format
        ), f"`data_format` should be one of `{allowed_data_format}`, but got `{data_format}`"

        allowed_units = ["mv", "uv", "μv", "muv"]
        assert (
            units is None or units.lower() in allowed_units
        ), f"`units` should be one of `{allowed_units}` or None, but got `{units}`"

        rdrecord_kwargs = dict(
            sampfrom=sampfrom or 0,
            sampto=sampto,
            physical=units is not None,
            return_res=DEFAULTS.DTYPE.INT,
        )
        wfdb_rec = wfdb.rdrecord(fp, **rdrecord_kwargs)

        # p_signal or d_signal is in the format of "channel_last", and with units in "μV"
        if units is None:
            data = wfdb_rec.d_signal
        else:
            data = wfdb_rec.p_signal
        # elif units.lower() in ["μv", "uv", "muv"]:
        #     data = wfdb_rec.p_signal
        # elif units.lower() == "mv":
        #     data = wfdb_rec.p_signal / 1000

        data = (
            data[:, metadata_row["diff_inds"][0]]
            - data[:, metadata_row["diff_inds"][1]]
        )

        fs = fs or self.fs
        if fs is not None and fs != wfdb_rec.fs:
            data = SS.resample_poly(data, fs, wfdb_rec.fs, axis=0).astype(data.dtype)
        else:
            fs = wfdb_rec.fs

        if data_format.lower() == "channel_first":
            data = data.T

        return data

    def load_aux_data(
        self,
        rec: Union[str, int],
        signal_type: Optional[str] = None,
        channels: Optional[Union[str, int, Sequence[Union[str, int]]]] = None,
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        data_format: str = "channel_first",
        fs: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[str], int]:
        """Load auxiliary (**physical**) data from the record.

        Parameters
        ----------
        rec : str or int
            Record name or the index of the record in :attr:`all_records`.
            Note that if `rec` is of type int, then the recording
            would be inferred from the `signal_type` and `channels` that
            corresponds to the EEG recording, which might not exist.
        signal_type : {"ECG", "REF", "OTHER"}
            Type of the auxiliary data.
            If is None, `channels` should be provided,
            and `signal_type` will be inferred from the channel names;
            or `rec` is of type str, and `signal_type` will be inferred
        channels : str or int or Sequence[str] or Sequence[int], optional
            Names or indices of the channel(s) to load.
            If is None, all channels will be loaded.
        sampfrom : int, optional
            Start index of the data to be loaded.
        sampto : int, optional
            End index of the data to be loaded.
        data_format : {"channel_first", "channel_last", "flat", "plain"}
            Format of the data, default "channel_first".
            Can be "flat" (alias "plain") if `channels` is a single channel.
            Case insensitive.
        fs : int, optional
            Sampling frequency of the record,
            defaults to the raw sampling frequency of the record.
            NOTE the behavior of `fs` is different from that of :meth:`load_data`
            for loading EEG data.

        Returns
        -------
        data : numpy.ndarray
            The loaded auxiliary data.
        channels : list of str
            Names of the loaded channels.
        fs : int
            Sampling frequency of the loaded data.

        """
        if isinstance(rec, int):
            rec = self[rec]
        else:
            signal_type = self._df_records_all.loc[rec, "sig_type"].index[0]
        if signal_type is None:
            assert (
                channels is not None
            ), "`signal_type` should be provided when `channels` is None"
            if isinstance(channels, str):
                signal_type = self._channel_names_to_signal_types[channels]
            elif isinstance(channels, (list, tuple)):
                signal_type = self._channel_names_to_signal_types[channels[0]]
            else:
                raise TypeError(
                    f"Could not determine `signal_type` from `channels` of type `{type(channels)}`"
                )
        else:
            # if rec is obtained from the index of all_records
            rec = rec.replace("EEG", signal_type)
        fp = str(self.get_absolute_path(rec, signal_type))
        rec = Path(fp).stem
        rdrecord_kwargs = dict()
        # normalize channels
        if channels is not None:
            if isinstance(channels, (str, int)):
                channels = [channels]
            channels = [
                self._df_records_all.loc[rec, "sig_name"].index(chn)
                if isinstance(chn, str)
                else chn
                for chn in channels
            ]
            rdrecord_kwargs["channels"] = channels
            n_channels = len(channels)
        else:
            n_channels = self._df_records_all.loc[rec, "n_sig"]
        allowed_data_format = ["channel_first", "channel_last", "flat", "plain"]
        assert (
            data_format.lower() in allowed_data_format
        ), f"`data_format` should be one of `{allowed_data_format}`, but got `{data_format}`"
        if n_channels > 1:
            assert data_format.lower() in ["channel_first", "channel_last"], (
                "`data_format` should be one of `['channel_first', 'channel_last']` "
                f"when the passed number of `channels` is larger than 1, but got `{data_format}`"
            )

        rdrecord_kwargs.update(
            dict(
                sampfrom=sampfrom or 0,
                sampto=sampto,
                physical=True,
                return_res=DEFAULTS.DTYPE.INT,
            )
        )
        wfdb_rec = wfdb.rdrecord(fp, **rdrecord_kwargs)

        data = wfdb_rec.p_signal
        if fs is not None and fs != wfdb_rec.fs:
            data = SS.resample_poly(data, fs, wfdb_rec.fs, axis=0).astype(data.dtype)
        else:
            fs = wfdb_rec.fs

        if data_format.lower() == "channel_first":
            data = data.T
        elif data_format.lower() in ["flat", "plain"]:
            data = data.flatten()

        channels = wfdb_rec.sig_name

        return data, channels, fs

    def load_ann(self, rec_or_sbj: Union[str, int]) -> Dict[str, Union[str, int]]:
        """Load classification annotation corresponding to
        the record `rec` or the subject `sbj`.

        Parameters
        ----------
        rec_or_sbj : str or int
            Record name or the index of the record in :attr:`all_records`
            or the subject name.
        class_map : dict, optional
            Mapping of the annotation classes.

        Returns
        -------
        ann : dict
            A dictionary of annotation corresponding to
            the record or the subject, with items "outcome", "cpc".

        """
        subject = self.get_subject_id(rec_or_sbj)
        row = self._df_subjects.loc[subject]
        ann = dict(
            outcome=row["Outcome"],
            cpc=row["CPC"],
        )
        return ann

    def load_outcome(
        self, rec_or_sbj: Union[str, int], class_map: Optional[Dict[str, int]] = None
    ) -> Union[str, int]:
        """Load Outcome annotation corresponding to
        the record `rec` or the subject `sbj`.

        Parameters
        ----------
        rec_or_sbj : str or int
            Record name or the index of the record in :attr:`all_records`
            or the subject name.
        class_map : dict, optional
            Mapping of the annotation classes.

        Returns
        -------
        outcome : str or int
            The Outcome annotation corresponding to the record or the subject.
            If `class_map` is not None, the outcome will be mapped to the
            corresponding class index.

        """
        outcome = self.load_ann(rec_or_sbj)["outcome"]
        if class_map is not None:
            outcome = class_map[outcome]
        return outcome

    def load_cpc(
        self, rec_or_sbj: Union[str, int], class_map: Optional[Dict[str, int]] = None
    ) -> Union[str, int]:
        """Load CPC annotation corresponding to
        the record `rec` or the subject `sbj`.

        Parameters
        ----------
        rec_or_sbj : str or int
            Record name or the index of the record in :attr:`all_records`
            or the subject name.
        class_map : dict, optional
            Mapping of the annotation classes.

        Returns
        -------
        cpc : str or int
            The CPC annotation corresponding to the record or the subject.
            If `class_map` is not None, the outcome will be mapped to the
            corresponding class index.

        """
        cpc = self.load_ann(rec_or_sbj)["cpc"]
        if class_map is not None:
            cpc = class_map[cpc]
        return cpc

    def load_quality_table(self, sbj: Union[str, int]) -> pd.DataFrame:
        """Load recording quality table of the subject.

        Parameters
        ----------
        sbj : str or int
            Subject name or the index of the subject in :attr:`all_subjects`.

        Returns
        -------
        df_quality : pandas.DataFrame
            The quality table of the subject.

        """
        # if isinstance(sbj, int):
        #     sbj = self.all_subjects[sbj]
        # fp = self.get_absolute_path(sbj, self.quality_ext)
        # df_quality = (
        #     pd.read_csv(fp, sep="\t").dropna(subset=["Record"]).set_index("Record")
        # )
        # df_quality.index.name = "record"
        # return df_quality
        print("Quality tables are removed from the database starting from version 2.")
        return pd.DataFrame()

    def get_metadata(
        self, rec: Union[str, int], field: Optional[str] = None
    ) -> Union[Dict[str, Any], Any]:
        """Get metadata of the record.

        Metadata of the record includes the following fields:

        - "hour": the hour after cardiac arrest when the recording was recorded.
        - "start_sec": the start time of the recording in seconds in the hour.
        - "end_sec": the end time of the recording in seconds in the hour.
        - "utility_freq": the AC frequency of the apparatus used to record the signal.
        - "fs": the sampling frequency of the recording.
        - "sig_len": the length of the recording in samples.
        - "n_sig": the number of signals (channels) in the recording.
        - "sig_name": the names of the signals (channels) in the recording.
        - "ECG": whether or not the recording has simultaneous ECG signal.
        - "REF": whether or not the recording has simultaneous reference signal.
        - "OTHER": whether or not the recording has simultaneous other signal.

        Parameters
        ----------
        rec : str or int
            Record name or the index of the record in :attr:`all_records`.
        field : str, optional
            The field to return. If None, the whole metadata dictionary will be returned.

        Returns
        -------
        metadata : dict or any
            The metadata of the record.

        """
        if isinstance(rec, int):
            rec = self.all_records[rec]
        metadata = self._df_records.loc[rec].to_dict()
        for item in ["path", "sig_type"]:
            metadata.pop(item)
        if field is None:
            return metadata
        else:
            return metadata[field]

    @property
    def all_subjects(self) -> List[str]:
        return self._all_subjects

    @property
    def all_records(self) -> List[str]:
        return self._all_records

    @property
    def all_records_all(self) -> List[str]:
        return self._all_records_all

    @property
    def subject_records(self) -> Dict[str, List[str]]:
        return self._subject_records

    @property
    def subject_records_all(self) -> Dict[str, List[str]]:
        return self._subject_records_all

    def plot(self, rec: Union[str, int], **kwargs) -> None:
        """Plot the record with metadata and segmentation.

        Parameters
        ----------
        rec : str or int
            Record name or the index of the record in :attr:`all_records`.
        kwargs : dict, optional
            Additional keyword arguments for :func:`matplotlib.pyplot.plot`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure of the record.
        ax : matplotlib.axes.Axes
            The axes of the figure.

        """
        pass

    def download(self, full: bool = True) -> None:
        """Download the database from PhysioNet or Google Drive."""
        # url = self._url_compressed["full" if full else "subset"]
        # dl_file = "training.tar.gz" if full else "training_subset.tar.gz"
        # dl_file = str(self.db_dir / dl_file)
        # if full:
        #     http_get(url, self.db_dir, extract=True)
        # else:
        #     gdown.download(url, dl_file, quiet=False)
        #     _untar_file(dl_file, self.db_dir)
        # self._ls_rec()
        if full:
            print("The full database is too large.")
            print(
                "Please download from PhysioNet or Google Cloud Platform "
                "manually or using tools like `wget`, `gsutil`."
            )
            print(f"Webpage of the database at PhysioNet: {self.webpage}")
            print(
                f"Webpage of the database at Google Cloud Platform: {self.gcp_webpage}"
            )
        else:
            url = self._url_compressed["subset"]
            dl_file = str(self.db_dir / "training_subset.tar.gz")
            gdown.download(url, dl_file, quiet=False)
            _untar_file(dl_file, self.db_dir)
            self._ls_rec()

    @property
    def database_info(self) -> DataBaseInfo:
        return _CINC2023_INFO

    @property
    def gcp_webpage(self) -> str:
        return (
            "https://console.cloud.google.com/storage/browser/i-care-2.0.physionet.org/"
        )
