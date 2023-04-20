"""
"""

import json
from copy import deepcopy
from pathlib import Path
from typing import Optional, List, Sequence, Dict

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch_ecg.cfg import CFG, DEFAULTS
from torch_ecg.utils.misc import ReprMixin, list_sum
from torch_ecg.utils.utils_data import stratified_train_test_split
from torch_ecg._preprocessors import PreprocManager
from tqdm import tqdm

from cfg import BaseCfg, TrainCfg, ModelCfg  # noqa: F401
from data_reader import CINC2023Reader


__all__ = [
    "CinC2023Dataset",
]


class CinC2023Dataset(Dataset, ReprMixin):
    """ """

    __name__ = "CinC2023Dataset"

    def __init__(
        self,
        config: CFG,
        task: str,
        training: bool = True,
        lazy: bool = True,
        **reader_kwargs,
    ) -> None:
        """
        Parameters
        ----------
        config : CFG
            configuration for the dataset
        task : str
            task to be performed using the dataset
        training : bool, default True
            whether the dataset is for training or validation
        lazy : bool, default True
            whether to load all data into memory at initialization
        reader_kwargs : dict, optional
            keyword arguments for the data reader class

        """
        super().__init__()
        self.config = CFG(deepcopy(config))
        # self.task = task.lower()  # task will be set in self.__set_task
        self.training = training
        self.lazy = lazy

        if self.config.get("db_dir", None) is None:
            self.config.db_dir = reader_kwargs.pop("db_dir", None)
            assert self.config.db_dir is not None, "db_dir must be specified"
        else:
            reader_kwargs.pop("db_dir", None)
        self.config.db_dir = Path(self.config.db_dir).expanduser().resolve()

        self.reader = CINC2023Reader(db_dir=self.config.db_dir, **reader_kwargs)

        self.subjects = self._train_test_split()
        self.records = list_sum(
            [self.reader.subject_records[sbj] for sbj in self.subjects]
        )
        if self.training:
            DEFAULTS.RNG.shuffle(self.records)

        if self.config.torch_dtype == torch.float64:
            self.dtype = np.float64
        else:
            self.dtype = np.float32

        self.__cache = None
        self.ppm = None
        self.__set_task(task, lazy)

    def __len__(self) -> int:
        if self.cache is None:
            self._load_all_data()
        return self.cache["waveforms"].shape[0]

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        if self.cache is None:
            self._load_all_data()
        return {k: v[index] for k, v in self.cache.items()}

    def __set_task(self, task: str, lazy: bool) -> None:
        """ """
        assert task.lower() in TrainCfg.tasks, f"illegal task \042{task}\042"
        if (
            hasattr(self, "task")
            and self.task == task.lower()
            and self.cache is not None
            and len(self.cache["waveforms"]) > 0
        ):
            return
        self.task = task.lower()

        ppm_config = CFG(random=False)
        ppm_config.update(deepcopy(self.config[self.task]))
        self.ppm = PreprocManager.from_config(ppm_config)

        self.siglen = int(self.config[self.task].fs * self.config[self.task].siglen)
        self.classes = self.config[task].classes
        self.n_classes = len(self.config[task].classes)
        self.lazy = lazy

        if self.task in ["classification"]:
            self.fdr = FastDataReader(
                self.reader, self.records, self.config, self.task, self.ppm
            )
        # elif self.task in ["multi_task"]:
        #     self.fdr = MutiTaskFastDataReader(
        #         self.reader, self.records, self.config, self.task, self.ppm
        #     )
        else:  # TODO: implement contrastive learning task
            raise ValueError("Illegal task")

        if self.lazy:
            return

        tmp_cache = []
        with tqdm(range(len(self.fdr)), desc="Loading data", unit="record") as pbar:
            for idx in pbar:
                tmp_cache.append(self.fdr[idx])
        keys = tmp_cache[0].keys()
        self.__cache = {k: np.concatenate([v[k] for v in tmp_cache]) for k in keys}
        # for k in keys:
        #     if self.__cache[k].ndim == 1:
        #         self.__cache[k] = self.__cache[k]

    def _load_all_data(self) -> None:
        """ """
        self.__set_task(self.task, lazy=False)

    def _train_test_split(
        self, train_ratio: float = 0.8, force_recompute: bool = False
    ) -> List[str]:
        """ """
        _train_ratio = int(train_ratio * 100)
        _test_ratio = 100 - _train_ratio
        assert _train_ratio * _test_ratio > 0

        train_file = self.reader.db_dir / f"train_ratio_{_train_ratio}.json"
        test_file = self.reader.db_dir / f"test_ratio_{_test_ratio}.json"
        (BaseCfg.project_dir / "utils").mkdir(exist_ok=True)
        aux_train_file = (
            BaseCfg.project_dir / "utils" / f"train_ratio_{_train_ratio}.json"
        )
        aux_test_file = BaseCfg.project_dir / "utils" / f"test_ratio_{_test_ratio}.json"

        if not force_recompute and train_file.exists() and test_file.exists():
            if self.training:
                return json.loads(train_file.read_text())
            else:
                return json.loads(test_file.read_text())

        if not force_recompute and aux_train_file.exists() and aux_test_file.exists():
            if self.training:
                return json.loads(aux_train_file.read_text())
            else:
                return json.loads(aux_test_file.read_text())

        df = self.reader._df_subjects.copy()
        df.loc[:, "Age"] = (
            df["Age"].fillna(df["Age"].mean()).astype(int)
        )  # only one nan
        # to age group
        df.loc[:, "Age"] = df["Age"].apply(lambda x: str(20 * (x // 20)))
        for col in ["OHCA", "VFib"]:
            df.loc[:, col] = df[col].apply(
                lambda x: 1 if x is True else 0 if x is False else x
            )
            df.loc[:, col] = df[col].fillna(-1).astype(int)
            df.loc[:, col] = df[col].astype(int).astype(str)

        df_train, df_test = stratified_train_test_split(
            df,
            [
                "Age",
                "Sex",
                "OHCA",
                "VFib",
                "CPC",
            ],
            test_ratio=1 - train_ratio,
            reset_index=False,
        )

        train_set = df_train.index.tolist()
        test_set = df_test.index.tolist()

        train_file.write_text(json.dumps(train_set, ensure_ascii=False))
        aux_train_file.write_text(json.dumps(train_set, ensure_ascii=False))
        test_file.write_text(json.dumps(test_set, ensure_ascii=False))
        aux_test_file.write_text(json.dumps(test_set, ensure_ascii=False))

        DEFAULTS.RNG.shuffle(train_set)
        DEFAULTS.RNG.shuffle(test_set)

        if self.training:
            return train_set
        else:
            return test_set

    @property
    def cache(self) -> List[Dict[str, np.ndarray]]:
        return self.__cache

    def extra_repr_keys(self) -> List[str]:
        return ["task", "training"]


class FastDataReader(ReprMixin, Dataset):
    """ """

    def __init__(
        self,
        reader: CINC2023Reader,
        records: Sequence[str],
        config: CFG,
        task: str,
        ppm: Optional[PreprocManager] = None,
    ) -> None:
        """ """
        self.reader = reader
        self.records = records
        self.config = config
        self.task = task
        self.ppm = ppm
        if self.config.torch_dtype == torch.float64:
            self.dtype = np.float64
        else:
            self.dtype = np.float32

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        rec = self.records[index]
        waveforms = self.reader.load_data(
            rec,
            data_format=self.config[self.task].data_format,
        )[np.newaxis, ...]
        if self.ppm:
            waveforms, _ = self.ppm(waveforms, self.reader.fs)
        label_cpc = self.reader.load_cpc(rec)
        if self.config[self.task].loss != "CrossEntropyLoss":
            label_cpc = np.isin(self.config[self.task].classes, label_cpc).astype(
                self.dtype
            )[np.newaxis, ...]
        else:
            label_cpc = np.array([self.config[self.task].classes.index(label_cpc)])
        out_tensors = {
            "waveforms": waveforms.astype(self.dtype),
            "cpc": label_cpc.astype(self.dtype),
        }
        return out_tensors

    def extra_repr_keys(self) -> List[str]:
        return [
            "reader",
            "ppm",
        ]
