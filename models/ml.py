"""
Currently NOT used, NOT tested.
"""

import json
import pickle
import multiprocessing as mp
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, List, Union, Tuple, Sequence

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import ParameterGrid, GridSearchCV

# from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
)
from xgboost import XGBClassifier
from torch_ecg.cfg import CFG, DEFAULTS
from torch_ecg.components.outputs import ClassificationOutput
from torch_ecg.components.loggers import LoggerManager
from torch_ecg.utils.utils_data import stratified_train_test_split  # noqa: F401
from torch_ecg.utils.utils_metrics import _cls_to_bin
from tqdm.auto import tqdm

from utils.scoring_metrics import compute_challenge_metrics
from utils.features import get_features, get_labels
from cfg import BaseCfg
from data_reader import CINC2023Reader
from outputs import CINC2023Outputs
from helper_code import get_hospital


__all__ = [
    "ML_Classifier_CINC2023",
]


class ML_Classifier_CINC2023(object):
    """Classifier for CINC2023 using sklearn and/or xgboost.

    Parameters:
    -----------
    config : CFG, optional
        configurations

    """

    __name__ = "ML_Classifier_CINC2023"

    def __init__(
        self,
        config: Optional[CFG] = None,
        **kwargs: Any,
    ) -> None:
        self.config = deepcopy(config)
        assert self.config.get("output_target", None) in [
            "cpc",
            "outcome",
        ], "`output_target` is not set or not supported in `config`."
        self.__imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        self.__scaler = StandardScaler()

        self.logger_manager = None
        self.reader = None
        self.__df_features = None
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None
        self.train_hospitals, self.test_hospitals = None, None
        self._prepare_training_data()

        self.__cache = {}
        self.best_clf, self.best_params, self.best_score = None, None, None
        self._no = 1

        self._num_workers = max(1, mp.cpu_count() - 2)

    @property
    def y_col(self) -> str:
        return self.config.output_target

    @property
    def feature_list(self) -> List[str]:
        return self.config.feature_list

    def _prepare_training_data(self, db_dir: Optional[Union[str, Path]] = None) -> None:
        """Prepares training data.

        Parameters
        ----------
        db_dir : str, optional
            database directory; if is None, do nothing.

        """
        if db_dir is not None:
            self.config.db_dir = db_dir
        self.config.db_dir = self.config.get("db_dir", None)
        if self.config.db_dir is None:
            return

        if self.config.cont_scaler.lower() == "minmax":
            self.__scaler = MinMaxScaler()
        elif self.config.cont_scaler.lower() == "standard":
            self.__scaler = StandardScaler()
        else:
            raise ValueError(f"Scaler: {self.config.cont_scaler} not supported.")
        if self.logger_manager is None:
            logger_config = dict(
                log_dir=self.config.get("log_dir", None),
                log_suffix="ML-GridSearch",
                tensorboardx_logger=False,
            )
            self.logger_manager = LoggerManager.from_config(logger_config)

        self.config.db_dir = Path(self.config.db_dir).resolve().absolute()
        self.reader = CINC2023Reader(self.config.db_dir)

        self.__df_features = pd.DataFrame()
        for subject in self.reader.all_subjects:
            metadata_string = self.reader.get_absolute_path(
                subject, extension=self.reader.ann_ext
            ).read_text()
            patient_features = get_features(metadata_string, ret_type="dict")
            patient_features.update(get_labels(metadata_string, ret_type="dict"))
            patient_features["hospital"] = get_hospital(metadata_string)
            self.__df_features = pd.concat(
                [self.__df_features, patient_features],
                axis=0,
                ignore_index=True,
            )
        self.__df_features.loc[:, "subject"] = self.reader.all_subjects
        self.__df_features = self.__df_features.set_index("subject")

        # apply imputer and scaler
        self.__df_features.loc[:, self.feature_list] = self.__imputer.fit_transform(
            self.__df_features.loc[:, self.feature_list]
        )
        self.__df_features.loc[
            :, self.config.cont_features
        ] = self.__scaler.fit_transform(
            self.__df_features.loc[:, self.config.cont_features]
        )

        train_set, test_set = self._train_test_split()
        df_train = self.__df_features.loc[train_set]
        df_test = self.__df_features.loc[test_set]
        self.X_train = df_train[self.feature_list].values
        self.y_train = df_train[self.y_col].values
        self.X_test = df_test[self.feature_list].values
        self.y_test = df_test[self.y_col].values
        self.train_hospitals = df_train["hospital"].values
        self.test_hospitals = df_test["hospital"].values

    def get_model(
        self, model_name: str, params: Optional[dict] = None
    ) -> BaseEstimator:
        """Returns a model instance.

        Parameters
        ----------
        model_name : str
            model name, ref. `self.model_map`
        params : dict, optional
            model parameters

        Returns
        -------
        BaseEstimator
            model instance

        """
        model_cls = self.model_map[model_name]
        if model_cls in [GradientBoostingClassifier, SVC]:
            params.pop("n_jobs", None)
        return model_cls(**(params or {}))

    def save_model(
        self,
        model: BaseEstimator,
        imputer: SimpleImputer,
        scaler: BaseEstimator,
        config: CFG,
        model_path: Union[str, Path],
    ) -> None:
        """Saves a model to a file.

        Parameters
        ----------
        model : BaseEstimator
            model instance to save
        imputer : SimpleImputer
            imputer instance to save
        scaler : BaseEstimator
            scaler instance to save
        config : CFG
            configurations of the model
        model_path : str or pathlib.Path
            path to save the model

        """
        _config = deepcopy(config)
        _config.pop("db_dir", None)
        Path(model_path).write_bytes(
            pickle.dumps(
                {
                    "config": _config,
                    "imputer": imputer,
                    "scaler": scaler,
                    "classifier": model,
                }
            )
        )

    def save_best_model(self, model_name: Optional[str] = None) -> None:
        """Saves the best model to a file.

        Parameters
        ----------
        model_name : str, optional
            model name,
            defaults to f"{self.best_clf.__class__.__name__}_{self.best_score}.pkl"

        """
        if model_name is None:
            model_name = f"{self.best_clf.__class__.__name__}_{self.best_score}.pkl"
        self.save_model(
            self.best_clf,
            self.__imputer,
            self.__scaler,
            self.config,
            Path(self.config.get("model_dir", ".")) / model_name,
        )

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "ML_Classifier_CINC2023":
        """Loads a ML_Classifier_CINC2023 instance from a file.

        Parameters
        ----------
        path : str or pathlib.Path
            path to the model file

        Returns
        -------
        ML_Classifier_CINC2023
            ML_Classifier_CINC2023 instance

        """
        loaded = pickle.loads(Path(path).read_bytes())
        config = loaded["config"]
        clf = cls(config)
        clf.__imputer = loaded["imputer"]
        clf.__scaler = loaded["scaler"]
        clf.best_clf = loaded["classifier"]
        return clf

    def inference(self, patient_metadata: str) -> ClassificationOutput:
        """Helper function to infer the cpc and/or outcome of a patient.

        Parameters
        ----------
        patient_data : str
            patient metadata, read from a (.txt) file

        Returns
        -------
        ClassificationOutput
            classification output, with the following items:
            - classes: list of str,
                list of class names
            - prob: ndarray,
                probability array of each class, of shape (1, n_classes)
            - pred: ndarray,
                predicted class index, of shape (1,)
            - bin_pred: ndarray,
                binarized prediction (one-hot), of shape (1, n_classes)

        """
        raise NotImplementedError

    def search(
        self,
        model_name: str = "rf",
        cv: Optional[int] = None,
        experiment_tag: Optional[str] = None,
    ) -> Tuple[BaseEstimator, dict, float]:
        """Performs a grid search on the model.

        Parameters
        ----------
        model_name : str
            model name, ref. to self.config.model_map
        cv : int, optional
            number of cross-validation folds,
            None for no cross-validation
        experiment_tag : str, optional
            tag for the experiment,
            used to create key for the experiment to save in cache

        Returns
        -------
        BaseEstimator
            the best model instance
        dict
            the best model parameters
        float
            the best model score

        """
        assert self.reader is not None, "No training data found."
        cache_key = self._get_cache_key(model_name, cv, experiment_tag)

        if cv is None:
            msg = "Performing grid search with no cross validation."
            self.logger_manager.log_message(msg)
            (
                self.best_clf,
                self.best_params,
                self.best_score,
            ) = self._perform_grid_search_no_cv(
                model_name,
                self.config.grids[model_name],
                self.X_train,
                self.y_train,
                self.X_test,
                self.y_test,
                self.train_hospitals,
                self.test_hospitals,
            )

            # save in self.__cache
            self.__cache[cache_key] = dict(
                best_clf=deepcopy(self.best_clf),
                best_params=deepcopy(self.best_params),
                best_score=self.best_score,
            )

            self._no += 1

            return self.best_clf, self.best_params, self.best_score
        else:
            msg = f"Performing grid search with {cv}-fold cross validation."
            self.logger_manager.log_message(msg)
            (
                self.best_clf,
                self.best_params,
                self.best_score,
            ) = self._perform_grid_search_cv(
                model_name,
                self.config.grids[model_name],
                self.X_train,
                self.y_train,
                self.X_test,
                self.y_test,
                self.train_hospitals,
                self.test_hospitals,
                cv,
            )

            # save in self.__cache
            self.__cache[cache_key] = dict(
                best_clf=deepcopy(self.best_clf),
                best_params=deepcopy(self.best_params),
                best_score=self.best_score,
            )

            self._no += 1

            return self.best_clf, self.best_params, self.best_score

    def _perform_grid_search_no_cv(
        self,
        model_name: str,
        param_grid: ParameterGrid,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        train_hospitals: Sequence[str],
        val_hospitals: Sequence[str],
    ) -> Tuple[BaseEstimator, dict, float]:
        """Performs a grid search on the given model
        and parameters without cross validation.

        Parameters
        ----------
        model_name : str
            Model name, ref. to ``self.config.model_map``.
        param_grid : ParameterGrid
            Parameter grid for grid search.
        X_train : np.ndarray
            Training features, of shape ``(n_samples, n_features)``.
        y_train : np.ndarray
            Training labels, of shape ``(n_samples,)``.
        X_val : np.ndarray
            Validation features, of shape ``(n_samples, n_features)``.
        y_val : np.ndarray
            Validation labels, of shape ``(n_samples,)``.
        train_hospitals : Sequence[str]
            List of hospitals of the samples in ``X_train``.
        val_hospitals : Sequence[str]
            List of hospitals of the samples in ``X_val``.

        Returns
        -------
        BaseEstimator
            The best model instance
        dict
            The best model parameters
        float
            The best model score

        """
        best_score = np.inf
        best_clf = None
        best_params = None
        with tqdm(enumerate(param_grid)) as pbar:
            for idx, params in pbar:
                updated_params = deepcopy(params)
                updated_params["n_jobs"] = self._num_workers
                try:
                    clf_gs = self.get_model(model_name, params)
                    clf_gs.fit(X_train, y_train)
                except Exception:
                    continue

                y_prob = clf_gs.predict_proba(X_val)
                y_pred = clf_gs.predict(X_val)
                bin_pred = _cls_to_bin(
                    y_pred, shape=(y_pred.shape[0], len(self.config.classes))
                )
                outputs = CINC2023Outputs(
                    outcome_output=ClassificationOutput(
                        classes=self.config.classes,
                        prob=y_prob,
                        pred=y_pred,
                        bin_pred=bin_pred,
                    ),
                    murmur_output=None,
                    segmentation_output=None,
                )

                val_metrics = compute_challenge_metrics(
                    labels=[{"outcome": y_val}],
                    outputs=[outputs],
                    hospitals=[val_hospitals],
                )

                msg = (
                    f"""Model - {self.model_map[model_name].__name__}\nParameters:\n"""
                )
                for k, v in params.items():
                    msg += f"""{k} = {v}\n"""
                self.logger_manager.log_message(msg)

                self.logger_manager.log_metrics(
                    metrics=val_metrics,
                    step=idx,
                    epoch=self._no,
                    part="val",
                )

                if val_metrics[self.config.monitor] < best_score:
                    best_score = val_metrics[self.config.monitor]
                    best_clf = clf_gs
                    best_params = params

        return best_clf, best_params, best_score

    def _perform_grid_search_cv(
        self,
        model_name: str,
        param_grid: ParameterGrid,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        train_hospitals: Sequence[str],
        val_hospitals: Sequence[str],
        cv: int = 5,
    ) -> Tuple[BaseEstimator, dict, float]:
        """Performs a grid search on the given model
        and parameters with cross validation.

        Parameters
        ----------
        model_name : str
            Model name, ref. to ``self.config.model_map``.
        param_grid : ParameterGrid
            Parameter grid for grid search.
        X_train : np.ndarray
            Training features, of shape ``(n_samples, n_features)``.
        y_train : np.ndarray
            Training labels, of shape ``(n_samples,)``.
        X_val : np.ndarray
            Validation features, of shape ``(n_samples, n_features)``.
        y_val : np.ndarray
            Validation labels, of shape ``(n_samples,)``.
        train_hospitals : Sequence[str]
            List of hospitals of the samples in ``X_train``.
        val_hospitals : Sequence[str]
            List of hospitals of the samples in ``X_val``.
        cv : int, default 5
            Number of cross validation folds.

        Returns
        -------
        BaseEstimator
            The best model instance.
        dict
            The best model parameters.
        float
            The best model score.

        """
        gscv = GridSearchCV(
            estimator=self.get_model(model_name),
            param_grid=param_grid.param_grid,
            cv=cv,
            n_jobs=self._num_workers,
            verbose=1,
        )
        gscv.fit(X_train, y_train)
        best_clf = gscv.best_estimator_
        best_params = gscv.best_params_
        # best_score = gscv.best_score_
        y_prob = best_clf.predict_proba(X_val)
        y_pred = best_clf.predict(X_val)
        bin_pred = _cls_to_bin(
            y_pred, shape=(y_pred.shape[0], len(self.config.classes))
        )
        outputs = CINC2023Outputs(
            cpc_output=ClassificationOutput(
                classes=self.config.classes,
                prob=y_prob,
                pred=y_pred,
                bin_pred=bin_pred,
            ),
        )

        val_metrics = compute_challenge_metrics(
            labels=[{"outcome": y_val}],
            outputs=[outputs],
            hospitals=[val_hospitals],
        )
        best_score = val_metrics[self.config.monitor]

        msg = f"""Model - {self.model_map[model_name].__name__}\nParameters:\n"""
        for k, v in best_params.items():
            msg += f"""{k} = {v}\n"""
        self.logger_manager.log_message(msg)

        self.logger_manager.log_metrics(
            metrics=val_metrics,
            step=self._no,
            epoch=self._no,
            part="val",
        )

        return best_clf, best_params, best_score

    def get_cache(
        self,
        model_name: str = "rf",
        cv: Optional[int] = None,
        name: Optional[str] = None,
    ) -> dict:
        """Gets the cache for historical grid searches.

        Parameters
        ----------
        model_name : str
            model name, ref. to self.config.model_map
        cv : int, default None
            number of cross validation folds
            None for no cross validation
        name : str, default None
            suffix name of the cache

        Returns
        -------
        dict
            the cached grid search results

        """
        key = self._get_cache_key(model_name, cv, name)
        return self.__cache[key]

    def _get_cache_key(
        self,
        model_name: str = "rf",
        cv: Optional[int] = None,
        name: Optional[str] = None,
    ) -> str:
        """Gets the cache key for historical grid searches.

        Parameters
        ----------
        model_name : str
            model name, ref. to self.config.model_map
        cv : int, default None
            number of cross validation folds
            None for no cross validation
        name : str, default None
            suffix name of the cache

        Returns
        -------
        str
            the cache key

        """
        key = model_name
        if cv is not None:
            key += f"_{cv}"
        if name is None:
            name = f"ex{self._no}"
        key += f"_{name}"
        return key

    def list_cache(self) -> List[str]:
        return list(self.__cache)

    @property
    def df_features(self) -> pd.DataFrame:
        return self.__df_features

    @property
    def imputer(self) -> SimpleImputer:
        return self.__imputer

    @property
    def scaler(self) -> BaseEstimator:
        return self.__scaler

    @property
    def model_map(self) -> Dict[str, BaseEstimator]:
        """Returns a map of model name to model class."""
        return {
            "svm": SVC,
            "svc": SVC,
            "random_forest": RandomForestClassifier,
            "rf": RandomForestClassifier,
            "gradient_boosting": GradientBoostingClassifier,
            "gdbt": GradientBoostingClassifier,
            "gb": GradientBoostingClassifier,
            "bagging": BaggingClassifier,
            "xgboost": XGBClassifier,
            "xgb": XGBClassifier,
        }

    def _train_test_split(
        self, train_ratio: float = 0.8, force_recompute: bool = False
    ) -> Tuple[List[str], List[str]]:
        """Stratified train/test split.

        Parameters
        ----------
        train_ratio : float, default 0.8
            ratio of training data to total data
        force_recompute : bool, default False
            if True, recompute the train/test split

        Returns
        -------
        train_set : List[str]
            list of training record names
        test_set: List[str]
            list of testing record names

        """
        _train_ratio = int(train_ratio * 100)
        _test_ratio = 100 - _train_ratio
        assert _train_ratio * _test_ratio > 0

        # let the data reader (re-)load the metadata dataframes
        # in which case would be read from the disk via `pd.read_csv`
        # and the string values parsed from the txt files
        # are automatically converted to the correct data types
        # e.g. "50" -> 50 or 50.0 depending on whether the column has nan values
        # and "True" -> True or "False" -> False, "nan" -> np.nan, etc.
        self.reader._ls_rec()

        train_file = self.reader.db_dir / f"train_ratio_{_train_ratio}.json"
        test_file = self.reader.db_dir / f"test_ratio_{_test_ratio}.json"
        (BaseCfg.project_dir / "utils").mkdir(exist_ok=True)
        aux_train_file = (
            BaseCfg.project_dir / "utils" / f"train_ratio_{_train_ratio}.json"
        )
        aux_test_file = BaseCfg.project_dir / "utils" / f"test_ratio_{_test_ratio}.json"

        if not force_recompute and train_file.exists() and test_file.exists():
            return json.loads(train_file.read_text()), json.loads(test_file.read_text())

        df = self.reader._df_subjects.copy()
        df.loc[:, "Age"] = (
            df["Age"].fillna(df["Age"].mean()).astype(int)
        )  # only one nan
        # to age group
        df.loc[:, "Age"] = df["Age"].apply(lambda x: str(20 * (x // 20)))
        for col in ["OHCA", "Shockable Rhythm"]:
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
                "Shockable Rhythm",
                "CPC",
            ],
            test_ratio=1 - train_ratio,
            reset_index=False,
        )

        train_set = df_train.index.tolist()
        test_set = df_test.index.tolist()

        if force_recompute or not train_file.exists():
            train_file.write_text(json.dumps(train_set, ensure_ascii=False))
            test_file.write_text(json.dumps(test_set, ensure_ascii=False))

        if force_recompute or not aux_train_file.exists():
            aux_train_file.write_text(json.dumps(train_set, ensure_ascii=False))
            aux_test_file.write_text(json.dumps(test_set, ensure_ascii=False))

        DEFAULTS.RNG.shuffle(train_set)
        DEFAULTS.RNG.shuffle(test_set)

        return train_set, test_set
