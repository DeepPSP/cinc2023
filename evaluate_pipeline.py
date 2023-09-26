from copy import deepcopy
from pathlib import Path
from typing import Union, Dict, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from cfg import TrainCfg
from dataset import CinC2023Dataset
from models import CRNN_CINC2023
from team_code import run_challenge_models
from helper_code import (
    save_challenge_outputs,
    load_text_file,
    get_hospital,
    get_outcome,
    get_cpc,
    get_outcome_probability,
)
from evaluate_model import (  # noqa: F401
    compute_challenge_score,
    compute_auc,
    compute_accuracy,
    compute_f_measure,
    compute_mse,
    compute_mae,
    compute_one_vs_rest_confusion_matrix,
)  # noqa: F401


@torch.no_grad()
def evaluate_pipeline(
    db_dir: Union[Path, str],
    model_path: Union[Path, str],
    data_part: str = "val",
    patient_ids: Union[str, Sequence[str]] = None,
    allow_failures: bool = False,
    verbose: int = 2,
) -> Dict[str, Union[float, np.ndarray]]:
    """Evaluate a model on the CINC2023 dataset.

    Parameters
    ----------
    db_dir : Union[pathlib.Path, str]
        Path to the root of the CINC2023 dataset.
    model_path : Union[pathlib.Path, str]
        Path to the model to evaluate.
    data_part : {"train", "val"}
        The part of the dataset to evaluate on.
    patient_ids : Union[str, Sequence[str]], default None
        If not None, only evaluate on the specified patients,
        which overrides the `data_part` argument.
    allow_failures : bool, default False
        Whether to allow the model to fail on parts of the data; helpful for debugging.
    verbose : int, default 2
        Verbosity level.

    Returns
    -------
    dict
        Dictionary containing the evaluation metrics:
        - challenge_score: float
        - auroc_outcomes: float
        - auprc_outcomes: float
        - accuracy_outcomes: float
        - f_measure_outcomes: float
        - mse_cpcs: float
        - mae_cpcs: float
        - ovr_cm_cpcs: numpy.ndarray
        - ovr_cm_outcomes: numpy.ndarray

    """
    assert data_part in [
        "train",
        "val",
    ], f"""Invalid data part: {data_part}, must be one of ["train", "val"]"""
    ds = CinC2023Dataset(
        TrainCfg, db_dir=db_dir, training=True if data_part == "train" else False
    )
    input_patient_ids = deepcopy(patient_ids)
    if patient_ids is not None:
        if patient_ids == "all":
            patient_ids = ds.reader._df_records_all_bak.subject.unique().tolist()
        elif isinstance(patient_ids, str):
            patient_ids = [patient_ids]
    else:
        patient_ids = ds.subjects
    num_patients = len(patient_ids)
    data_folder = ds.reader._df_records.path.iloc[0].parents[1]

    model_path = Path(model_path).expanduser().resolve()

    model, train_config = CRNN_CINC2023.from_checkpoint(model_path)
    models = dict(
        main_model=model,
        train_cfg=train_config,
        aux_model=None,
        imputer=None,
        outcome_model=None,
        cpc_model=None,
    )

    output_folder = Path("./eval_res").resolve() / model_path.stem
    output_folder.mkdir(parents=True, exist_ok=True)

    ####################################################################
    # this enclosed block is adapted from run_model.py

    # Iterate over the patients.
    for i in tqdm(
        range(num_patients), total=num_patients, desc="Evaluating", unit="patient"
    ):
        patient_id = patient_ids[i]
        # os.makedirs(os.path.join(output_folder, patient_id), exist_ok=True)
        output_file = output_folder / patient_id / (patient_id + ".txt")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if Path(output_file).exists():
            continue

        # Allow or disallow the model(s) to fail on parts of the data; this can be helpful for debugging.
        try:
            outcome_binary, outcome_probability, cpc = run_challenge_models(
                models, data_folder, patient_id, verbose
            )
        except Exception as e:
            if allow_failures:
                if verbose >= 2:
                    print("... failed.")
                outcome_binary, outcome_probability, cpc = (
                    float("nan"),
                    float("nan"),
                    float("nan"),
                )
            else:
                raise e

        # Save Challenge outputs.
        save_challenge_outputs(
            output_file, patient_id, outcome_binary, outcome_probability, cpc
        )

        del outcome_binary, outcome_probability, cpc

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ####################################################################

    hospitals = list()
    label_outcomes = list()
    label_cpcs = list()

    label_folder = data_folder
    for i in range(num_patients):
        patient_data_file = str(
            Path(label_folder) / patient_ids[i] / (patient_ids[i] + ".txt")
        )
        patient_data = load_text_file(patient_data_file)

        hospital = get_hospital(patient_data)
        label_outcome = get_outcome(patient_data)
        label_cpc = get_cpc(patient_data)

        hospitals.append(hospital)
        label_outcomes.append(label_outcome)
        label_cpcs.append(label_cpc)

    # Load the model outputs.
    output_outcomes = list()
    output_outcome_probabilities = list()
    output_cpcs = list()

    for i in range(num_patients):
        output_file = str(output_folder / patient_ids[i] / (patient_ids[i] + ".txt"))
        output_data = load_text_file(output_file)

        output_outcome = get_outcome(output_data)
        output_outcome_probability = get_outcome_probability(output_data)
        output_cpc = get_cpc(output_data)

        output_outcomes.append(output_outcome)
        output_outcome_probabilities.append(output_outcome_probability)
        output_cpcs.append(output_cpc)

    # Evaluate the models.
    challenge_score = compute_challenge_score(
        label_outcomes, output_outcome_probabilities, hospitals
    )
    auroc_outcomes, auprc_outcomes = compute_auc(
        label_outcomes, output_outcome_probabilities
    )
    accuracy_outcomes, _, _ = compute_accuracy(label_outcomes, output_outcomes)
    f_measure_outcomes, _, _ = compute_f_measure(label_outcomes, output_outcomes)
    mse_cpcs = compute_mse(label_cpcs, output_cpcs)
    mae_cpcs = compute_mae(label_cpcs, output_cpcs)
    # ovr_cm_cpcs = compute_one_vs_rest_confusion_matrix(
    #     label_cpcs,
    #     output_cpcs,
    #     labels=TrainCfg.cpc,
    # )
    # ovr_cm_outcomes = compute_one_vs_rest_confusion_matrix(
    #     label_outcomes,
    #     output_outcomes,
    #     labels=np.unique(np.concatenate((label_outcomes, output_outcomes))),
    # )

    # Construct a string with scores.
    # This string is copied from evaluate_model.py
    output_string = (
        "Challenge Score: {:.3f}\n".format(challenge_score)
        + "Outcome AUROC: {:.3f}\n".format(auroc_outcomes)
        + "Outcome AUPRC: {:.3f}\n".format(auprc_outcomes)
        + "Outcome Accuracy: {:.3f}\n".format(accuracy_outcomes)
        + "Outcome F-measure: {:.3f}\n".format(f_measure_outcomes)
        + "CPC MSE: {:.3f}\n".format(mse_cpcs)
        + "CPC MAE: {:.3f}\n".format(mae_cpcs)
    )
    print(output_string)
    if input_patient_ids is None or input_patient_ids == "all":
        # write to a file if evaluating on specific sets of patients
        if input_patient_ids == "all":
            result_file = output_folder / "all-results.txt"
        else:
            result_file = output_folder / f"{data_part}-results.txt"
        result_file.write_text(output_string)

    return dict(
        challenge_score=challenge_score,
        auroc_outcomes=auroc_outcomes,
        auprc_outcomes=auprc_outcomes,
        accuracy_outcomes=accuracy_outcomes,
        f_measure_outcomes=f_measure_outcomes,
        mse_cpcs=mse_cpcs,
        mae_cpcs=mae_cpcs,
        # ovr_cm_cpcs=ovr_cm_cpcs,
        # ovr_cm_outcomes=ovr_cm_outcomes,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db-dir",
        type=str,
        help="Path to the root of the CINC2023 dataset.",
        dest="db_dir",
        required=True,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to the model to evaluate.",
        dest="model_path",
        required=True,
    )
    parser.add_argument(
        "--data-part",
        type=str,
        help="The part of the dataset to evaluate on.",
        dest="data_part",
        default="val",
    )
    parser.add_argument(
        "--patient-ids",
        type=str,
        help=(
            "comma-separated list of patient IDs to evaluate on. "
            "If not None, only evaluate on the specified patients, "
            "which overrides the `data_part` argument. "
            "If 'all', evaluate on all patients."
        ),
        dest="patient_ids",
        default=None,
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Whether to allow the model to fail on parts of the data; helpful for debugging.",
        dest="allow_failures",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        help="Verbosity level.",
        dest="verbose",
        default=2,
    )
    args = parser.parse_args()

    if Path(args.model_path).is_file():
        model_path = [args.model_path]
    else:
        model_path = list(Path(args.model_path).glob("*.pth.tar"))

    for mp in model_path:
        evaluate_pipeline(
            args.db_dir,
            args.model_path,
            args.data_part,
            args.patient_ids,
            args.allow_failures,
            args.verbose,
        )
