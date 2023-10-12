"""
"""

import posixpath
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from .misc import url_is_reachable

__all__ = [
    "get_ranking",
    "get_score",
    "get_team_digest",
]


_DOMAIN = "https://moody-challenge.physionet.org"
_FOLDER = "2023/results"

_URLS = {
    "summary": posixpath.join(_DOMAIN, _FOLDER, "team_summary_table.tsv"),
}
_URLS.update(
    {
        f"official_scores_{hour}h": posixpath.join(_DOMAIN, _FOLDER, f"official_scores_{hour}.tsv")
        for hour in ["12", "24", "48", "72"]
    }
)
_URLS.update(
    {
        f"unofficial_scores_{hour}h": posixpath.join(_DOMAIN, _FOLDER, f"unofficial_scores_{hour}.tsv")
        for hour in ["12", "24", "48", "72"]
    }
)
_BACKUPS = {name: Path(__file__).resolve().parents[1] / "final_results" / Path(url).name for name, url in _URLS.items()}


def _fetch_final_results(latest: bool = False, key: Optional[str] = None) -> Union[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Fetch the final results from the PhysioNet server or from the local backup.

    Parameters
    ----------
    latest : bool, default False
        Whether to fetch the latest results from the PhysioNet server.
    key : str, optional
        The key of the final results to fetch. If None, all the final results will be fetched.

    Returns
    -------
    dfs : dict of pd.DataFrame or pd.DataFrame
        The final results.

    """
    dfs = {}
    if key is None:
        keys = _URLS.keys()
    else:
        keys = [key]
    if latest and url_is_reachable(_DOMAIN):
        for name in keys:
            url = _URLS[name]
            dfs[name] = pd.read_csv(url, sep="\t", comment="#")
    else:
        for name in keys:
            backup = _BACKUPS[name]
            dfs[name] = pd.read_csv(backup, sep="\t", comment="#")
    if key is None:
        return dfs
    else:
        return dfs[key]


def _get_row(
    team_name: str,
    hour_limit: int = 72,
    metric: Optional[str] = None,
    target: Optional[str] = None,
    evaluated_set: str = "test",
    col: Optional[str] = None,
    latest: bool = False,
) -> Tuple[Optional[pd.Series], Optional[pd.DataFrame], bool]:
    """Get the row of the team in the final results table.

    Parameters
    ----------
    team_name : str
        Name of the team.
    hour_limit : {12, 24, 48, 72}, default 72
        Hour limit of the data used. 72 for challenge main metric.
    metric : {"AUROC", "AUPRC", "F-measure", "Accuracy", "MSE", "MAE"}, optional
        Metric evaluated. None for challenge main metric.
        The challenge main metric is the largest TPR such that FPR <= 0.05.
    target : {"Outcome", "CPC"}, optional
        Target evaluated. None for challenge main target.
    evaluated_set : {"test", "training", "validation"}, default "test"
        The part of the data used for evaluation. Case insensitive.
    col : str, optional
        Column name of the score.
        If not None, `metric` and `target` will be ignored.
    latest : bool, default False
        Whether to fetch the latest results from the PhysioNet server.

    Returns
    -------
    row : pd.Series
        The row of the team in the final results table.
    df : pd.DataFrame
        The final results table.
    is_official : bool
        Whether the final results are officially eligible for rankings and prizes.

    """
    assert hour_limit in [12, 24, 48, 72], f"Invalid hour limit {hour_limit}."

    fetch_key = "summary"
    df = _fetch_final_results(latest, fetch_key)
    if team_name not in df[df["Eligible for rankings and prizes?"].fillna(False)]["Team name"].to_list():
        fetch_key_prefix = "unofficial_scores"
        is_official = False
    else:
        fetch_key_prefix = "official_scores"
        is_official = True
    fetch_key = f"{fetch_key_prefix}_{hour_limit}h"
    df = _fetch_final_results(latest, fetch_key)

    col = _format_col(metric, target, evaluated_set, col)

    if fetch_key_prefix == "official_scores":
        # sort by col
        if metric in ["MAE", "MSE"]:
            # ascending order
            df = df.sort_values(by=col, ascending=True).reset_index(drop=True)
        else:
            # descending order
            df = df.sort_values(by=col, ascending=False).reset_index(drop=True)
        # reset "Rank" column as the index + 1
        # rows with the same score will have the same rank
        df["Rank"] = df.index + 1
        df["Rank"] = df.apply(lambda r: df[df[col] == r[col]]["Rank"].iloc[0], axis=1)
    row = df[df["Team"] == team_name]

    assert not row.empty, f"Team {team_name} not found in the final results table."

    return row, df, is_official


def _format_col(
    metric: Optional[str] = None,
    target: Optional[str] = None,
    evaluated_set: str = "test",
    col: Optional[str] = None,
) -> str:
    """Format column name from given metric, target and evaluated set.

    Parameters
    ----------
    metric : {"AUROC", "AUPRC", "F-measure", "Accuracy", "MSE", "MAE"}, optional
        Metric evaluated. None for challenge main metric.
        The challenge main metric is the largest TPR such that FPR <= 0.05.
    target : {"Outcome", "CPC"}, optional
        Target evaluated. None for challenge main target.
    evaluated_set : {"test", "training", "validation"}, default "test"
        The part of the data used for evaluation. Case insensitive.
    col : str, optional
        Column name of the score.
        If not None, `metric` and `target` will be ignored,
        and will be returned directly.

    Returns
    -------
    col : str
        Column name of the score.

    """
    if col is not None:
        return col

    if target is None:
        target = "Challenge Score"
    assert target in ["Outcome", "CPC", "Challenge Score"]
    evaluated_set = evaluated_set.lower()
    assert evaluated_set in ["test", "training", "validation"]
    if target == "Outcome":
        assert metric in ["AUROC", "AUPRC", "F-measure", "Accuracy"]
    elif target == "CPC":
        assert metric in ["MSE", "MAE"]
    if target == "Challenge Score":
        col = f"{target} on the {evaluated_set} set"
    else:
        col = f"{target} {metric} on the {evaluated_set} set"

    return col


def get_ranking(
    team_name: str,
    hour_limit: int = 72,
    metric: Optional[str] = None,
    target: Optional[str] = None,
    evaluated_set: str = "test",
    col: Optional[str] = None,
    latest: bool = False,
) -> str:
    """Get ranking of a team in the final results table
    with specified metric and target.

    Parameters
    ----------
    team_name : str
        Name of the team.
    hour_limit : {12, 24, 48, 72}, default 72
        Hour limit of the data used. 72 for challenge main metric.
    metric : {"AUROC", "AUPRC", "F-measure", "Accuracy", "MSE", "MAE"}, optional
        Metric evaluated. None for challenge main metric.
        The challenge main metric is the largest TPR such that FPR <= 0.05.
    target : {"Outcome", "CPC"}, optional
        Target evaluated. None for challenge main target.
    evaluated_set : {"test", "training", "validation"}, default "test"
        The part of the data used for evaluation. Case insensitive.
    col : str, optional
        Column name of the score.
        If not None, `metric` and `target` will be ignored.
    latest : bool, default False
        Whether to fetch the latest results from the PhysioNet server.

    Returns
    -------
    ranking : str
        The ranking of the team in the final results table of the fomat "rank / total".
        If the team is not officially eligible for rankings and prizes, "unofficial" will be returned.

    """
    row, df, is_official = _get_row(team_name, hour_limit, metric, target, evaluated_set, col, latest)
    if not is_official:
        ranking = "unofficial"
    else:
        ranking = row["Rank"].values[0]
        ranking = f"{ranking} / {len(df)}"
    return ranking


def get_score(
    team_name: str,
    hour_limit: int = 72,
    metric: Optional[str] = None,
    target: Optional[str] = None,
    evaluated_set: str = "test",
    col: Optional[str] = None,
    latest: bool = False,
) -> str:
    """Get score of a team in the final results table
    with specified metric and target.

    Parameters
    ----------
    team_name : str
        Name of the team.
    hour_limit : {12, 24, 48, 72}, default 72
        Hour limit of the data used. 72 for challenge main metric.
    metric : {"AUROC", "AUPRC", "F-measure", "Accuracy", "MSE", "MAE"}, optional
        Metric evaluated. None for challenge main metric.
        The challenge main metric is the largest TPR such that FPR <= 0.05.
    target : {"Outcome", "CPC"}, optional
        Target evaluated. None for challenge main target.
    evaluated_set : {"test", "training", "validation"}, default "test"
        The part of the data used for evaluation. Case insensitive.
    col : str, optional
        Column name of the score.
        If not None, `metric` and `target` will be ignored.
    latest : bool, default False
        Whether to fetch the latest results from the PhysioNet server.

    Returns
    -------
    score : str
        The score of the team in the final results table in string format.

    """
    row, _, _ = _get_row(team_name, hour_limit, metric, target, evaluated_set, col, latest)
    col = _format_col(metric, target, evaluated_set, col)
    score = row[col].values[0]
    return score


def get_team_digest(
    team_name: str,
    fmt: str = "pd",
    latest: bool = False,
    hour_limits: Optional[List[int]] = None,
    targets: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
) -> Union[str, pd.DataFrame]:
    """Get the digest of a team in the final results table.

    Parameters
    ----------
    team_name : str
        Name of the team.
    fmt : {"pd", "tex", "latex"}, default "pd"
        Format of the digest.
        - "pd": pd.DataFrame
        - "tex" or "latex": LaTeX table
    latest : bool, default False
        Whether to fetch the latest results from the PhysioNet server.
    hour_limits : list of {12, 24, 48, 72}, optional
        Hour limits of the data used.
        If None, all the hour limits will be included.
    targets : list of {"Outcome", "CPC"}, optional
        Targets evaluated. If None, all the targets will be included.
    metrics : list of {"AUROC", "AUPRC", "F-measure", "Accuracy", "MSE", "MAE"}, optional
        Metrics evaluated. If None, all the metrics will be included.

    Returns
    -------
    digest : pd.DataFrame or str
        The digest of the team in the final results table.

    """
    assert fmt.lower() in [
        "pd",
        "tex",
        "latex",
    ], f"`fmt` must be pd, tex or latex, but got {fmt}"
    if hour_limits is None:
        hour_limits = [72, 48, 24, 12]
    evaluated_sets = ["Test", "Validation", "Training"]

    rows = []
    multi_index_arr1 = []
    multi_index_arr2 = []
    multi_index_arr3 = []
    # multi-row 1: challenge main metric
    for es in evaluated_sets:
        new_row = []
        for hour in hour_limits:
            new_row.extend(
                [
                    get_score(team_name, hour, latest=latest),
                    get_ranking(team_name, hour, latest=latest),
                ]
            )
        rows.append(new_row)
    multi_index_arr1 = ["Challenge Score"] * len(evaluated_sets)
    multi_index_arr2 = [""] * len(evaluated_sets)
    multi_index_arr3 = evaluated_sets.copy()

    # multi-row 2: CPC
    if targets is None or "CPC" in targets:
        cpc_metrics = ["MAE", "MSE"]
        if metrics is not None:
            cpc_metrics = [metric for metric in cpc_metrics if metric in metrics]
        for metric in cpc_metrics:
            for es in evaluated_sets:
                new_row = []
                for hour in hour_limits:
                    new_row.extend(
                        [
                            get_score(team_name, hour, metric, "CPC", es, latest=latest),
                            get_ranking(team_name, hour, metric, "CPC", es, latest=latest),
                        ]
                    )
                rows.append(new_row)
                multi_index_arr1.append("CPC")
                multi_index_arr2.append(metric)
                multi_index_arr3.append(es)

    # multi-row 3: Outcome
    if targets is None or "Outcome" in targets:
        outcome_metrics = ["AUROC", "AUPRC", "F-measure", "Accuracy"]
        if metrics is not None:
            outcome_metrics = [metric for metric in outcome_metrics if metric in metrics]
        for metric in outcome_metrics:
            for es in evaluated_sets:
                new_row = []
                for hour in hour_limits:
                    new_row.extend(
                        [
                            get_score(team_name, hour, metric, "Outcome", es, latest=latest),
                            get_ranking(team_name, hour, metric, "Outcome", es, latest=latest),
                        ]
                    )
                rows.append(new_row)
                multi_index_arr1.append("Outcome")
                multi_index_arr2.append(metric)
                multi_index_arr3.append(es)

    df = pd.DataFrame(rows, dtype=str)

    # set multi-column for df
    df.columns = pd.MultiIndex.from_product([[f"{hour}h after ROSC" for hour in hour_limits], ["Score", "Rank"]])

    # set multi-index for df
    df.index = pd.MultiIndex.from_arrays([multi_index_arr1, multi_index_arr2, multi_index_arr3])

    if fmt.lower() == "pd":
        return df
    elif fmt.lower() in ["latex", "tex"]:
        tex = df.to_latex().replace("\\multirow[t]", "\\multirow[c]")
        tex = tex.replace("Validation", "Val.")
        tex = tex.replace("Training", "Train")
        tex = tex.replace("toprule", "hlineB{3.5}")
        tex = tex.replace("bottomrule", "hlineB{3.5}")
        tex = tex.replace("midrule", "hlineB{2.5}")
        tex = tex.replace("\\multicolumn{2}{r}", "\\multicolumn{2}{c}")
        tex = tex.replace(
            "\\multirow[c]{3}{*}{Challenge Score} & \\multirow[c]{3}{*}{}",
            "\\multicolumn{2}{c}{\\multirow[c]{3}{*}{\\textbf{Challenge Score}}}",
        )
        if "72h after ROSC" in tex:
            tex = tex.replace("72h after ROSC", "\\textbf{72h after ROSC}")
        tex = [
            r"% requires packages boldline, multirow",
            r"% put the following in the preamble",
            r"% \usepackage{multirow}",
            r"% \usepackage{boldline}",
            r"% and probably one needs the following to make \textbf work",
            r"% \usepackage[T1]{fontenc}",
            "\\setlength\\tabcolsep{1pt}",
            "\\setlength\\extrarowheight{2pt}",
            "\\begin{tabular}{@{\\extracolsep{4pt}}cclllllllll@{}}",
        ] + tex.splitlines()[1:]
        emphasized_row_idx = [13, 14]
        for row_idx in emphasized_row_idx:
            emphasized_row = tex[row_idx].rstrip(" \\\\").split(" & ")
            for idx in range(1, len(emphasized_row)):
                if len(emphasized_row[idx].strip()) > 0:
                    emphasized_row[idx] = "\\textbf{" + emphasized_row[idx].strip() + "}"
            tex[row_idx] = " & ".join(emphasized_row) + " \\\\"
        tex = "\n".join(tex)
        return tex
    else:
        raise ValueError(f"Invalid format {fmt}.")
