import os
import traceback
from typing import List

import pandas as pd
from natsort import natsort_keygen

from analysis.constants import BASELINES, CATEGORIES


def get_folder_name(config_frame: pd.DataFrame, name: str, line: int) -> str:
    folder = config_frame["folder"][line]
    folder_name = config_frame[name][line]
    if folder_name.rfind("/") == len(folder_name) - 1:
        folder_name = folder_name[:-1]
    folder_name = folder_name[folder_name.rfind("/") + 1 :]
    folder_name = os.path.join(folder, folder_name)
    return folder_name  # type: ignore


def extract_algo(data: pd.DataFrame, algos: str, columns: List) -> pd.DataFrame:
    sub_data = data[data["algo"].str.contains(algos)].reset_index(drop=True)
    if sub_data.empty:
        return sub_data
    sub_data = sub_data.sort_values(columns, key=natsort_keygen(), ignore_index=True)
    return sub_data


def extract_nonalgo(data: pd.DataFrame, algos: str, columns: List) -> pd.DataFrame:
    sub_data = data[~data["algo"].str.contains(algos)].reset_index(drop=True)
    if sub_data.empty:
        return sub_data
    sub_data = sub_data.sort_values(columns, key=natsort_keygen(), ignore_index=True)
    return sub_data


def sort_data(
    data: pd.DataFrame,
    columns: List = ["env_name", "algo", "line"],
    order: List = BASELINES + CATEGORIES,
) -> pd.DataFrame:
    final_data = extract_algo(data, order[0], columns=columns)
    left_data = extract_nonalgo(data, order[0], columns=columns)
    added_names = order[0]
    for i in range(1, len(order)):
        final_data = pd.concat(
            [
                final_data,
                extract_algo(left_data, order[i], columns=columns),
            ],
            ignore_index=True,
        )
        added_names += "|" + order[i]
        left_data = extract_nonalgo(data, added_names, columns=columns)
    final_data = pd.concat([final_data, left_data], ignore_index=True)
    return final_data


def uniformise_xaxis(data: pd.DataFrame, xaxis: str) -> pd.DataFrame:
    for exp in data["env"].drop_duplicates().values:
        for variant in data["algo"].drop_duplicates().values:
            sub_data = data[(data["env"] == exp) & (data["algo"] == variant)]
            sub_data = sub_data.sort_values(["rep", xaxis], ignore_index=True)
            replications = sub_data["rep"].drop_duplicates().values
            replications_evals = []  # type: List
            evals = []  # type: List
            need_rewrite = False
            for i, repl in enumerate(replications):
                replications_evals.append(
                    sub_data[sub_data["rep"] == repl][xaxis].values
                )
                if len(replications_evals[i]) > len(evals):
                    evals = replications_evals[i]
                elif any(
                    [
                        replications_evals[i][j] != evals[j]
                        for j in range(len(replications_evals[i]))
                    ]
                ):
                    need_rewrite = True
            if not need_rewrite:
                continue
            try:
                assert len(evals) > 0
                for i, repl in enumerate(replications):
                    for j in range(len(replications_evals[i])):
                        data.loc[
                            (data["env"] == exp)
                            & (data["algo"] == variant)
                            & (data["rep"] == repl)
                            & (data[xaxis] == replications_evals[i][j]),
                            xaxis,
                        ] = evals[j]
            except Exception:
                print(
                    "\n!!!WARNING!!! Cannot uniformise",
                    xaxis,
                    "for",
                    variant,
                    "in",
                    exp,
                )
                print(traceback.format_exc(-1))
    return data
