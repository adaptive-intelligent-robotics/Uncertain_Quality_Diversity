from typing import Any

import pandas as pd
from scipy.stats import ranksums


def p_value_ranksum(
    frame: pd.DataFrame, reference_label: str, compare_label: str, stat: str
) -> Any:
    """Compute one p-value for one reference and one compare label for a given stat."""
    _, p = ranksums(
        frame[frame["algo"] == reference_label][stat].to_numpy(),
        frame[frame["algo"] == compare_label][stat].to_numpy(),
    )
    return p


def compute_p_values(
    frame: pd.DataFrame,
    file_name: str,
    stat: str,
) -> pd.DataFrame:
    """Write p-value of stat in a table."""

    p_frame = pd.DataFrame(columns=["Reference label", "Label", "p-value"])
    labels = frame["algo"].drop_duplicates().values

    # For each labels-couple
    for reference_label in labels:
        for compare_label in labels:
            p_value = p_value_ranksum(frame, reference_label, compare_label, stat)
            p_frame = pd.concat(
                [
                    p_frame,
                    pd.DataFrame.from_dict(
                        {
                            "Reference label": [reference_label],
                            "Label": [compare_label],
                            "p-value": [p_value],
                        }
                    ),
                ],
                ignore_index=True,
            )
    written_p_frame = p_frame.pivot(
        index="Reference label", columns="Label", values="p-value"
    )
    p_file = open(file_name, "a")
    p_file.write(written_p_frame.to_markdown())
    p_file.close()

    return p_frame


def general_p_values(
    p_value_frame: pd.DataFrame,
    file_name: str,
) -> None:
    """Write a general p-value in a table."""
    p_frame = pd.DataFrame()
    for reference_label in p_value_frame["Reference label"].drop_duplicates().values:
        for compare_label in p_value_frame["Label"].drop_duplicates().values:
            sub_p_value_frame = p_value_frame[
                (p_value_frame["Reference label"] == reference_label)
                & (p_value_frame["Label"] == compare_label)
            ]
            p_value = max(sub_p_value_frame["p-value"])

            p_frame = pd.concat(
                [
                    p_frame,
                    pd.DataFrame.from_dict(
                        {
                            "Reference label": [reference_label],
                            "Label": [compare_label],
                            "p-value": [p_value],
                        }
                    ),
                ],
                ignore_index=True,
            )
    written_p_frame = p_frame.pivot(
        index="Reference label", columns="Label", values="p-value"
    )
    p_file = open(file_name, "a")
    p_file.write(written_p_frame.to_markdown())
    p_file.close()


def p_values(
    plot_folder: str,
    compare_size: str,
    stat: str,
    dataframe: pd.DataFrame,
    suffixe: str = "",
) -> None:
    """Main function to write all main stats p-values in tables."""

    all_p_value_frame = pd.DataFrame()

    # Compute p-values for each env and each compare_size
    for env in dataframe["env"].drop_duplicates().values:
        for size in dataframe[compare_size].drop_duplicates().values:

            sub_dataframe = dataframe[
                (dataframe["env"] == env) & (dataframe[compare_size] == size)
            ]
            file_name = f"{plot_folder}/{env}_{size}_{stat}{suffixe}.md"
            p_value_frame = compute_p_values(
                sub_dataframe,
                file_name,
                stat,
            )
            p_value_frame["env"] = env
            all_p_value_frame = pd.concat(
                [all_p_value_frame, p_value_frame], ignore_index=True
            )

    # Use concatenanted dataframes to build per-env summary
    for env in all_p_value_frame["env"].drop_duplicates().values:
        p_value_frame = all_p_value_frame[(all_p_value_frame["env"] == env)]

        file_name = f"{plot_folder}/{env}_{stat}{suffixe}.md"
        general_p_values(p_value_frame, file_name)

    # Use concatenanted dataframes to build general summary
    file_name = f"{plot_folder}/general_{stat}{suffixe}.md"
    general_p_values(all_p_value_frame, file_name)
