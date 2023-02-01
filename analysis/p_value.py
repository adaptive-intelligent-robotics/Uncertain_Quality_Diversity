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
            p_frame = pd.concat(
                [
                    p_frame,
                    pd.DataFrame.from_dict(
                        {
                            "Reference label": [reference_label],
                            "Label": [compare_label],
                            "p-value": [
                                p_value_ranksum(
                                    frame, reference_label, compare_label, stat
                                )
                            ],
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
    times_data: pd.DataFrame,
    losses_data: pd.DataFrame,
    var_data: pd.DataFrame,
    prefixe: str = "",
    prefixe_title: str = "",
) -> None:
    """Main function to write all main stats p-values in tables."""

    reeval_qd_score = pd.DataFrame()
    loss_qd_score = pd.DataFrame()
    var_desc = pd.DataFrame()
    run_time = pd.DataFrame()

    # Compute p-values for each env, each num_reevals and each compare_size
    for env in times_data["env"].drop_duplicates().values:
        for num_reevals in times_data["num_reevals"].drop_duplicates().values:
            for size in times_data[compare_size].drop_duplicates().values:

                sub_times_data = times_data[
                    (times_data["env"] == env)
                    & (times_data["num_reevals"] == num_reevals)
                    & (times_data[compare_size] == size)
                ]
                sub_losses_data = losses_data[
                    (losses_data["env"] == env)
                    & (losses_data["num_reevals"] == num_reevals)
                    & (losses_data[compare_size] == size)
                ]
                sub_var_data = var_data[
                    (var_data["env"] == env)
                    & (var_data["num_reevals"] == num_reevals)
                    & (var_data[compare_size] == size)
                ]

                # Corrected QD-Score
                file_name = (
                    f"{plot_folder}/{env}_{num_reevals}reevals_"
                    + f"{size}size_{prefixe}_reeval_qdscore_p_value.md"
                )
                new_reeval_qd_score = compute_p_values(
                    sub_times_data, file_name, f"{prefixe}reeval_qd_score"
                )
                new_reeval_qd_score["env"] = env
                new_reeval_qd_score["num_reevals"] = num_reevals
                reeval_qd_score = pd.concat(
                    [reeval_qd_score, new_reeval_qd_score], ignore_index=True
                )

                # Loss QD-Score
                file_name = (
                    f"{plot_folder}/{env}_{num_reevals}reevals_"
                    + f"{size}size_{prefixe}_loss_qdscore_p_value.md"
                )
                new_loss_qd_score = compute_p_values(
                    sub_losses_data, file_name, f"loss_{prefixe}reeval_qd_score"
                )
                new_loss_qd_score["env"] = env
                new_loss_qd_score["num_reevals"] = num_reevals
                loss_qd_score = pd.concat(
                    [loss_qd_score, new_loss_qd_score], ignore_index=True
                )

                # Variance descriptor
                file_name = (
                    f"{plot_folder}/{env}_{num_reevals}reevals_"
                    + f"{size}size_{prefixe}_var_desc_p_value.md"
                )
                new_var_desc = compute_p_values(
                    sub_var_data, file_name, f"{prefixe}avg_fit_var_qd_score"
                )
                new_var_desc["env"] = env
                new_var_desc["num_reevals"] = num_reevals
                var_desc = pd.concat([var_desc, new_var_desc], ignore_index=True)

                # Time to convergence
                file_name = (
                    f"{plot_folder}/{env}_{num_reevals}reevals_"
                    + f"{size}size_{prefixe}_time_p_value.md"
                )
                new_run_time = compute_p_values(
                    sub_times_data, file_name, f"{prefixe}time"
                )
                new_run_time["env"] = env
                new_run_time["num_reevals"] = num_reevals
                run_time = pd.concat([run_time, new_run_time], ignore_index=True)

    # Use concatenanted dataframes to build per-env summary
    for env in times_data["env"].drop_duplicates().values:
        for num_reevals in times_data["num_reevals"].drop_duplicates().values:
            sub_reeval_qd_score = reeval_qd_score[
                (reeval_qd_score["env"] == env)
                & (reeval_qd_score["num_reevals"] == num_reevals)
            ]
            sub_loss_qd_score = loss_qd_score[
                (loss_qd_score["env"] == env)
                & (loss_qd_score["num_reevals"] == num_reevals)
            ]
            sub_var_desc = var_desc[
                (var_desc["env"] == env) & (var_desc["num_reevals"] == num_reevals)
            ]
            sub_run_time = run_time[
                (run_time["env"] == env) & (run_time["num_reevals"] == num_reevals)
            ]

            file_name = (
                f"{plot_folder}/general_{env}_{num_reevals}reevals_"
                + f"{prefixe}_reeval_qdscore_p_value.md"
            )
            general_p_values(sub_reeval_qd_score, file_name)

            file_name = (
                f"{plot_folder}/general_{env}_{num_reevals}reevals_"
                + f"{prefixe}_loss_qdscore_p_value.md"
            )
            general_p_values(sub_loss_qd_score, file_name)

            file_name = (
                f"{plot_folder}/general_{env}_{num_reevals}reevals_"
                + f"{prefixe}_var_desc_p_value.md"
            )
            general_p_values(sub_var_desc, file_name)

            file_name = (
                f"{plot_folder}/general_{env}_{num_reevals}reevals_"
                + f"{prefixe}_time_p_value.md"
            )
            general_p_values(sub_run_time, file_name)

    # Use concatenanted dataframes to build general summary
    for num_reevals in times_data["num_reevals"].drop_duplicates().values:
        sub_reeval_qd_score = reeval_qd_score[
            (reeval_qd_score["num_reevals"] == num_reevals)
        ]
        sub_loss_qd_score = loss_qd_score[(loss_qd_score["num_reevals"] == num_reevals)]
        sub_var_desc = var_desc[(var_desc["num_reevals"] == num_reevals)]
        sub_run_time = run_time[(run_time["num_reevals"] == num_reevals)]

        file_name = (
            f"{plot_folder}/general_{num_reevals}reevals_"
            + f"{prefixe}_reeval_qdscore_p_value.md"
        )
        general_p_values(sub_reeval_qd_score, file_name)

        file_name = (
            f"{plot_folder}/general_{num_reevals}reevals_"
            + f"{prefixe}_loss_qdscore_p_value.md"
        )
        general_p_values(sub_loss_qd_score, file_name)

        file_name = (
            f"{plot_folder}/general_{num_reevals}reevals_"
            + f"{prefixe}_var_desc_p_value.md"
        )
        general_p_values(sub_var_desc, file_name)

        file_name = (
            f"{plot_folder}/general_{num_reevals}reevals_"
            + f"{prefixe}_time_p_value.md"
        )
        general_p_values(sub_run_time, file_name)
