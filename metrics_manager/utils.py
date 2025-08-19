import csv
import os
from typing import Any, Dict

from qdax.custom_types import Metrics

from set_up_container import EXTRACTOR_LIST


def save_config(save_folder: str, name: str, args: Any):
    """Save the current config in the config.csv file."""

    # Convert arguments to a dictionary
    args_dict = vars(args)
    args_dict["policy_hidden_layer_sizes"] = "_".join(
        map(str, args_dict["policy_hidden_layer_sizes"])
    )
    args_dict["pg_critic_hidden_layer_sizes"] = "_".join(
        map(str, args_dict["pg_critic_hidden_layer_sizes"])
    )
    args_dict["min_bd"] = "_".join(map(str, args_dict["min_bd"]))
    args_dict["max_bd"] = "_".join(map(str, args_dict["max_bd"]))

    # Add the name in first position
    args_dict = {**{"name": name}, **args_dict}

    # Create results folder if needed
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # Opening config file and writing header
    file_name = f"{save_folder}/config.csv"
    if not os.path.exists(file_name):
        with open(file_name, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(args_dict.keys())

    # Writting config
    with open(file_name, mode="a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(args_dict.values())


def args_check(args: Any) -> None:
    if args.reeval_scan_size > 0:
        if args.num_reevals == 0:
            print(
                "!!!WARNING!!! --reeval-scan-size has no impact with --num-reevals 0."
            )
            args.reeval_scan_size = 0
        elif (args.num_reevals * args.num_centroids) % args.reeval_scan_size != 0:
            assert 0, "\n!!!ERROR!!! --num-reevals non divible by --reeval-scan-size."
    assert (
        args.reeval_fitness_extractor in EXTRACTOR_LIST.keys()
    ), "\n !!!ERROR!!! invalid reeval_fitness_extractor."
    assert (
        args.reeval_fitness_reproducibility_extractor in EXTRACTOR_LIST.keys()
    ), "\n !!!ERROR!!! invalid reeval_fitness_reproducibility_extractor."
    assert (
        args.reeval_descriptor_extractor in EXTRACTOR_LIST.keys()
    ), "\n !!!ERROR!!! invalid reeval_descriptor_extractor."
    assert (
        args.reeval_descriptor_reproducibility_extractor in EXTRACTOR_LIST.keys()
    ), "\n !!!ERROR!!! invalid reeval_descriptor_reproducibility_extractor."


def save_metrics(
    file_name: str,
    epoch: float,
    evals: float,
    real_evals: float,
    timesteps: int,
    real_timesteps: int,
    time: float,
    metrics: Metrics,
    reeval_metrics: Metrics,
    fit_reeval_metrics: Metrics,
    desc_reeval_metrics: Metrics,
    fit_var_metrics: Metrics,
    reeval_fit_var_metrics: Metrics,
    desc_var_metrics: Metrics,
    reeval_desc_var_metrics: Metrics,
    additional_metrics: Metrics,
    reeval_additional_metrics: Metrics,
    prefixe: str = "",
) -> None:
    """Save the current metrics in metric file."""

    def name(dic: Dict, name: str) -> Dict:
        return {f"{name}{key}": value for key, value in dic.items()}

    # Set name in all dictionaries
    metrics = name(metrics, prefixe)
    reeval_metrics = name(reeval_metrics, prefixe + "reeval_")
    fit_reeval_metrics = name(fit_reeval_metrics, prefixe + "fit_reeval_")
    desc_reeval_metrics = name(desc_reeval_metrics, prefixe + "desc_reeval_")
    fit_var_metrics = name(fit_var_metrics, prefixe + "fit_var_")
    reeval_fit_var_metrics = name(reeval_fit_var_metrics, prefixe + "reeval_fit_var_")
    desc_var_metrics = name(desc_var_metrics, prefixe + "desc_var_")
    reeval_desc_var_metrics = name(
        reeval_desc_var_metrics, prefixe + "reeval_desc_var_"
    )
    additional_metrics = name(additional_metrics, prefixe + "additional_")
    reeval_additional_metrics = name(
        reeval_additional_metrics, prefixe + "reeval_additional_"
    )

    # Combine all of them
    all_metrics = {
        **metrics,
        **reeval_metrics,
        **fit_reeval_metrics,
        **desc_reeval_metrics,
        **fit_var_metrics,
        **reeval_fit_var_metrics,
        **desc_var_metrics,
        **reeval_desc_var_metrics,
        **additional_metrics,
        **reeval_additional_metrics,
    }

    # Add epoch, eval, timestep and time
    all_metrics = {
        **{
            "epoch": epoch,
            "eval": evals,
            "real_eval": real_evals,
            "timestep": timesteps,
            "real_timestep": real_timesteps,
            "time": time,
        },
        **all_metrics,
    }

    # Opening metric file and writing header
    if not os.path.exists(file_name):
        with open(file_name, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(all_metrics.keys())

    # Writting config
    with open(file_name, mode="a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(all_metrics.values())
