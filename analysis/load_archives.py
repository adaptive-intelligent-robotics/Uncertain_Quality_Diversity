import os
import traceback
from typing import Tuple

import jax.numpy as jnp
import pandas as pd


def get_folder_name(config_frame: pd.DataFrame, name: str, line: int) -> str:
    folder = config_frame["folder"][line]
    folder_name = config_frame[name][line]
    if folder_name.rfind("/") == len(folder_name) - 1:
        folder_name = folder_name[:-1]
    folder_name = folder_name[folder_name.rfind("/") + 1 :]
    folder_name = os.path.join(folder, folder_name)
    return folder_name  # type: ignore


def find_min_max(
    plot_folder: str,
    config_frame: pd.DataFrame,
    prefixe: str = "",
) -> pd.DataFrame:
    """Function to find all the min and max for archives."""

    # If already an min_max_frame file in the folder, use it
    file_name = f"{plot_folder}_csv/{prefixe}min_max_frame.csv"
    if os.path.exists(file_name):
        print(f"Loading existing {prefixe}min_max_frame in", file_name)
        min_max_frame = pd.read_csv(file_name, header=0, index_col=False)
        return min_max_frame

    # If not, create it and populate it
    min_max_frame = pd.DataFrame()

    # Function used later on
    def _new_min_max(
        frame: str, name: str, line: int, min_values: float, max_values: float
    ) -> Tuple:
        folder = get_folder_name(frame, name, line)
        values = jnp.load(os.path.join(folder, "fitnesses.npy"))
        values_inf = jnp.where(values == -jnp.inf, jnp.inf, values)
        min_values = min(min_values, float(jnp.min(values_inf)))
        max_values = max(max_values, float(jnp.max(values)))
        return min_values, max_values

    # For each environment
    for env in config_frame["env"].drop_duplicates().values:

        env_config_frame = config_frame[(config_frame["env"] == env)].reset_index(
            drop=True
        )

        # Initialising all min and max
        min_fitness = jnp.inf
        max_fitness = -jnp.inf
        min_fit_var = jnp.inf
        max_fit_var = -jnp.inf
        min_desc_var = jnp.inf
        max_desc_var = -jnp.inf
        min_additional = jnp.inf
        max_additional = -jnp.inf

        # For each run of this environment
        for line in range(env_config_frame.shape[0]):

            # Update min_fitness and max_fitness
            try:
                min_fitness, max_fitness = _new_min_max(
                    env_config_frame,
                    "repertoire_folder",
                    line,
                    min_fitness,
                    max_fitness,
                )
            except Exception:
                print("\n!!!WARNING!!! Cannot open repertoire for:")
                print(f"{env_config_frame.loc[line]}")
                traceback.print_exc()

            # Update min_fitness and max_fitness
            try:
                min_fitness, max_fitness = _new_min_max(
                    env_config_frame,
                    f"{prefixe}reeval_repertoire_folder",
                    line,
                    min_fitness,
                    max_fitness,
                )
            except Exception:
                print(f"\n!!!WARNING!!! Cannot open {prefixe}reeval_repertoire for:")
                print(f"{env_config_frame.loc[line]}")
                traceback.print_exc()

            # Update min_fit_var and max_fit_var
            try:
                min_fit_var, max_fit_var = _new_min_max(
                    env_config_frame,
                    f"{prefixe}fit_var_repertoire_folder",
                    line,
                    min_fit_var,
                    max_fit_var,
                )
            except Exception:
                print(f"\n!!!WARNING!!! Cannot open {prefixe}fit_var_repertoire for:")
                print(f"{env_config_frame.loc[line]}")
                traceback.print_exc()

            # Update min_desc_var and max_desc_var
            try:
                min_desc_var, max_desc_var = _new_min_max(
                    env_config_frame,
                    f"{prefixe}desc_var_repertoire_folder",
                    line,
                    min_desc_var,
                    max_desc_var,
                )
            except Exception:
                print(f"\n!!!WARNING!!! Cannot open {prefixe}desc_var_repertoire for:")
                print(f"{env_config_frame.loc[line]}")
                traceback.print_exc()

            # Update min_additional and max_additional
            try:
                min_additional, max_additional = _new_min_max(
                    env_config_frame,
                    "additional_folder",
                    line,
                    min_additional,
                    max_additional,
                )
            except Exception:
                print("\n!!!WARNING!!! Cannot open additional repertoire for:")
                print(f"{env_config_frame.loc[line]}")
                traceback.print_exc()

        # Update the frame for this env
        min_max_frame = pd.concat(
            [
                min_max_frame,
                pd.DataFrame.from_dict(
                    {
                        "env": [env],
                        "min_fitness": min_fitness,
                        "max_fitness": max_fitness,
                        "min_fit_var": min_fit_var,
                        "max_fit_var": max_fit_var,
                        "min_desc_var": min_desc_var,
                        "max_desc_var": max_desc_var,
                        "min_additional": min_additional,
                        "max_additional": max_additional,
                    }
                ),
            ],
            ignore_index=True,
        )

    # Write in file
    min_max_frame.to_csv(file_name, index=None)

    # Return
    return min_max_frame
