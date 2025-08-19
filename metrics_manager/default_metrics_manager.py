import os
import time
from functools import partial
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from qdax.core.containers.repertoire import Repertoire
from qdax.custom_types import Metrics, RNGKey

from metrics_manager.utils import args_check, save_config, save_metrics
from set_up_container import EXTRACTOR_LIST


class MetricsManager:
    """
    Class that contains all metrics computation and writting.
    """

    def __init__(
        self,
        args: Any,
        name: str,
        scoring_fn: Callable,
        reevaluation_fn: Callable,
        reevaluation_in_cell_fn: Callable,
        metric_repertoire: Repertoire,
    ) -> None:
        """Set up all the necesary attributes and create all
        necessary folders and initialise all functions."""

        self._args = args
        self._name = name
        self._scoring_fn = scoring_fn
        self._reevaluation_fn = reevaluation_fn
        self._reevaluation_in_cell_fn = reevaluation_in_cell_fn
        self._metric_repertoire = metric_repertoire

        # Input check
        args_check(self._args)

        # Create results folder
        if not os.path.exists(self._args.results):
            os.mkdir(self._args.results)

        # Create all the folders to save repertoires
        self._create_all_repertoire_folders()

        # Create the metrics files
        self._create_metrics_files()

        # Prepare the reeval metric functions
        self._reevaluation_metrics_function = (
            self._default_reevaluation_metrics_function
        )

        # Write the corresponding config
        save_config(save_folder=self._args.results, name=self._name, args=self._args)

    def write_all_metrics(
        self,
        epoch: int,
        evals: float,
        real_evals: float,
        timesteps: int,
        real_timesteps: int,
        current_time: float,
        repertoire: Repertoire,
        projected_repertoire: Repertoire,
        emitter_state: Any,
        random_key: RNGKey,
        final: bool = False,
    ) -> Tuple[float, float, float]:
        """Main metric function. Compute and write all the metrics and
        all the repertoires.
        """
        metrics_t = 0.0
        reeval_t = 0.0
        write_t = 0.0

        # First, compute metrics on the projected repertoire
        start_t = time.time()
        metrics = self._metrics_function(projected_repertoire)
        metrics_t += time.time() - start_t

        # Second, compute reeval metrics on the projected repertoire
        start_t = time.time()
        (
            reeval_metrics,
            reeval_repertoire,
            fit_reeval_metrics,
            fit_reeval_repertoire,
            desc_reeval_metrics,
            desc_reeval_repertoire,
            fit_var_metrics,
            fit_var_repertoire,
            reeval_fit_var_metrics,
            reeval_fit_var_repertoire,
            desc_var_metrics,
            desc_var_repertoire,
            reeval_desc_var_metrics,
            reeval_desc_var_repertoire,
            additional_metrics,
            additional_repertoire,
            reeval_additional_metrics,
            reeval_additional_repertoire,
            random_key,
        ) = self._reevaluation_metrics_function(projected_repertoire, random_key)
        jax.tree_util.tree_map(
            lambda x: x.block_until_ready(), desc_var_metrics
        )  # ensure timing accuracy
        reeval_t += time.time() - start_t

        # Sanity check of min_fitness
        start_t = time.time()
        if self._args.container != "MOME-Reprod":
            fitnesses = repertoire.fitnesses
            fitnesses = jnp.where(fitnesses == -jnp.inf, jnp.inf, fitnesses)
            if jnp.min(fitnesses) < -self._args.qd_offset:
                print(
                    f"!!!WARNING!!! got min fit {jnp.min(fitnesses)} < {-self._args.qd_offset},"
                    "may lead to inacurate QD-Score."
                )

        reeval_fitnesses = reeval_repertoire.fitnesses
        reeval_fitnesses = jnp.where(
            reeval_fitnesses == -jnp.inf, jnp.inf, reeval_fitnesses
        )
        if jnp.min(reeval_fitnesses) < -self._args.qd_offset:
            print(
                f"!!!WARNING!!! got min fit {jnp.min(reeval_fitnesses)} < {-self._args.qd_offset},"
                "may lead to inacurate QD-Score."
            )
        metrics_t += time.time() - start_t

        # Write metrics
        start_t = time.time()
        save_metrics(
            file_name=self._args.metrics_file,
            epoch=epoch,
            evals=evals,
            real_evals=real_evals,
            timesteps=timesteps,
            real_timesteps=real_timesteps,
            time=current_time,
            metrics=metrics,
            reeval_metrics=reeval_metrics,
            fit_reeval_metrics=fit_reeval_metrics,
            desc_reeval_metrics=desc_reeval_metrics,
            fit_var_metrics=fit_var_metrics,
            reeval_fit_var_metrics=reeval_fit_var_metrics,
            desc_var_metrics=desc_var_metrics,
            reeval_desc_var_metrics=reeval_desc_var_metrics,
            additional_metrics=additional_metrics,
            reeval_additional_metrics=reeval_additional_metrics,
        )
        save_metrics(
            file_name=self._args.in_cell_metrics_file,
            epoch=epoch,
            evals=evals,
            real_evals=real_evals,
            timesteps=timesteps,
            real_timesteps=real_timesteps,
            time=current_time,
            metrics=metrics,
            reeval_metrics=reeval_metrics,
            fit_reeval_metrics=fit_reeval_metrics,
            desc_reeval_metrics=desc_reeval_metrics,
            fit_var_metrics=fit_var_metrics,
            reeval_fit_var_metrics=reeval_fit_var_metrics,
            desc_var_metrics=desc_var_metrics,
            reeval_desc_var_metrics=reeval_desc_var_metrics,
            additional_metrics=additional_metrics,
            reeval_additional_metrics=reeval_additional_metrics,
        )
        print("    -> Metrics saved in", self._args.metrics_file)
        print("    -> In cell metrics saved in", self._args.in_cell_metrics_file)

        # Write repertoire
        if epoch % self._args.archive_log_period == 0 or final:
            repertoire.save(path=self._args.results_repertoire)
            projected_repertoire.save(path=self._args.results_projected_repertoire)
            reeval_repertoire.save(path=self._args.results_reeval_repertoire)
            fit_reeval_repertoire.save(path=self._args.results_fit_reeval_repertoire)
            desc_reeval_repertoire.save(path=self._args.results_desc_reeval_repertoire)
            fit_var_repertoire.save(path=self._args.results_fit_var_repertoire)
            reeval_fit_var_repertoire.save(
                path=self._args.results_reeval_fit_var_repertoire
            )
            desc_var_repertoire.save(path=self._args.results_desc_var_repertoire)
            reeval_desc_var_repertoire.save(
                path=self._args.results_reeval_desc_var_repertoire
            )
            additional_repertoire.save(path=self._args.results_additional_repertoire)
            reeval_additional_repertoire.save(
                path=self._args.results_reeval_additional_repertoire
            )
            reeval_repertoire.save(path=self._args.results_in_cell_reeval_repertoire)
            fit_reeval_repertoire.save(
                path=self._args.results_in_cell_fit_reeval_repertoire
            )
            desc_reeval_repertoire.save(
                path=self._args.results_in_cell_desc_reeval_repertoire
            )
            fit_var_repertoire.save(path=self._args.results_in_cell_fit_var_repertoire)
            reeval_fit_var_repertoire.save(
                path=self._args.results_in_cell_reeval_fit_var_repertoire
            )
            desc_var_repertoire.save(
                path=self._args.results_in_cell_desc_var_repertoire
            )
            reeval_desc_var_repertoire.save(
                path=self._args.results_in_cell_reeval_desc_var_repertoire
            )
            print(
                "    -> All repertoire saved, original in",
                self._args.results_repertoire,
            )

        write_t += time.time() - start_t

        # Return all the timings
        return metrics_t, reeval_t, write_t

    @partial(jax.jit, static_argnames=("self",))
    def _metrics_function(self, repertoire: Repertoire) -> Metrics:
        repertoire_empty = repertoire.fitnesses == -jnp.inf
        qd_score = jnp.sum(repertoire.fitnesses, where=~repertoire_empty)
        qd_score += self._args.qd_offset * jnp.sum(1.0 - repertoire_empty)
        coverage = 100 * jnp.mean(1.0 - repertoire_empty)
        max_fitness = jnp.max(repertoire.fitnesses)
        min_fitness = jnp.min(
            jnp.where(repertoire.fitnesses == -jnp.inf, jnp.inf, repertoire.fitnesses)
        )
        return {
            "qd_score": qd_score,
            "max_fitness": max_fitness,
            "min_fitness": min_fitness,
            "coverage": coverage,
        }

    @partial(jax.jit, static_argnames=("self",))
    def _default_reevaluation_metrics_function(
        self,
        repertoire: Repertoire,
        random_key: RNGKey,
    ) -> Tuple:

        # Perform reevaluation
        (
            reeval_repertoire,
            fit_reeval_repertoire,
            desc_reeval_repertoire,
            fit_var_repertoire,
            reeval_fit_var_repertoire,
            desc_var_repertoire,
            reeval_desc_var_repertoire,
            random_key,
        ) = self._reevaluation_fn(
            repertoire=repertoire,
            random_key=random_key,
            metric_repertoire=self._metric_repertoire,
            scoring_fn=self._scoring_fn,
            num_reevals=self._args.num_reevals,
            scan_size=self._args.reeval_scan_size,
            fitness_extractor=EXTRACTOR_LIST[self._args.reeval_fitness_extractor],
            fitness_reproducibility_extractor=EXTRACTOR_LIST[
                self._args.reeval_fitness_reproducibility_extractor
            ],
            descriptor_extractor=EXTRACTOR_LIST[self._args.reeval_descriptor_extractor],
            descriptor_reproducibility_extractor=EXTRACTOR_LIST[
                self._args.reeval_descriptor_reproducibility_extractor
            ],
        )

        # If arm_gaussian_desc_bi_variance task, additional is task metrics
        if "arm_gaussian_desc_bi_variance" in self._args.env_name:

            # Compute the type of std for each genotype
            std_type = jax.vmap(lambda x: jnp.where(jnp.prod(x - 0.5) >= 0, 0.0, 1.0))(
                repertoire.genotypes
            )
            reeval_std_type = jax.vmap(
                lambda x: jnp.where(jnp.prod(x - 0.5) >= 0, 0.0, 1.0)
            )(reeval_repertoire.genotypes)

            # Set dead individuals to -jnp.inf
            std_type = jnp.where(repertoire.fitnesses > -jnp.inf, std_type, -jnp.inf)
            additional_repertoire = repertoire.replace(fitnesses=std_type)
            reeval_std_type = jnp.where(
                reeval_repertoire.fitnesses > -jnp.inf, reeval_std_type, -jnp.inf
            )
            reeval_additional_repertoire = reeval_repertoire.replace(
                fitnesses=reeval_std_type
            )

        # If arm_gaussian_desc_fitprop_variance task, additional is task metrics
        elif "arm_gaussian_desc_fitprop_variance" in self._args.env_name:

            # WARNING: hyperparameters as they are set nowhere as arguments
            min_fitness = -0.5
            max_fitness = 0
            prop_factors = jnp.asarray([0.1, 0.1], dtype=jnp.float32)

            # Set the stds as the additional score
            fitnesses = jax.vmap(
                lambda x: -jnp.sqrt(jnp.mean(jnp.square(x - jnp.mean(x))))
            )(repertoire.genotypes)
            stds = jax.vmap(
                lambda x: prop_factors
                * (
                    1
                    - (jnp.clip(x, min_fitness, max_fitness) - min_fitness)
                    / (max_fitness - min_fitness)
                )
            )(fitnesses)
            reeval_fitnesses = jax.vmap(
                lambda x: -jnp.sqrt(jnp.mean(jnp.square(x - jnp.mean(x))))
            )(reeval_repertoire.genotypes)
            reeval_stds = jax.vmap(
                lambda x: prop_factors
                * (
                    1
                    - (jnp.clip(x, min_fitness, max_fitness) - min_fitness)
                    / (max_fitness - min_fitness)
                )
            )(reeval_fitnesses)

            # Set dead individuals to -jnp.inf
            stds = jnp.where(
                jnp.repeat(
                    jnp.expand_dims(repertoire.fitnesses > -jnp.inf, axis=1),
                    prop_factors.shape[0],
                    axis=1,
                ),
                stds,
                -jnp.inf,
            )
            reeval_stds = jnp.where(
                jnp.repeat(
                    jnp.expand_dims(reeval_repertoire.fitnesses > -jnp.inf, axis=1),
                    prop_factors.shape[0],
                    axis=1,
                ),
                reeval_stds,
                -jnp.inf,
            )

            # Set as additional repertoire
            additional_repertoire = repertoire.replace(fitnesses=stds)
            reeval_additional_repertoire = reeval_repertoire.replace(
                fitnesses=reeval_stds
            )

        # Default to 0s
        else:
            additional_repertoire = repertoire.replace(
                fitnesses=jnp.zeros(self._args.num_centroids)
            )
            reeval_additional_repertoire = reeval_repertoire.replace(
                fitnesses=jnp.zeros(self._args.num_centroids)
            )

        # Compute all the metrics
        reeval_metrics = self._metrics_function(reeval_repertoire)
        fit_reeval_metrics = self._metrics_function(fit_reeval_repertoire)
        desc_reeval_metrics = self._metrics_function(desc_reeval_repertoire)
        fit_var_metrics = self._metrics_function(fit_var_repertoire)
        reeval_fit_var_metrics = self._metrics_function(reeval_fit_var_repertoire)
        desc_var_metrics = self._metrics_function(desc_var_repertoire)
        reeval_desc_var_metrics = self._metrics_function(reeval_desc_var_repertoire)
        additional_metrics = self._metrics_function(additional_repertoire)
        reeval_additional_metrics = self._metrics_function(reeval_additional_repertoire)

        # Return
        return (
            reeval_metrics,
            reeval_repertoire,
            fit_reeval_metrics,
            fit_reeval_repertoire,
            desc_reeval_metrics,
            desc_reeval_repertoire,
            fit_var_metrics,
            fit_var_repertoire,
            reeval_fit_var_metrics,
            reeval_fit_var_repertoire,
            desc_var_metrics,
            desc_var_repertoire,
            reeval_desc_var_metrics,
            reeval_desc_var_repertoire,
            additional_metrics,
            additional_repertoire,
            reeval_additional_metrics,
            reeval_additional_repertoire,
            random_key,
        )

    def _create_metrics_files(self) -> None:
        """Create and initialise the metric files."""
        self._args.metrics_file = (
            f"{self._args.results}/metrics_{self._name}_{str(self._args.seed)}.csv"
        )
        self._args.in_cell_metrics_file = f"{self._args.results}/in_cell_metrics_{self._name}_{str(self._args.seed)}.csv"

    def _create_all_repertoire_folders(self) -> None:
        """Name and create all the folder to save the repertoires."""

        # Name the folders
        repertoire_suffixe = (
            "repertoire_" + self._name + "_" + str(self._args.seed) + "/"
        )
        self._args.results_repertoire = self._args.results + "/" + repertoire_suffixe
        self._args.results_projected_repertoire = (
            self._args.results + "/projected_" + repertoire_suffixe
        )
        self._args.results_reeval_repertoire = (
            self._args.results + "/reeval_" + repertoire_suffixe
        )
        self._args.results_fit_reeval_repertoire = (
            self._args.results + "/fit_reeval_" + repertoire_suffixe
        )
        self._args.results_desc_reeval_repertoire = (
            self._args.results + "/desc_reeval_" + repertoire_suffixe
        )
        self._args.results_fit_var_repertoire = (
            self._args.results + "/fit_var_" + repertoire_suffixe
        )
        self._args.results_reeval_fit_var_repertoire = (
            self._args.results + "/reeval_fit_var_" + repertoire_suffixe
        )
        self._args.results_desc_var_repertoire = (
            self._args.results + "/desc_var_" + repertoire_suffixe
        )
        self._args.results_reeval_desc_var_repertoire = (
            self._args.results + "/reeval_desc_var_" + repertoire_suffixe
        )
        self._args.results_additional_repertoire = (
            self._args.results + "/additional_" + repertoire_suffixe
        )
        self._args.results_reeval_additional_repertoire = (
            self._args.results + "/reeval_additional_" + repertoire_suffixe
        )
        self._args.results_in_cell_reeval_repertoire = (
            self._args.results + "/in_cell_reeval_" + repertoire_suffixe
        )
        self._args.results_in_cell_fit_reeval_repertoire = (
            self._args.results + "/in_cell_fit_reeval_" + repertoire_suffixe
        )
        self._args.results_in_cell_desc_reeval_repertoire = (
            self._args.results + "/in_cell_desc_reeval_" + repertoire_suffixe
        )
        self._args.results_in_cell_fit_var_repertoire = (
            self._args.results + "/in_cell_fit_var_" + repertoire_suffixe
        )
        self._args.results_in_cell_reeval_fit_var_repertoire = (
            self._args.results + "/in_cell_reeval_fit_var_" + repertoire_suffixe
        )
        self._args.results_in_cell_desc_var_repertoire = (
            self._args.results + "/in_cell_desc_var_" + repertoire_suffixe
        )
        self._args.results_in_cell_reeval_desc_var_repertoire = (
            self._args.results + "/in_cell_reeval_desc_var_" + repertoire_suffixe
        )

        # Create the folders
        if not os.path.exists(self._args.results):
            os.mkdir(self._args.results)
        if not os.path.exists(self._args.results_repertoire):
            os.mkdir(self._args.results_repertoire)
        if not os.path.exists(self._args.results_projected_repertoire):
            os.mkdir(self._args.results_projected_repertoire)
        if not os.path.exists(self._args.results_reeval_repertoire):
            os.mkdir(self._args.results_reeval_repertoire)
        if not os.path.exists(self._args.results_fit_reeval_repertoire):
            os.mkdir(self._args.results_fit_reeval_repertoire)
        if not os.path.exists(self._args.results_desc_reeval_repertoire):
            os.mkdir(self._args.results_desc_reeval_repertoire)
        if not os.path.exists(self._args.results_fit_var_repertoire):
            os.mkdir(self._args.results_fit_var_repertoire)
        if not os.path.exists(self._args.results_reeval_fit_var_repertoire):
            os.mkdir(self._args.results_reeval_fit_var_repertoire)
        if not os.path.exists(self._args.results_desc_var_repertoire):
            os.mkdir(self._args.results_desc_var_repertoire)
        if not os.path.exists(self._args.results_reeval_desc_var_repertoire):
            os.mkdir(self._args.results_reeval_desc_var_repertoire)
        if not os.path.exists(self._args.results_additional_repertoire):
            os.mkdir(self._args.results_additional_repertoire)
        if not os.path.exists(self._args.results_reeval_additional_repertoire):
            os.mkdir(self._args.results_reeval_additional_repertoire)
        if not os.path.exists(self._args.results_in_cell_reeval_repertoire):
            os.mkdir(self._args.results_in_cell_reeval_repertoire)
        if not os.path.exists(self._args.results_in_cell_fit_reeval_repertoire):
            os.mkdir(self._args.results_in_cell_fit_reeval_repertoire)
        if not os.path.exists(self._args.results_in_cell_desc_reeval_repertoire):
            os.mkdir(self._args.results_in_cell_desc_reeval_repertoire)
        if not os.path.exists(self._args.results_in_cell_fit_var_repertoire):
            os.mkdir(self._args.results_in_cell_fit_var_repertoire)
        if not os.path.exists(self._args.results_in_cell_reeval_fit_var_repertoire):
            os.mkdir(self._args.results_in_cell_reeval_fit_var_repertoire)
        if not os.path.exists(self._args.results_in_cell_desc_var_repertoire):
            os.mkdir(self._args.results_in_cell_desc_var_repertoire)
        if not os.path.exists(self._args.results_in_cell_reeval_desc_var_repertoire):
            os.mkdir(self._args.results_in_cell_reeval_desc_var_repertoire)
