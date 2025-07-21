import os
import time
from functools import partial
from typing import Any, Callable, List, Tuple

import jax
import jax.numpy as jnp
from qdax.core.containers.repertoire import Repertoire
from qdax.types import Metrics, RNGKey

from core.stochasticity_utils import reevaluation_function
from metrics_manager.utils import create_metrics_csv, save_config, write_metrics_csv
from set_up_container import EXTRACTOR_LIST
from tasks.direct_mapping import direct_mapping


class MetricsManager:
    """
    Class that contains all metrics computation and writting.
    """

    def __init__(
        self,
        args: Any,
        name: str,
        scoring_fn: Callable,
        metric_repertoire: Repertoire,
        evals_per_iter: int,
        min_bd: List,
        max_bd: List,
        qd_offset: float,
    ) -> None:
        """Set up all the necesary attributes and create all
        necessary folders and initialise all functions."""

        self._args = args
        self._name = name
        self._scoring_fn = scoring_fn
        self._metric_repertoire = metric_repertoire
        self._qd_offset = qd_offset

        # Input check
        if self._args.reeval_scan_size > 0:
            if self._args.num_reevals == 0:
                print(
                    "!!!WARNING!!! --reeval-scan-size has no impact with --num-reevals 0."
                )
                self._args.reeval_scan_size = 0
            elif self._args.num_reevals % self._args.reeval_scan_size != 0:
                assert (
                    0
                ), "\n!!!ERROR!!! --num-reevals non divible by --reeval-scan-size."
        assert (
            self._args.reeval_fitness_extractor in EXTRACTOR_LIST.keys()
        ), "\n !!!ERROR!!! invalid reeval_fitness_extractor."
        assert (
            self._args.reeval_fitness_reproducibility_extractor in EXTRACTOR_LIST.keys()
        ), "\n !!!ERROR!!! invalid reeval_fitness_reproducibility_extractor."
        assert (
            self._args.reeval_descriptor_extractor in EXTRACTOR_LIST.keys()
        ), "\n !!!ERROR!!! invalid reeval_descriptor_extractor."
        assert (
            self._args.reeval_descriptor_reproducibility_extractor
            in EXTRACTOR_LIST.keys()
        ), "\n !!!ERROR!!! invalid reeval_descriptor_reproducibility_extractor."

        # Create results folder
        if not os.path.exists(self._args.results):
            os.mkdir(self._args.results)

        # Create all the folders to save repertoires
        self._create_all_repertoire_folders()

        # Create the metrics files
        self._create_metrics_files()

        # Prepare the reeval metric functions
        if "direct_mapping" in self._args.env_name:
            self._reevaluation_metrics_function = (
                self._direct_mapping_reevaluation_metrics_function
            )
        else:
            self._reevaluation_metrics_function = (
                self._default_reevaluation_metrics_function
            )

        # Write the corresponding config
        save_config(
            save_folder=self._args.results,
            name=self._name,
            seed=self._args.seed,
            env_name=self._args.env_name,
            episode_length=self._args.episode_length,
            params_std=self._args.params_std,
            min_bd=min_bd,
            max_bd=max_bd,
            batch_size=self._args.batch_size,
            sampling_size=self._args.sampling_size,
            sampling_use=evals_per_iter,
            num_iterations=self._args.num_iterations,
            policy_hidden_layer_sizes=self._args.policy_hidden_layer_sizes,
            num_init_cvt_samples=self._args.num_init_cvt_samples,
            num_centroids=self._args.num_centroids,
            num_samples=self._args.num_samples,
            num_reevals=self._args.num_reevals,
            depth=self._args.depth,
            delta_fitness=self._args.mer_delta_fitness,
            delta_reproducibility=self._args.mer_delta_reproducibility,
            metrics_file=self._metrics_file,
            in_cell_metrics_file=self._in_cell_metrics_file,
            save_folder_repertoire=self._results_repertoire,
            save_folder_projected_repertoire=self._results_projected_repertoire,
            save_folder_reeval_repertoire=self._results_reeval_repertoire,
            save_folder_fit_reeval_repertoire=self._results_fit_reeval_repertoire,
            save_folder_desc_reeval_repertoire=self._results_desc_reeval_repertoire,
            save_folder_fit_var_repertoire=self._results_fit_var_repertoire,
            save_folder_reeval_fit_var_repertoire=self._results_reeval_fit_var_repertoire,
            save_folder_desc_var_repertoire=self._results_desc_var_repertoire,
            save_folder_reeval_desc_var_repertoire=self._results_reeval_desc_var_repertoire,
            save_folder_additional_repertoire=self._results_additional_repertoire,
            save_folder_reeval_additional_repertoire=self._results_reeval_additional_repertoire,
            save_folder_in_cell_reeval_repertoire=self._results_in_cell_reeval_repertoire,
            save_folder_in_cell_fit_reeval_repertoire=self._results_in_cell_fit_reeval_repertoire,
            save_folder_in_cell_desc_reeval_repertoire=self._results_in_cell_desc_reeval_repertoire,
            save_folder_in_cell_fit_var_repertoire=self._results_in_cell_fit_var_repertoire,
            save_folder_in_cell_reeval_fit_var_repertoire=self._results_in_cell_reeval_fit_var_repertoire,
            save_folder_in_cell_desc_var_repertoire=self._results_in_cell_desc_var_repertoire,
            save_folder_in_cell_reeval_desc_var_repertoire=self._results_in_cell_reeval_desc_var_repertoire,
        )

    def write_all_metrics(
        self,
        epoch: int,
        evals: float,
        current_time: float,
        evals_per_offspring: int,
        evals_per_iter: int,
        batch_size: int,
        repertoire: Repertoire,
        projected_repertoire: Repertoire,
        emitter_state: Any,
        random_key: RNGKey,
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
        jax.tree_util.tree_map(
            lambda x: x.block_until_ready(), metrics
        )  # ensure timing accuracy
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
            if min(fitnesses) < -self._qd_offset:
                print(
                    f"!!!WARNING!!! got min fit {min(fitnesses)} < {-self._qd_offset},"
                    "may lead to inacurate QD-Score."
                )

        reeval_fitnesses = reeval_repertoire.fitnesses
        reeval_fitnesses = jnp.where(
            reeval_fitnesses == -jnp.inf, jnp.inf, reeval_fitnesses
        )
        if min(reeval_fitnesses) < -self._qd_offset:
            print(
                f"!!!WARNING!!! got min fit {min(reeval_fitnesses)} < {-self._qd_offset},"
                "may lead to inacurate QD-Score."
            )
        metrics_t += time.time() - start_t

        # Write metrics
        start_t = time.time()
        self._write_metrics_files(
            file_name=self._metrics_file,
            epoch=epoch,
            evals=evals,
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
            evals_per_offspring=evals_per_offspring,
            evals_per_iter=evals_per_iter,
            batch_size=batch_size,
        )
        self._write_metrics_files(
            file_name=self._in_cell_metrics_file,
            epoch=epoch,
            evals=evals,
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
            evals_per_offspring=evals_per_offspring,
            evals_per_iter=evals_per_iter,
            batch_size=batch_size,
        )
        print("    -> Metrics saved in", self._metrics_file)
        print("    -> In cell metrics saved in", self._in_cell_metrics_file)

        # Write repertoire
        if epoch % self._args.archive_log_period == 0:
            repertoire.save(path=self._results_repertoire)
            projected_repertoire.save(path=self._results_projected_repertoire)
            reeval_repertoire.save(path=self._results_reeval_repertoire)
            fit_reeval_repertoire.save(path=self._results_fit_reeval_repertoire)
            desc_reeval_repertoire.save(path=self._results_desc_reeval_repertoire)
            fit_var_repertoire.save(path=self._results_fit_var_repertoire)
            reeval_fit_var_repertoire.save(path=self._results_reeval_fit_var_repertoire)
            desc_var_repertoire.save(path=self._results_desc_var_repertoire)
            reeval_desc_var_repertoire.save(
                path=self._results_reeval_desc_var_repertoire
            )
            additional_repertoire.save(path=self._results_additional_repertoire)
            reeval_additional_repertoire.save(
                path=self._results_reeval_additional_repertoire
            )
            reeval_repertoire.save(path=self._results_in_cell_reeval_repertoire)
            fit_reeval_repertoire.save(path=self._results_in_cell_fit_reeval_repertoire)
            desc_reeval_repertoire.save(
                path=self._results_in_cell_desc_reeval_repertoire
            )
            fit_var_repertoire.save(path=self._results_in_cell_fit_var_repertoire)
            reeval_fit_var_repertoire.save(
                path=self._results_in_cell_reeval_fit_var_repertoire
            )
            desc_var_repertoire.save(path=self._results_in_cell_desc_var_repertoire)
            reeval_desc_var_repertoire.save(
                path=self._results_in_cell_reeval_desc_var_repertoire
            )
            print("    -> All repertoire saved, original in", self._results_repertoire)

        write_t += time.time() - start_t

        # Return all the timings
        return metrics_t, reeval_t, write_t

    @partial(jax.jit, static_argnames=("self",))
    def _metrics_function(self, repertoire: Repertoire) -> Metrics:
        repertoire_empty = repertoire.fitnesses == -jnp.inf
        qd_score = jnp.sum(repertoire.fitnesses, where=~repertoire_empty)
        qd_score += self._qd_offset * jnp.sum(1.0 - repertoire_empty)
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
        ) = reevaluation_function(
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

    @partial(jax.jit, static_argnames=("self",))
    def _direct_mapping_reevaluation_metrics_function(
        self,
        repertoire: Repertoire,
        random_key: RNGKey,
    ) -> Tuple:

        # Apply direct mapping function to get ground truth fitnesses and descriptors
        fitnesses, descriptors = jax.vmap(direct_mapping)(repertoire.genotypes)

        # Also get ground truth var
        fitnesses_var = jnp.zeros_like(fitnesses)
        if "direct_mapping_no_trade_off" in self._args.env_name:
            descriptors_var = jnp.ones_like(fitnesses)
        elif "direct_mapping_perfect_trade_off" in self._args.env_name:
            descriptors_var = fitnesses
        elif "direct_mapping_sharp_peak" in self._args.env_name:
            descriptors_var = jnp.where(fitnesses < 0.9, 0.0, 1.0)
        elif "direct_mapping_deceptive" in self._args.env_name:
            descriptors_var = jnp.where(
                jnp.logical_or(fitnesses < 0.6, fitnesses > 0.7), 0.0, 1.0
            )
        else:
            descriptors_var = jnp.zeros_like(fitnesses)

        # Fill-in reeval repertoire
        reeval_repertoire = self._metric_repertoire.empty()
        reeval_repertoire = reeval_repertoire.add(
            repertoire.genotypes,
            descriptors,
            fitnesses,
            {},
        )

        # Fill-in fit_reeval repertoire
        fit_reeval_repertoire = self._metric_repertoire.empty()
        fit_reeval_repertoire = fit_reeval_repertoire.add(
            repertoire.genotypes,
            repertoire.descriptors,
            fitnesses,
            {},
        )

        # Fill-in desc_reeval repertoire
        desc_reeval_repertoire = self._metric_repertoire.empty()
        desc_reeval_repertoire = desc_reeval_repertoire.add(
            repertoire.genotypes,
            descriptors,
            repertoire.fitnesses,
            {},
        )

        # Fill-in fit_var repertoire
        fit_var_repertoire = self._metric_repertoire.empty()
        fit_var_repertoire = fit_var_repertoire.add(
            repertoire.genotypes,
            repertoire.descriptors,
            fitnesses_var,
            {},
        )

        # Fill-in reeval_fit_var repertoire
        reeval_fit_var_repertoire = self._metric_repertoire.empty()
        reeval_fit_var_repertoire = reeval_fit_var_repertoire.add(
            repertoire.genotypes,
            descriptors,
            fitnesses_var,
            {},
        )

        # Fill-in desc_var repertoire
        desc_var_repertoire = self._metric_repertoire.empty()
        desc_var_repertoire = desc_var_repertoire.add(
            repertoire.genotypes,
            repertoire.descriptors,
            descriptors_var,
            {},
        )

        # Fill-in reeval_desc_var repertoire
        reeval_desc_var_repertoire = self._metric_repertoire.empty()
        reeval_desc_var_repertoire = reeval_desc_var_repertoire.add(
            repertoire.genotypes,
            descriptors,
            descriptors_var,
            {},
        )

        # Additional repertoire
        full_fitnesses = jnp.where(
            repertoire.fitnesses > -jnp.inf,
            jnp.clip(repertoire.genotypes.at[:, 2].get(), 0.0, 1.0),
            -jnp.inf,
        )
        additional_repertoire = repertoire.replace(fitnesses=full_fitnesses)
        reeval_full_fitnesses = jnp.where(
            reeval_repertoire.fitnesses > -jnp.inf,
            jnp.clip(reeval_repertoire.genotypes.at[:, 2].get(), 0.0, 1.0),
            -jnp.inf,
        )
        reeval_additional_repertoire = reeval_repertoire.replace(
            fitnesses=reeval_full_fitnesses
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

    def _create_all_repertoire_folders(self) -> None:
        """Name and create all the folder to save the repertoires."""

        # Name the folders
        repertoire_suffixe = (
            "repertoire_" + self._name + "_" + str(self._args.seed) + "/"
        )
        self._results_repertoire = self._args.results + "/" + repertoire_suffixe
        self._results_projected_repertoire = (
            self._args.results + "/projected_" + repertoire_suffixe
        )
        self._results_reeval_repertoire = (
            self._args.results + "/reeval_" + repertoire_suffixe
        )
        self._results_fit_reeval_repertoire = (
            self._args.results + "/fit_reeval_" + repertoire_suffixe
        )
        self._results_desc_reeval_repertoire = (
            self._args.results + "/desc_reeval_" + repertoire_suffixe
        )
        self._results_fit_var_repertoire = (
            self._args.results + "/fit_var_" + repertoire_suffixe
        )
        self._results_reeval_fit_var_repertoire = (
            self._args.results + "/reeval_fit_var_" + repertoire_suffixe
        )
        self._results_desc_var_repertoire = (
            self._args.results + "/desc_var_" + repertoire_suffixe
        )
        self._results_reeval_desc_var_repertoire = (
            self._args.results + "/reeval_desc_var_" + repertoire_suffixe
        )
        self._results_additional_repertoire = (
            self._args.results + "/additional_" + repertoire_suffixe
        )
        self._results_reeval_additional_repertoire = (
            self._args.results + "/reeval_additional_" + repertoire_suffixe
        )
        self._results_in_cell_reeval_repertoire = (
            self._args.results + "/in_cell_reeval_" + repertoire_suffixe
        )
        self._results_in_cell_fit_reeval_repertoire = (
            self._args.results + "/in_cell_fit_reeval_" + repertoire_suffixe
        )
        self._results_in_cell_desc_reeval_repertoire = (
            self._args.results + "/in_cell_desc_reeval_" + repertoire_suffixe
        )
        self._results_in_cell_fit_var_repertoire = (
            self._args.results + "/in_cell_fit_var_" + repertoire_suffixe
        )
        self._results_in_cell_reeval_fit_var_repertoire = (
            self._args.results + "/in_cell_reeval_fit_var_" + repertoire_suffixe
        )
        self._results_in_cell_desc_var_repertoire = (
            self._args.results + "/in_cell_desc_var_" + repertoire_suffixe
        )
        self._results_in_cell_reeval_desc_var_repertoire = (
            self._args.results + "/in_cell_reeval_desc_var_" + repertoire_suffixe
        )

        # Create the folders
        if not os.path.exists(self._args.results):
            os.mkdir(self._args.results)
        if not os.path.exists(self._results_repertoire):
            os.mkdir(self._results_repertoire)
        if not os.path.exists(self._results_projected_repertoire):
            os.mkdir(self._results_projected_repertoire)
        if not os.path.exists(self._results_reeval_repertoire):
            os.mkdir(self._results_reeval_repertoire)
        if not os.path.exists(self._results_fit_reeval_repertoire):
            os.mkdir(self._results_fit_reeval_repertoire)
        if not os.path.exists(self._results_desc_reeval_repertoire):
            os.mkdir(self._results_desc_reeval_repertoire)
        if not os.path.exists(self._results_fit_var_repertoire):
            os.mkdir(self._results_fit_var_repertoire)
        if not os.path.exists(self._results_reeval_fit_var_repertoire):
            os.mkdir(self._results_reeval_fit_var_repertoire)
        if not os.path.exists(self._results_desc_var_repertoire):
            os.mkdir(self._results_desc_var_repertoire)
        if not os.path.exists(self._results_reeval_desc_var_repertoire):
            os.mkdir(self._results_reeval_desc_var_repertoire)
        if not os.path.exists(self._results_additional_repertoire):
            os.mkdir(self._results_additional_repertoire)
        if not os.path.exists(self._results_reeval_additional_repertoire):
            os.mkdir(self._results_reeval_additional_repertoire)
        if not os.path.exists(self._results_in_cell_reeval_repertoire):
            os.mkdir(self._results_in_cell_reeval_repertoire)
        if not os.path.exists(self._results_in_cell_fit_reeval_repertoire):
            os.mkdir(self._results_in_cell_fit_reeval_repertoire)
        if not os.path.exists(self._results_in_cell_desc_reeval_repertoire):
            os.mkdir(self._results_in_cell_desc_reeval_repertoire)
        if not os.path.exists(self._results_in_cell_fit_var_repertoire):
            os.mkdir(self._results_in_cell_fit_var_repertoire)
        if not os.path.exists(self._results_in_cell_reeval_fit_var_repertoire):
            os.mkdir(self._results_in_cell_reeval_fit_var_repertoire)
        if not os.path.exists(self._results_in_cell_desc_var_repertoire):
            os.mkdir(self._results_in_cell_desc_var_repertoire)
        if not os.path.exists(self._results_in_cell_reeval_desc_var_repertoire):
            os.mkdir(self._results_in_cell_reeval_desc_var_repertoire)

    def _create_metrics_files(self) -> None:
        """Create and initialise the metric files."""
        self._metrics_file = (
            f"{self._args.results}/metrics_{self._name}_{str(self._args.seed)}.csv"
        )
        create_metrics_csv(file_name=self._metrics_file, prefixe="")
        self._in_cell_metrics_file = f"{self._args.results}/in_cell_metrics_{self._name}_{str(self._args.seed)}.csv"
        create_metrics_csv(file_name=self._in_cell_metrics_file, prefixe="in_cell_")

    def _write_metrics_files(
        self,
        file_name: str,
        epoch: float,
        evals: float,
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
        evals_per_offspring: int,
        evals_per_iter: int,
        batch_size: int,
    ) -> None:
        """Write the current metrics."""

        write_metrics_csv(
            file_name,
            epoch,
            evals,
            time,
            metrics["qd_score"],
            metrics["coverage"],
            metrics["max_fitness"],
            metrics["min_fitness"],
            reeval_metrics["qd_score"],
            reeval_metrics["coverage"],
            reeval_metrics["max_fitness"],
            reeval_metrics["min_fitness"],
            fit_reeval_metrics["qd_score"],
            fit_reeval_metrics["coverage"],
            fit_reeval_metrics["max_fitness"],
            fit_reeval_metrics["min_fitness"],
            desc_reeval_metrics["qd_score"],
            desc_reeval_metrics["coverage"],
            desc_reeval_metrics["max_fitness"],
            desc_reeval_metrics["min_fitness"],
            fit_var_metrics["qd_score"],
            fit_var_metrics["coverage"],
            fit_var_metrics["max_fitness"],
            fit_var_metrics["min_fitness"],
            reeval_fit_var_metrics["qd_score"],
            reeval_fit_var_metrics["coverage"],
            reeval_fit_var_metrics["max_fitness"],
            reeval_fit_var_metrics["min_fitness"],
            desc_var_metrics["qd_score"],
            desc_var_metrics["coverage"],
            desc_var_metrics["max_fitness"],
            desc_var_metrics["min_fitness"],
            reeval_desc_var_metrics["qd_score"],
            reeval_desc_var_metrics["coverage"],
            reeval_desc_var_metrics["max_fitness"],
            reeval_desc_var_metrics["min_fitness"],
            additional_metrics["qd_score"],
            additional_metrics["coverage"],
            additional_metrics["max_fitness"],
            additional_metrics["min_fitness"],
            reeval_additional_metrics["qd_score"],
            reeval_additional_metrics["coverage"],
            reeval_additional_metrics["max_fitness"],
            reeval_additional_metrics["min_fitness"],
            evals_per_offspring,
            evals_per_iter,
            batch_size,
        )
