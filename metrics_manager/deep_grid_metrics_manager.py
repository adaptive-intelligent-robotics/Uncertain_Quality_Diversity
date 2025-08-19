import time
from typing import Any, Tuple

import jax
import jax.numpy as jnp
from qdax.core.containers.repertoire import Repertoire
from qdax.custom_types import RNGKey

from metrics_manager.default_metrics_manager import MetricsManager
from metrics_manager.utils import save_metrics
from set_up_container import EXTRACTOR_LIST


class DeepGridMetricsManager(MetricsManager):
    """
    Class that compute in_cell metrics on top of standard metrics.
    """

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

        # Third, get all in-cell reeval repertoire
        start_t = time.time()
        (
            in_cell_repertoire,
            in_cell_reeval_repertoire,
            in_cell_fit_reeval_repertoire,
            in_cell_desc_reeval_repertoire,
            in_cell_fit_var_repertoire,
            in_cell_reeval_fit_var_repertoire,
            in_cell_desc_var_repertoire,
            in_cell_reeval_desc_var_repertoire,
            random_key,
        ) = self._reevaluation_in_cell_fn(
            repertoire=repertoire,
            random_key=random_key,
            metric_repertoire=self._metric_repertoire,
            scoring_fn=self._scoring_fn,
            depth=self._args.depth,
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

        # Fourth, compute in-cell reeval metrics
        in_cell_reeval_metrics = self._metrics_function(in_cell_reeval_repertoire)
        in_cell_fit_reeval_metrics = self._metrics_function(
            in_cell_fit_reeval_repertoire
        )
        in_cell_desc_reeval_metrics = self._metrics_function(
            in_cell_desc_reeval_repertoire
        )
        in_cell_fit_var_metrics = self._metrics_function(in_cell_fit_var_repertoire)
        in_cell_reeval_fit_var_metrics = self._metrics_function(
            in_cell_reeval_fit_var_repertoire
        )
        in_cell_desc_var_metrics = self._metrics_function(in_cell_desc_var_repertoire)
        in_cell_reeval_desc_var_metrics = self._metrics_function(
            in_cell_reeval_desc_var_repertoire
        )
        jax.tree_util.tree_map(
            lambda x: x.block_until_ready(), in_cell_desc_var_metrics
        )  # ensure timing accuracy
        reeval_t += time.time() - start_t

        # Fifth, compute the in-cell metrics
        start_t = time.time()
        in_cell_metrics = self._metrics_function(in_cell_repertoire)
        jax.tree_util.tree_map(
            lambda x: x.block_until_ready(), in_cell_metrics
        )  # ensure timing accuracy
        metrics_t += time.time() - start_t

        # Sanity check of min_fitness
        start_t = time.time()
        if self._args.container != "MOME-Reprod":
            fitnesses = repertoire.fitnesses
            fitnesses = jnp.where(fitnesses == -jnp.inf, jnp.inf, fitnesses)
            if min(fitnesses) < -self._args.qd_offset:
                print(
                    f"!!!WARNING!!! got min fit {min(fitnesses)} < {-self._args.qd_offset},"
                    "may lead to inacurate QD-Score."
                )

        reeval_fitnesses = reeval_repertoire.fitnesses
        reeval_fitnesses = jnp.where(
            reeval_fitnesses == -jnp.inf, jnp.inf, reeval_fitnesses
        )
        if min(reeval_fitnesses) < -self._args.qd_offset:
            print(
                f"!!!WARNING!!! got min fit {min(reeval_fitnesses)} < {-self._args.qd_offset},"
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
            metrics=in_cell_metrics,
            reeval_metrics=in_cell_reeval_metrics,
            fit_reeval_metrics=in_cell_fit_reeval_metrics,
            desc_reeval_metrics=in_cell_desc_reeval_metrics,
            fit_var_metrics=in_cell_fit_var_metrics,
            reeval_fit_var_metrics=in_cell_reeval_fit_var_metrics,
            desc_var_metrics=in_cell_desc_var_metrics,
            reeval_desc_var_metrics=in_cell_reeval_desc_var_metrics,
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
            in_cell_reeval_repertoire.save(
                path=self._args.results_in_cell_reeval_repertoire
            )
            in_cell_fit_reeval_repertoire.save(
                path=self._args.results_in_cell_fit_reeval_repertoire
            )
            in_cell_desc_reeval_repertoire.save(
                path=self._args.results_in_cell_desc_reeval_repertoire
            )
            in_cell_fit_var_repertoire.save(
                path=self._args.results_in_cell_fit_var_repertoire
            )
            in_cell_reeval_fit_var_repertoire.save(
                path=self._args.results_in_cell_reeval_fit_var_repertoire
            )
            in_cell_desc_var_repertoire.save(
                path=self._args.results_in_cell_desc_var_repertoire
            )
            in_cell_reeval_desc_var_repertoire.save(
                path=self._args.results_in_cell_reeval_desc_var_repertoire
            )
            print(
                "    -> All repertoire saved, original in",
                self._args.results_repertoire,
            )

        write_t += time.time() - start_t

        # Return all the timings
        return metrics_t, reeval_t, write_t
