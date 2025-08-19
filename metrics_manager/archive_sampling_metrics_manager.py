import time
from typing import Any, Callable, Tuple

import jax.numpy as jnp
from qdax.core.containers.repertoire import Repertoire
from qdax.custom_types import RNGKey

from metrics_manager.default_metrics_manager import MetricsManager


class ArchiveSamplingMetricsManager(MetricsManager):
    """
    Write the usual MetricsManager to add an additional files
    with a few more metrics specific to ArchiveSampling.
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

        super().__init__(
            args=args,
            name=name,
            scoring_fn=scoring_fn,
            reevaluation_fn=reevaluation_fn,
            reevaluation_in_cell_fn=reevaluation_in_cell_fn,
            metric_repertoire=metric_repertoire,
        )

        # Create an additional file for saving specific metrics
        self._archive_sampling_metrics_file = f"{self._args.results}/archive_sampling_metrics_{self._name}_{str(self._args.seed)}.csv"
        file_metrics = open(self._archive_sampling_metrics_file, "w")
        file_metrics.write(
            "epoch,eval,time,timestep,"
            + "overall_total_samples,overall_average_samples,"
            + "overall_min_samples,overall_max_samples,"
            + "overall_count_max,"
            + "top_total_samples,top_average_samples,"
            + "top_min_samples,top_max_samples,"
            + "top_count_max,"
            + "\n"
        )
        file_metrics.flush()
        file_metrics.close()

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

        metrics_t, reeval_t, write_t = super().write_all_metrics(
            epoch=epoch,
            evals=evals,
            real_evals=real_evals,
            timesteps=timesteps,
            real_timesteps=real_timesteps,
            current_time=current_time,
            repertoire=repertoire,
            projected_repertoire=projected_repertoire,
            emitter_state=emitter_state,
            random_key=random_key,
            final=final,
        )

        start_t = time.time()

        # Computing number of samples in the archive
        overall_samples = jnp.nansum(
            jnp.logical_not(jnp.isnan(repertoire.fitnesses_depth_all)),
            axis=2,
        )
        overall_samples = jnp.where(
            overall_samples == overall_samples, overall_samples, 0
        )
        top_samples = overall_samples[:, 0]

        # Getting specific metrics
        overall_total_samples = jnp.sum(overall_samples)
        overall_average_samples = jnp.sum(overall_samples) / jnp.sum(
            overall_samples > 0
        )
        overall_min_samples = jnp.min(overall_samples)
        overall_max_samples = jnp.max(overall_samples)
        overall_count_max = jnp.sum(overall_samples == self._args.max_number_evals)

        top_total_samples = jnp.sum(top_samples)
        top_average_samples = jnp.sum(top_samples) / jnp.sum(top_samples > 0)
        top_min_samples = jnp.min(top_samples)
        top_max_samples = jnp.max(top_samples)
        top_count_max = jnp.sum(top_samples == self._args.max_number_evals)

        # Saving specific metrics in additional file
        file_metrics = open(self._archive_sampling_metrics_file, "a")
        file_metrics.write(
            f"{epoch},{evals},{current_time},{timesteps},"
            + f"{overall_total_samples},{overall_average_samples},"
            + f"{overall_min_samples},{overall_max_samples},"
            + f"{overall_count_max},"
            + f"{top_total_samples},{top_average_samples},"
            + f"{top_min_samples},{top_max_samples},"
            + f"{top_count_max},"
            + "\n"
        )
        file_metrics.flush()
        file_metrics.close()

        write_t += time.time() - start_t

        return metrics_t, reeval_t, write_t
