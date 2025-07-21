import time
from functools import partial
from typing import Any, Callable, List, Tuple

import jax
import jax.numpy as jnp
from qdax.core.containers.repertoire import Repertoire
from qdax.types import Descriptor, Fitness, Genotype, RNGKey

from core.containers.mapelites_repertoire import MapElitesRepertoire


def project_reproducibility_mome_archive(
    repertoire: Repertoire,
    cell_projection_function: Callable,
) -> MapElitesRepertoire:
    """Project the MOME archive onto a single objective space."""

    projected_fitnesses, projected_descriptors, projected_genotypes = jax.vmap(
        cell_projection_function
    )(
        repertoire.fitnesses,
        repertoire.descriptors,
        repertoire.genotypes,
    )

    projected_archive = MapElitesRepertoire.init(
        genotypes=projected_genotypes,
        fitnesses=projected_fitnesses,
        descriptors=projected_descriptors,
        extra_scores={},
        centroids=repertoire.centroids,
    )

    return projected_archive


def weighted_sum_projection_function(
    cell_fitnesses: Fitness,
    cell_descriptors: Descriptor,
    cell_genotypes: Genotype,
    preference: jnp.ndarray,
) -> Tuple[Fitness, Descriptor, Genotype]:

    scalar_fitnesses = cell_fitnesses @ preference
    best_index = jnp.nanargmax(scalar_fitnesses)

    projected_fitnesses = cell_fitnesses.at[best_index].get()
    projected_fitnesses = projected_fitnesses[
        0
    ]  # only take first fitness as second is reproducibility
    projected_descriptors = cell_descriptors.at[best_index].get()
    projected_genotypes = jax.tree_util.tree_map(
        lambda x: x.at[best_index].get(), cell_genotypes
    )

    return projected_fitnesses, projected_descriptors, projected_genotypes


class MOMEMetricsManager:
    """
    Class that wrap DefaultMetricsManager to compute the same metrics for multiple
    projections of the same MOME grid.
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
        DefaultMetricsManager: Any,
    ) -> None:

        # Split the different projections
        self._projected_delta_fitness = [
            float(x) for x in args.mome_projected_delta_fitness.split("_")
        ]
        self._projected_delta_reproducibility = [
            float(x) for x in args.mome_projected_delta_reproducibility.split("_")
        ]
        self._rho = args.mer_rho

        # Some user prints
        assert len(self._projected_delta_fitness) == len(
            self._projected_delta_reproducibility
        ), "!!!ERROR!!! fitness and reproducibility projections should be the same size."
        self._num_projections = len(self._projected_delta_fitness)
        print("\n!!!WARNING!!! Using MOME multi metrics_manager.")
        print(
            "Computing metrics for each of these projections:",
            f"{self._projected_delta_fitness}, {self._projected_delta_reproducibility}",
        )

        # Create one DefaultMetricsManager per projection, each with a different name
        self._metrics_managers = []
        for projection in range(self._num_projections):

            # First, create a name for this projection
            projection_name = (
                name
                + "-proj-f"
                + str(self._projected_delta_fitness[projection])
                + "-r"
                + str(self._projected_delta_reproducibility[projection])
                + "-"
                + str(self._rho)
            )

            # Second, create the corresponding metrics_manager
            metrics_manager = DefaultMetricsManager(
                args=args,
                name=projection_name,
                scoring_fn=scoring_fn,
                metric_repertoire=metric_repertoire,
                evals_per_iter=evals_per_iter,
                min_bd=min_bd,
                max_bd=max_bd,
                qd_offset=qd_offset,
            )
            self._metrics_managers.append(metrics_manager)

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
        """Wrap the main metrics function of DefaultMetricsManager for each projection."""

        metrics_t = 0.0
        reeval_t = 0.0
        write_t = 0.0

        # Call write_all_metrics for each projection
        for projection in range(self._num_projections):

            # First, compute the projection
            start_t = time.time()
            mome_projection_preference = jnp.array(
                [
                    1,
                    (self._projected_delta_fitness[projection] + self._rho)
                    / (self._projected_delta_reproducibility[projection] + self._rho),
                ]
            )
            cell_projection_fn = partial(
                weighted_sum_projection_function, preference=mome_projection_preference
            )
            current_projected_repertoire = project_reproducibility_mome_archive(
                repertoire=repertoire,
                cell_projection_function=cell_projection_fn,
            )
            metrics_t += time.time() - start_t

            # Second, call the metrics for this projection
            sub_metrics_t, sub_reeval_t, sub_write_t = self._metrics_managers[
                projection
            ].write_all_metrics(
                epoch=epoch,
                evals=evals,
                current_time=current_time,
                evals_per_offspring=evals_per_offspring,
                evals_per_iter=evals_per_iter,
                batch_size=batch_size,
                repertoire=repertoire,
                projected_repertoire=current_projected_repertoire,
                emitter_state=emitter_state,
                random_key=random_key,
            )

            # Third, add the timings
            metrics_t += sub_metrics_t
            reeval_t += sub_reeval_t
            write_t += sub_write_t

        # Return all the timings
        return metrics_t, reeval_t, write_t
