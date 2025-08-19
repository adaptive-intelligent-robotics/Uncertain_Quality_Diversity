import os
import time
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from qdax.core.containers.repertoire import Repertoire
from qdax.custom_types import Descriptor, Fitness, Genotype, RNGKey


@jax.jit
def weighted_sum_projection_function(
    repertoire: Repertoire,
    preference: jnp.ndarray,
    metric_repertoire: Repertoire,
) -> Repertoire:
    """
    Project according to a weighted sum of preferences.
    """

    @jax.jit
    def _project_cell(
        cell_fitnesses: Fitness,
        cell_descriptors: Descriptor,
        cell_genotypes: Genotype,
    ) -> Tuple[Fitness, Descriptor, Genotype]:

        scalar_fitnesses = cell_fitnesses @ preference
        best_index = jnp.nanargmax(scalar_fitnesses)

        cell_projected_fitnesses = cell_fitnesses.at[best_index, 0].get()
        cell_projected_descriptors = cell_descriptors.at[best_index].get()
        cell_projected_genotypes = jax.tree_util.tree_map(
            lambda x: x.at[best_index].get(), cell_genotypes
        )
        return (
            cell_projected_fitnesses,
            cell_projected_descriptors,
            cell_projected_genotypes,
        )

    # Get the projected individuals
    projected_fitnesses, projected_descriptors, projected_genotypes = jax.vmap(
        _project_cell
    )(
        repertoire.fitnesses,
        repertoire.descriptors,
        repertoire.genotypes,
    )

    # Add to projected repertoire
    projected_repertoire = metric_repertoire.add(
        projected_genotypes,
        projected_descriptors,
        projected_fitnesses,
        {},
    )
    return projected_repertoire


@partial(jax.jit, static_argnames=("objective",))
def extreme_projection_function(
    repertoire: Repertoire,
    metric_repertoire: Repertoire,
    objective: int,
) -> Repertoire:
    """
    Project into the best fitness indiv per cell.
    """

    @jax.jit
    def _project_cell(
        cell_fitnesses: Fitness,
        cell_descriptors: Descriptor,
        cell_genotypes: Genotype,
    ) -> Tuple[Fitness, Descriptor, Genotype]:

        best_index = jnp.nanargmax(cell_fitnesses.at[:, objective].get())

        cell_projected_fitnesses = cell_fitnesses.at[best_index, 0].get()
        cell_projected_descriptors = cell_descriptors.at[best_index].get()
        cell_projected_genotypes = jax.tree_util.tree_map(
            lambda x: x.at[best_index].get(), cell_genotypes
        )
        return (
            cell_projected_fitnesses,
            cell_projected_descriptors,
            cell_projected_genotypes,
        )

    # Get the projected individuals
    projected_fitnesses, projected_descriptors, projected_genotypes = jax.vmap(
        _project_cell
    )(
        repertoire.fitnesses,
        repertoire.descriptors,
        repertoire.genotypes,
    )

    # Add to projected repertoire
    projected_repertoire = metric_repertoire.add(
        projected_genotypes,
        projected_descriptors,
        projected_fitnesses,
        {},
    )
    return projected_repertoire


@partial(jax.jit, static_argnames=("objective",))
def epistemic_extreme_projection_function(
    repertoire: Repertoire,
    metric_repertoire: Repertoire,
    objective: int,
) -> Repertoire:
    """
    Project into a depth grid maximising objective,
    then project again to get one indiv per cell using
    epistemic projection.
    """

    @jax.jit
    def _project_cell(
        cell_fitnesses_depth: Fitness,
        cell_fitnesses_depth_all: Fitness,
        cell_descriptors_depth: Descriptor,
        cell_genotypes_depth: Genotype,
    ) -> Tuple[Fitness, Descriptor, Genotype]:

        depth = cell_fitnesses_depth.shape[0]
        pareto_front = cell_fitnesses_depth.shape[1]

        # First, project into a depth grid maximising objective
        all_fitnesses = jnp.reshape(
            cell_fitnesses_depth.at[:, :, objective].get(),
            (depth * pareto_front,),
        )
        all_evaluations = jnp.reshape(
            jnp.sum(jnp.logical_not(jnp.isnan(cell_fitnesses_depth_all)), axis=2),
            (depth * pareto_front,),
        )
        _, top_fit_indices = jax.lax.top_k(all_fitnesses, depth)
        all_evaluations = jnp.where(
            jnp.isin(jnp.arange(len(all_fitnesses)), top_fit_indices),
            all_evaluations,
            0,
        )

        # Second, keep the indiv with higher evaluations
        final_indices = jnp.nanargmax(all_evaluations)
        cell_projected_fitnesses = (
            jnp.reshape(
                cell_fitnesses_depth.at[:, :, 0].get(),
                (depth * pareto_front,),
            )
            .at[final_indices]
            .get()
        )
        cell_projected_descriptors = (
            jnp.reshape(
                cell_descriptors_depth,
                (depth * pareto_front, -1),
            )
            .at[final_indices]
            .get()
        )
        cell_projected_genotypes = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (depth * pareto_front,) + x.shape[2:])
            .at[final_indices]
            .get(),
            cell_genotypes_depth,
        )
        return (
            cell_projected_fitnesses,
            cell_projected_descriptors,
            cell_projected_genotypes,
        )

    # Get the projected individuals
    projected_fitnesses, projected_descriptors, projected_genotypes = jax.vmap(
        _project_cell
    )(
        repertoire.fitnesses_depth,
        repertoire.fitnesses_depth_all,
        repertoire.descriptors_depth,
        repertoire.genotypes_depth,
    )

    # Add to projected repertoire
    projected_repertoire = metric_repertoire.add(
        projected_genotypes,
        projected_descriptors,
        projected_fitnesses,
        {},
    )
    return projected_repertoire


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
        reevaluation_fn: Callable,
        reevaluation_in_cell_fn: Callable,
        metric_repertoire: Repertoire,
        DefaultMetricsManager: Any,
    ) -> None:

        self._args = deepcopy(args)

        # Split the different projections
        self._projected_delta_fitness = [
            x for x in args.mome_projected_delta_fitness.split("_")
        ]
        self._projected_delta_reproducibility = [
            x for x in args.mome_projected_delta_reproducibility.split("_")
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
                + self._projected_delta_fitness[projection]
                + "-r"
                + self._projected_delta_reproducibility[projection]
                + "-"
                + str(self._rho)
            )

            # Second, create the corresponding metrics_manager
            metrics_manager = DefaultMetricsManager(
                args=deepcopy(args),
                name=projection_name,
                scoring_fn=scoring_fn,
                reevaluation_fn=reevaluation_fn,
                reevaluation_in_cell_fn=reevaluation_in_cell_fn,
                metric_repertoire=metric_repertoire,
            )
            self._metrics_managers.append(metrics_manager)

        # Also saved the non-projectd archive
        repertoire_suffixe = "repertoire_original_MOME_" + str(self._args.seed) + "/"
        self._mome_repertoire = self._args.results + "/" + repertoire_suffixe
        if not os.path.exists(self._mome_repertoire):
            os.mkdir(self._mome_repertoire)

        # Pre-jit the projection functions
        self.projection_function = partial(
            weighted_sum_projection_function,
            metric_repertoire=metric_repertoire,
        )
        self.fit_projection_function = partial(
            extreme_projection_function,
            objective=0,
            metric_repertoire=metric_repertoire,
        )
        self.reprod_projection_function = partial(
            extreme_projection_function,
            objective=1,
            metric_repertoire=metric_repertoire,
        )
        self.fit_epistemic_projection_function = partial(
            epistemic_extreme_projection_function,
            objective=0,
            metric_repertoire=metric_repertoire,
        )
        self.reprod_epistemic_projection_function = partial(
            epistemic_extreme_projection_function,
            objective=1,
            metric_repertoire=metric_repertoire,
        )

        # Create an additional file for saving specific metrics
        self._mome_metrics_file = (
            f"{self._args.results}/mome_metrics_{name}_{str(self._args.seed)}.csv"
        )
        file_metrics = open(self._mome_metrics_file, "w")
        if self._args.depth > 1:
            mome_metrics_name = (
                "epoch,eval,time,timestep,"
                + "total_filled,average_filled,"
                + "min_filled,max_filled,"
                + "count_max,"
                + "overall_total_filled,overall_average_filled,"
                + "overall_min_filled,overall_max_filled,"
                + "overall_count_max,"
                + "\n"
            )
        else:
            mome_metrics_name = (
                "epoch,eval,time,timestep,"
                + "total_filled,average_filled,"
                + "min_filled,max_filled,"
                + "count_max,"
                + "\n"
            )
        file_metrics.write(mome_metrics_name)
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
        """Wrap the main metrics function of DefaultMetricsManager for each projection."""

        metrics_t = 0.0
        reeval_t = 0.0
        write_t = 0.0

        # Call write_all_metrics for each projection
        for projection in range(self._num_projections):

            # First, compute the projection
            start_t = time.time()
            if self._projected_delta_fitness[projection] == "fit":
                current_projected_repertoire = self.fit_projection_function(
                    repertoire=repertoire,
                )
            elif self._projected_delta_fitness[projection] == "reprod":
                current_projected_repertoire = self.reprod_projection_function(
                    repertoire=repertoire,
                )
            elif self._projected_delta_fitness[projection] == "fitepistemic":
                current_projected_repertoire = self.fit_epistemic_projection_function(
                    repertoire=repertoire,
                )
            elif self._projected_delta_fitness[projection] == "reprodepistemic":
                current_projected_repertoire = (
                    self.reprod_epistemic_projection_function(
                        repertoire=repertoire,
                    )
                )
            else:
                mome_projection_preference = jnp.array(
                    [
                        1,
                        (float(self._projected_delta_fitness[projection]))
                        / (float(self._projected_delta_reproducibility[projection])),
                    ]
                )
                current_projected_repertoire = self.projection_function(
                    repertoire=repertoire,
                    preference=mome_projection_preference,
                )
            metrics_t += time.time() - start_t

            # Second, call the metrics for this projection
            sub_metrics_t, sub_reeval_t, sub_write_t = self._metrics_managers[
                projection
            ].write_all_metrics(
                epoch=epoch,
                evals=evals,
                real_evals=real_evals,
                timesteps=timesteps,
                real_timesteps=real_timesteps,
                current_time=current_time,
                repertoire=repertoire,
                projected_repertoire=current_projected_repertoire,
                emitter_state=emitter_state,
                random_key=random_key,
                final=final,
            )

            # Third, add the timings
            metrics_t += sub_metrics_t
            reeval_t += sub_reeval_t
            write_t += sub_write_t

        # Also saved the non-projectd archive
        start_t = time.time()
        if epoch % self._args.archive_log_period == 0 or final:
            repertoire.save(path=self._mome_repertoire)
        write_t += time.time() - start_t

        # Computing number of filled in the archive
        start_t = time.time()
        filled_pareto = jnp.all(repertoire.fitnesses > -jnp.inf, axis=2)
        filled_cell = jnp.any(filled_pareto, axis=1)

        total_filled = jnp.sum(filled_pareto)
        average_filled = total_filled / jnp.sum(filled_cell)

        filled_pareto_number = jnp.sum(filled_pareto, axis=1)
        min_filled = jnp.min(
            jnp.where(filled_pareto_number == 0, jnp.inf, filled_pareto_number)
        )
        max_filled = jnp.max(filled_pareto_number)
        count_max = jnp.sum(filled_pareto_number == repertoire.fitnesses.shape[1])

        if self._args.depth > 1:
            overall_filled_pareto = jnp.all(
                repertoire.fitnesses_depth > -jnp.inf, axis=3
            )
            overall_filled_cell = jnp.any(overall_filled_pareto, axis=2)

            overall_total_filled = jnp.sum(overall_filled_pareto)
            overall_average_filled = overall_total_filled / jnp.sum(overall_filled_cell)

            overall_filled_pareto_number = jnp.sum(overall_filled_pareto, axis=2)
            overall_min_filled = jnp.min(
                jnp.where(
                    overall_filled_pareto_number == 0,
                    jnp.inf,
                    overall_filled_pareto_number,
                )
            )
            overall_max_filled = jnp.max(overall_filled_pareto_number)
            overall_count_max = jnp.sum(
                overall_filled_pareto_number == repertoire.fitnesses.shape[1]
            )

        # Saving specific metrics in additional file
        file_metrics = open(self._mome_metrics_file, "a")
        if self._args.depth > 1:
            mome_metrics_values = (
                f"{epoch},{evals},{current_time},{timesteps},"
                + f"{total_filled},{average_filled},"
                + f"{min_filled},{max_filled},"
                + f"{count_max},"
                + f"{overall_total_filled},{overall_average_filled},"
                + f"{overall_min_filled},{overall_max_filled},"
                + f"{overall_count_max},"
                + "\n"
            )
        else:
            mome_metrics_values = (
                f"{epoch},{evals},{current_time},{timesteps},"
                + f"{total_filled},{average_filled},"
                + f"{min_filled},{max_filled},"
                + f"{count_max},"
                + "\n"
            )
        file_metrics.write(mome_metrics_values)
        file_metrics.flush()
        file_metrics.close()

        write_t += time.time() - start_t

        # Return all the timings
        return metrics_t, reeval_t, write_t
