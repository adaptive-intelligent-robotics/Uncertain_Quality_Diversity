from __future__ import annotations

from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from qdax.core.containers.mapelites_repertoire import get_cells_indices
from qdax.custom_types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    RNGKey,
)

from core.containers.archive_sampling_repertoire import ArchiveSamplingRepertoire


class AdaptiveSamplingRepertoire(ArchiveSamplingRepertoire):
    """
    Class for the Adaptive-Sampling from Justesen et al.
    WARNING: This is not jited.
    """

    random_key: RNGKey
    total_evaluations: int
    maximum_evals_per_generation: int

    def save(self, path: str = "./") -> None:
        """Saves the grid on disk in the form of .npy files.

        Flattens the genotypes to store it with .npy format. Supposes that
        a user will have access to the reconstruction function when loading
        the genotypes.

        Args:
            path: Path where the data will be saved. Defaults to "./".
        """

        super().save(path=path)
        jnp.save(path + "total_evaluations.npy", self.total_evaluations)

    @classmethod
    def load(
        cls, reconstruction_fn: Callable, random_key: RNGKey, path: str = "./"
    ) -> AdaptiveSamplingRepertoire:
        """Loads a MAP Elites Grid.

        Args:
            reconstruction_fn: Function to reconstruct a PyTree
                from a flat array.
            path: Path where the data is saved. Defaults to "./".

        Returns:
            A MAP Elites Repertoire.
        """

        repertoire = super().load(
            reconstruction_fn=reconstruction_fn,
            path=path,
        )
        total_evaluations = jnp.load(path + "total_evaluations.npy")

        return AdaptiveSamplingRepertoire(
            genotypes=repertoire.genotypes,
            genotypes_depth=repertoire.genotypes_depth,
            fitnesses=repertoire.fitnesses,
            fitnesses_depth=repertoire.fitnesses_depth,
            fitnesses_depth_all=repertoire.fitnesses_depth_all,
            descriptors=repertoire.descriptors,
            descriptors_depth=repertoire.descriptors_depth,
            descriptors_depth_all=repertoire.descriptors_depth_all,
            centroids=repertoire.centroids,
            total_evaluations=total_evaluations,
            maximum_evals_per_generation=0,
            random_key=random_key,
        )

    @jax.jit
    def _add_indiv(
        self,
        repertoire: AdaptiveSamplingRepertoire,
        indices: jnp.ndarray,
        genotypes: Genotype,
        fitnesses: Fitness,
        all_fitnesses: Fitness,
        descriptors: Descriptor,
        all_descriptors: Descriptor,
        extra_scores: ExtraScores,
    ) -> AdaptiveSamplingRepertoire:
        """Wrap _add_per_cell to add to the grid."""

        num_centroids = self.fitnesses_depth.shape[0]
        depth = self.fitnesses_depth.shape[1]
        batch_size = all_fitnesses.shape[0]
        out_of_bound = max(
            num_centroids * depth,
            batch_size,
        )

        @jax.jit
        def _add_per_cell(
            cell_idx: jnp.ndarray,
            cell_genotypes_depth: Genotype,
            cell_fitnesses_depth: Fitness,
            cell_fitnesses_depth_all: Fitness,
            cell_descriptors_depth: Descriptor,
            cell_descriptors_depth_all: Descriptor,
            batch_of_indices: jnp.ndarray,
            batch_of_genotypes: Genotype,
            batch_of_fitnesses: Fitness,
            batch_of_all_fitnesses: Fitness,
            batch_of_descriptors: Descriptor,
            batch_of_all_descriptors: Descriptor,
        ) -> Tuple[
            Genotype,
            Fitness,
            Fitness,
            Descriptor,
            Descriptor,
            Genotype,
            Fitness,
            Descriptor,
        ]:
            """
            For a given cell with index cell_idx, filter candidate
            indivs for this cell, and add them to it, reordering so
            highest-fitness individuals are first.

            Args:
              cell_idx: cell index
              cell_genotypes_depth: genotype in the cell
              cell_fitnesses_depth: fitnesses in the cell
              cell_fitnesses_depth_all
              cell_descriptors_depth: descriptors in the cell
              cell_descriptors_depth_all

            Returns:
              new_cell_genotypes_depth
              new_cell_fitnesses_depth
              new_cell_fitnesses_depth_all
              new_cell_descriptors_depth
              new_cell_descriptors_depth_all
              new_cell_genotype: genotype in the top layer of the cell
              new_cell_fitnesses: fitnesses in the top layer of the cell
              new_cell_descriptors: descriptors in the top layer of the cell
            """

            # Order existing and candidate indivs by fitness
            candidate_fitnesses = jnp.where(
                batch_of_indices == cell_idx, batch_of_fitnesses, -jnp.inf
            )
            all_fitnesses = jnp.concatenate(
                [cell_fitnesses_depth, candidate_fitnesses],
                axis=0,
            )
            _, final_indices = jax.lax.top_k(all_fitnesses, depth)

            # First, move around existing indivs to follow order
            cell_indices = jnp.where(
                final_indices < depth,
                final_indices,
                out_of_bound,
            )
            new_cell_genotypes_depth = jax.tree_map(
                lambda x: x.at[cell_indices].get(),
                cell_genotypes_depth,
            )
            new_cell_fitnesses_depth = cell_fitnesses_depth.at[cell_indices].get()
            new_cell_fitnesses_depth_all = cell_fitnesses_depth_all.at[
                cell_indices
            ].get()
            new_cell_descriptors_depth = cell_descriptors_depth.at[cell_indices].get()
            new_cell_descriptors_depth_all = cell_descriptors_depth_all.at[
                cell_indices
            ].get()

            # Second, add the candidate indivs
            candidate_indices = jnp.where(
                final_indices >= depth,
                final_indices - depth,
                out_of_bound,
            )
            depth_indices = jnp.where(
                candidate_indices < out_of_bound,
                jnp.arange(0, depth, step=1),
                out_of_bound,
            )
            new_cell_genotypes_depth = jax.tree_map(
                lambda x, y: x.at[depth_indices].set(y[candidate_indices]),
                new_cell_genotypes_depth,
                batch_of_genotypes,
            )
            new_cell_fitnesses_depth = new_cell_fitnesses_depth.at[depth_indices].set(
                batch_of_fitnesses[candidate_indices]
            )
            new_cell_fitnesses_depth_all = new_cell_fitnesses_depth_all.at[
                depth_indices
            ].set(batch_of_all_fitnesses[candidate_indices])
            new_cell_descriptors_depth = new_cell_descriptors_depth.at[
                depth_indices
            ].set(batch_of_descriptors[candidate_indices])
            new_cell_descriptors_depth_all = new_cell_descriptors_depth_all.at[
                depth_indices
            ].set(batch_of_all_descriptors[candidate_indices])

            # Also return the top layer of the grid
            new_cell_genotype = jax.tree_map(
                lambda x: x.at[0].get(),
                new_cell_genotypes_depth,
            )
            new_cell_fitnesses = new_cell_fitnesses_depth.at[0].get()
            new_cell_descriptors = new_cell_descriptors_depth.at[0].get()

            # Return the updated cell
            return (
                new_cell_genotypes_depth,
                new_cell_fitnesses_depth,
                new_cell_fitnesses_depth_all,
                new_cell_descriptors_depth,
                new_cell_descriptors_depth_all,
                new_cell_genotype,
                new_cell_fitnesses,
                new_cell_descriptors,
            )

        _add_per_cell_fn = partial(
            _add_per_cell,
            batch_of_indices=indices,
            batch_of_genotypes=genotypes,
            batch_of_fitnesses=fitnesses,
            batch_of_all_fitnesses=all_fitnesses,
            batch_of_descriptors=descriptors,
            batch_of_all_descriptors=all_descriptors,
        )

        (
            new_genotypes_depth,
            new_fitnesses_depth,
            new_fitnesses_depth_all,
            new_descriptors_depth,
            new_descriptors_depth_all,
            new_genotypes,
            new_fitnesses,
            new_descriptors,
        ) = jax.vmap(_add_per_cell_fn)(
            jnp.arange(0, num_centroids, step=1),
            repertoire.genotypes_depth,
            repertoire.fitnesses_depth,
            repertoire.fitnesses_depth_all,
            repertoire.descriptors_depth,
            repertoire.descriptors_depth_all,
        )

        return repertoire.replace(  # type: ignore
            genotypes_depth=new_genotypes_depth,
            fitnesses_depth=new_fitnesses_depth,
            fitnesses_depth_all=new_fitnesses_depth_all,
            descriptors_depth=new_descriptors_depth,
            descriptors_depth_all=new_descriptors_depth_all,
            genotypes=new_genotypes,
            fitnesses=new_fitnesses,
            descriptors=new_descriptors,
        )

    def add(
        self,
        batch_of_genotypes: Genotype,
        fitness_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        fitness_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        descriptor_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        descriptor_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        scoring_fn: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ],
    ) -> AdaptiveSamplingRepertoire:
        """
        Add a batch of elements to the repertoire.
        WARNING: This addition makes the hypothesis that batch_of_all_descriptors
        and batch_of_all_fitnesses are already dimensions num_evals.

        Args:
            batch_of_genotypes: a batch of genotypes to be added to the repertoire.
                Similarly to the self.genotypes argument, this is a PyTree in which
                the leaves have a shape (batch_size, num_features)

        Returns:
            The updated MAP-Elites repertoire.
        """

        num_centroids = self.fitnesses_depth.shape[0]
        depth = self.fitnesses_depth.shape[1]
        max_elite_evals = self.fitnesses_depth_all.shape[2]
        batch_size = jax.tree_util.tree_leaves(batch_of_genotypes)[0].shape[0]
        out_of_bound = max(
            num_centroids * depth,
            batch_size,
        )

        def _consider_one_indiv(
            repertoire: AdaptiveSamplingRepertoire,
            genotypes: Genotype,
            all_fitnesses: Fitness,
            all_descriptors: Descriptor,
            extra_scores: ExtraScores,
            add_indiv_fn: Callable,
            evals_per_generation: int,
            random_key: RNGKey,
        ) -> Tuple[AdaptiveSamplingRepertoire, int, RNGKey]:
            """
            Direct implementation of the evaluate method in Justesen et al.
            """

            # If reached maximum evaluations, return the unchanged repertoire
            if (evals_per_generation + 1) > self.maximum_evals_per_generation:
                return repertoire, self.maximum_evals_per_generation, random_key

            # New repertoire to store the old one when exceeding evaluations
            new_repertoire = repertoire.replace(
                total_evaluations=repertoire.total_evaluations + 1
            )

            # Acount for initial evaluation of this offspring
            evals_per_generation = evals_per_generation + 1
            initial_evaluation = True

            # Get candidate infos
            fitnesses = fitness_extractor(all_fitnesses)
            fitnesses = jnp.where(jnp.isnan(fitnesses), -jnp.inf, fitnesses)
            descriptors = descriptor_extractor(all_descriptors)
            indices = get_cells_indices(descriptors, new_repertoire.centroids)
            frequency = jnp.sum(jnp.logical_not(jnp.isnan(all_fitnesses)), axis=-1)

            # Get last elite of the cell infos
            frequency_elite = jnp.sum(
                jnp.logical_not(
                    jnp.isnan(new_repertoire.fitnesses_depth_all.at[indices, -1].get())
                ),
                axis=-1,
            )
            fitnesses_elite = new_repertoire.fitnesses_depth.at[indices, -1].get()

            while initial_evaluation or frequency < frequency_elite:

                # Only re-evaluate after the first loop entry
                if initial_evaluation:
                    initial_evaluation = False
                else:

                    # If reached maximum evaluations, return the unchanged repertoire
                    if (evals_per_generation + 1) > self.maximum_evals_per_generation:
                        return (
                            repertoire,
                            self.maximum_evals_per_generation + 1,
                            random_key,
                        )

                    # Evaluate once more
                    (
                        new_fitnesses,
                        new_descriptors,
                        _,
                        random_key,
                    ) = scoring_fn(genotypes, random_key)
                    new_repertoire = new_repertoire.replace(
                        total_evaluations=new_repertoire.total_evaluations + 1
                    )
                    evals_per_generation = evals_per_generation + 1

                    # Concatenate with existing evaluations
                    all_fitnesses = jnp.concatenate(
                        [
                            jnp.expand_dims(new_fitnesses, axis=1),
                            all_fitnesses.at[:, :-1].get(),
                        ],
                        axis=1,
                    )
                    all_descriptors = jnp.concatenate(
                        [
                            jnp.expand_dims(new_descriptors, axis=1),
                            all_descriptors.at[:, :-1].get(),
                        ],
                        axis=1,
                    )

                    # Get candidate infos
                    fitnesses = fitness_extractor(all_fitnesses)
                    fitnesses = jnp.where(jnp.isnan(fitnesses), -jnp.inf, fitnesses)
                    descriptors = descriptor_extractor(all_descriptors)
                    indices = get_cells_indices(descriptors, new_repertoire.centroids)
                    frequency = jnp.sum(
                        jnp.logical_not(jnp.isnan(all_fitnesses)), axis=-1
                    )

                    # Get last elite of the cell infos
                    frequency_elite = jnp.sum(
                        jnp.logical_not(
                            jnp.isnan(
                                new_repertoire.fitnesses_depth_all.at[indices, -1].get()
                            )
                        ),
                        axis=-1,
                    )
                    fitnesses_elite = new_repertoire.fitnesses_depth.at[
                        indices, -1
                    ].get()

                # If there is space in the cell, add
                if frequency_elite == 0:
                    new_repertoire = add_indiv_fn(
                        repertoire=new_repertoire,
                        indices=indices,
                        genotypes=genotypes,
                        fitnesses=fitnesses,
                        all_fitnesses=all_fitnesses,
                        descriptors=descriptors,
                        all_descriptors=all_descriptors,
                        extra_scores=extra_scores,
                    )
                    return new_repertoire, evals_per_generation, random_key

                # If the performance is less than the elite, re-evaluate elites
                # WARNING: add a condition that is not in the original paper here,
                # Not re-evaluating elites if reached the max of elites evaluations,
                # This is to avoid the algorithm getting "stuck" on one cell.
                if fitnesses <= fitnesses_elite and frequency_elite != max_elite_evals:

                    # If reached maximum evaluations, return the unchanged repertoire
                    if (
                        evals_per_generation + depth
                    ) > self.maximum_evals_per_generation:
                        return (
                            repertoire,
                            self.maximum_evals_per_generation + 1,
                            random_key,
                        )

                    # Evaluate all elites in cell once more
                    elites_genotypes = jax.tree_util.tree_map(
                        lambda x: x.at[indices].get().squeeze(axis=0),
                        new_repertoire.genotypes_depth,
                    )
                    (
                        new_elites_fitnesses,
                        new_elites_descriptors,
                        elites_extra_scores,
                        random_key,
                    ) = scoring_fn(elites_genotypes, random_key)
                    new_repertoire = new_repertoire.replace(
                        total_evaluations=new_repertoire.total_evaluations + depth
                    )
                    evals_per_generation = evals_per_generation + depth

                    # Concatenate with existing evaluations of elites
                    elites_all_fitnesses = jnp.concatenate(
                        [
                            jnp.expand_dims(new_elites_fitnesses, axis=1),
                            new_repertoire.fitnesses_depth_all.at[indices, :, :-1]
                            .get()
                            .squeeze(axis=0),
                        ],
                        axis=1,
                    )
                    elites_all_descriptors = jnp.concatenate(
                        [
                            jnp.expand_dims(new_elites_descriptors, axis=1),
                            new_repertoire.descriptors_depth_all.at[indices, :, :-1]
                            .get()
                            .squeeze(axis=0),
                        ],
                        axis=1,
                    )
                    elites_fitnesses = fitness_extractor(elites_all_fitnesses)
                    elites_fitnesses = jnp.where(
                        jnp.isnan(elites_fitnesses), -jnp.inf, elites_fitnesses
                    )
                    elites_descriptors = descriptor_extractor(elites_all_descriptors)
                    elites_indices = get_cells_indices(
                        elites_descriptors, new_repertoire.centroids
                    )

                    # Extract re-evaluated elites from repertoire
                    new_genotypes = jax.tree_util.tree_map(
                        lambda x: x.at[indices].set(0),
                        new_repertoire.genotypes,
                    )
                    new_genotypes_depth = jax.tree_util.tree_map(
                        lambda x: x.at[indices].set(
                            jnp.zeros(
                                (
                                    1,
                                    depth,
                                )
                                + x.shape[2:]
                            )
                        ),
                        new_repertoire.genotypes_depth,
                    )
                    new_fitnesses = new_repertoire.fitnesses.at[indices].set(-jnp.inf)
                    new_fitnesses_depth = new_repertoire.fitnesses_depth.at[
                        indices
                    ].set(
                        -jnp.inf
                        * jnp.ones(
                            (
                                1,
                                depth,
                            )
                        )
                    )
                    new_fitnesses_depth_all = new_repertoire.fitnesses_depth_all.at[
                        indices
                    ].set(
                        -jnp.inf * jnp.ones((1, depth, elites_all_fitnesses.shape[-1]))
                    )
                    new_repertoire = new_repertoire.replace(
                        fitnesses=new_fitnesses,
                        fitnesses_depth=new_fitnesses_depth,
                    )
                    new_descriptors = new_repertoire.descriptors.at[indices].set(
                        jnp.zeros(
                            (
                                1,
                                descriptors.shape[-1],
                            )
                        )
                    )
                    new_descriptors_depth = new_repertoire.descriptors_depth.at[
                        indices
                    ].set(jnp.zeros((1, depth, descriptors.shape[-1])))
                    new_descriptors_depth_all = new_repertoire.descriptors_depth_all.at[
                        indices
                    ].set(
                        jnp.zeros(
                            (
                                1,
                                depth,
                                elites_all_fitnesses.shape[-1],
                                descriptors.shape[-1],
                            )
                        )
                    )
                    new_repertoire = new_repertoire.replace(
                        genotypes=new_genotypes,
                        genotypes_depth=new_genotypes_depth,
                        fitnesses=new_fitnesses,
                        fitnesses_depth=new_fitnesses_depth,
                        fitnesses_depth_all=new_fitnesses_depth_all,
                        descriptors=new_descriptors,
                        descriptors_depth=new_descriptors_depth,
                        descriptors_depth_all=new_descriptors_depth_all,
                    )

                    # Add back all those that belong to the same cell
                    add_elites_indices = jnp.where(
                        elites_indices == indices, elites_indices, out_of_bound
                    )
                    new_repertoire = add_indiv_fn(
                        repertoire=new_repertoire,
                        indices=add_elites_indices,
                        genotypes=elites_genotypes,
                        fitnesses=elites_fitnesses,
                        all_fitnesses=elites_all_fitnesses,
                        descriptors=elites_descriptors,
                        all_descriptors=elites_all_descriptors,
                        extra_scores=elites_extra_scores,
                    )

                    # For the others, move them one by one to new cell
                    for elite in range(depth):
                        if elites_indices.at[elite].get() != indices:
                            (
                                new_repertoire,
                                evals_per_generation,
                                random_key,
                            ) = _consider_one_indiv(
                                repertoire=new_repertoire,
                                genotypes=jax.tree_util.tree_map(
                                    lambda x: x.at[elite : elite + 1].get(),
                                    elites_genotypes,
                                ),
                                all_fitnesses=elites_all_fitnesses.at[
                                    elite : elite + 1
                                ].get(),
                                all_descriptors=elites_all_descriptors.at[
                                    elite : elite + 1
                                ].get(),
                                extra_scores=jax.tree_util.tree_map(
                                    lambda x: x.at[elite : elite + 1].get(),
                                    elites_extra_scores,
                                ),
                                add_indiv_fn=add_indiv_fn,
                                evals_per_generation=evals_per_generation,
                                random_key=random_key,
                            )

                            # If reached maximum evaluations, return the unchanged repertoire
                            if evals_per_generation > self.maximum_evals_per_generation:
                                return (
                                    repertoire,
                                    self.maximum_evals_per_generation + 1,
                                    random_key,
                                )

                # If reached maximum evaluations, return the unchanged repertoire
                if evals_per_generation > self.maximum_evals_per_generation:
                    return repertoire, self.maximum_evals_per_generation + 1, random_key

                # Get last elite of the cell updated infos
                frequency_elite = jnp.sum(
                    jnp.logical_not(
                        jnp.isnan(
                            new_repertoire.fitnesses_depth_all.at[indices, -1].get()
                        )
                    ),
                    axis=-1,
                )
                fitnesses_elite = new_repertoire.fitnesses_depth.at[indices, -1].get()

                # If there is now space in the cell, add
                if frequency_elite == 0:
                    new_repertoire = add_indiv_fn(
                        repertoire=new_repertoire,
                        indices=indices,
                        genotypes=genotypes,
                        fitnesses=fitnesses,
                        all_fitnesses=all_fitnesses,
                        descriptors=descriptors,
                        all_descriptors=all_descriptors,
                        extra_scores=extra_scores,
                    )
                    return new_repertoire, evals_per_generation, random_key

                # If the performance is still less than elites, do not add
                if fitnesses <= fitnesses_elite:
                    return new_repertoire, evals_per_generation, random_key

            # If evaluated the same number of time as the elite and outperforming it, add it
            new_repertoire = add_indiv_fn(
                repertoire=new_repertoire,
                indices=indices,
                genotypes=genotypes,
                fitnesses=fitnesses,
                all_fitnesses=all_fitnesses,
                descriptors=descriptors,
                all_descriptors=all_descriptors,
                extra_scores=extra_scores,
            )
            return new_repertoire, evals_per_generation, random_key

        # Pre-jit functions
        add_indiv_fn = self._add_indiv

        # Do not exceed self.maximum_evals_per_generation
        evals_per_generation = 0

        # Add candidate one by one
        candidate = 0
        random_key = self.random_key
        repertoire = self
        while (
            evals_per_generation < self.maximum_evals_per_generation
            and candidate < batch_size
        ):
            # Initial evaluation
            genotypes = jax.tree_util.tree_map(
                lambda x: x.at[candidate : candidate + 1].get(), batch_of_genotypes
            )
            (
                all_fitnesses,
                all_descriptors,
                extra_scores,
                random_key,
            ) = scoring_fn(genotypes, random_key)

            # Extend to correct shape
            all_fitnesses = jnp.pad(
                jnp.expand_dims(all_fitnesses, axis=1),
                ((0, 0), (0, max_elite_evals - 1)),
                "constant",
                constant_values=jnp.nan,
            )
            all_descriptors = jnp.pad(
                jnp.expand_dims(all_descriptors, axis=1),
                ((0, 0), (0, max_elite_evals - 1), (0, 0)),
                "constant",
                constant_values=jnp.nan,
            )

            # Call consider one indiv
            repertoire, evals_per_generation, random_key = _consider_one_indiv(
                repertoire=repertoire,
                genotypes=genotypes,
                all_fitnesses=all_fitnesses,
                all_descriptors=all_descriptors,
                extra_scores=extra_scores,
                add_indiv_fn=add_indiv_fn,
                evals_per_generation=evals_per_generation,
                random_key=random_key,
            )
            candidate = candidate + 1

        return repertoire.replace(random_key=random_key)  # type: ignore

    @classmethod
    def init(
        cls,
        genotypes: Genotype,
        centroids: Centroid,
        depth: int,
        num_evals: int,
        fitness_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        fitness_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        descriptor_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        descriptor_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        random_key: RNGKey,
        scoring_fn: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ],
        maximum_evals_per_generation: int,
    ) -> AdaptiveSamplingRepertoire:
        """
        Initialize a Map-Elites repertoire with an initial population of genotypes.
        Requires the definition of centroids that can be computed with any method
        such as CVT or Euclidean mapping.

        Note: this function has been kept outside of the object MapElites, so it can
        be called easily called from other modules.

        Args:
            genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            centroids: tesselation centroids of shape (batch_size, num_descriptors)
            depth
            num_evals

        Returns:
            an initialized MAP-Elite repertoire
        """

        # Initialize grid with default values
        num_centroids = centroids.shape[0]
        default_fitnesses = -jnp.inf * jnp.ones(shape=num_centroids)
        default_fitnesses_depth = -jnp.inf * jnp.ones(shape=(num_centroids, depth))
        default_fitnesses_depth_all = jnp.nan * jnp.ones(
            shape=(num_centroids, depth, num_evals)
        )
        default_genotypes = jax.tree_map(
            lambda x: jnp.zeros(shape=(num_centroids,) + x.shape[1:]),
            genotypes,
        )
        default_genotypes_depth = jax.tree_map(
            lambda x: jnp.zeros(
                shape=(
                    num_centroids,
                    depth,
                )
                + x.shape[1:]
            ),
            genotypes,
        )
        default_descriptors = jnp.zeros(shape=(num_centroids, centroids.shape[-1]))
        default_descriptors_depth = jnp.zeros(
            shape=(num_centroids, depth, centroids.shape[-1])
        )
        default_descriptors_depth_all = jnp.nan * jnp.ones(
            shape=(num_centroids, depth, num_evals, centroids.shape[-1])
        )

        repertoire = AdaptiveSamplingRepertoire(
            genotypes=default_genotypes,
            genotypes_depth=default_genotypes_depth,
            fitnesses=default_fitnesses,
            fitnesses_depth=default_fitnesses_depth,
            fitnesses_depth_all=default_fitnesses_depth_all,
            descriptors=default_descriptors,
            descriptors_depth=default_descriptors_depth,
            descriptors_depth_all=default_descriptors_depth_all,
            centroids=centroids,
            random_key=random_key,
            total_evaluations=0,
            maximum_evals_per_generation=maximum_evals_per_generation,
        )

        # Add initial values to the grid
        new_repertoire = repertoire.add(
            batch_of_genotypes=genotypes,
            fitness_extractor=fitness_extractor,
            fitness_reproducibility_extractor=fitness_reproducibility_extractor,
            descriptor_extractor=descriptor_extractor,
            descriptor_reproducibility_extractor=descriptor_reproducibility_extractor,
            scoring_fn=scoring_fn,
        )

        return new_repertoire  # type: ignore

    @jax.jit
    def empty(self) -> AdaptiveSamplingRepertoire:
        """
        Empty the grid from all existing individuals.

        Returns:
            An empty AdaptiveSamplingRepertoire
        """

        new_fitnesses = jnp.full_like(self.fitnesses, -jnp.inf)
        new_fitnesses_depth = jnp.full_like(self.fitnesses_depth, -jnp.inf)
        new_fitnesses_depth_all = jnp.full_like(self.fitnesses_depth_all, jnp.nan)
        new_descriptors = jnp.zeros_like(self.descriptors)
        new_descriptors_depth = jnp.zeros_like(self.descriptors_depth)
        new_descriptors_depth_all = jnp.full_like(self.descriptors_depth_all, jnp.nan)
        new_genotypes = jax.tree_map(lambda x: jnp.zeros_like(x), self.genotypes)
        new_genotypes_depth = jax.tree_map(
            lambda x: jnp.zeros_like(x), self.genotypes_depth
        )
        return AdaptiveSamplingRepertoire(
            genotypes=new_genotypes,
            genotypes_depth=new_genotypes_depth,
            fitnesses=new_fitnesses,
            fitnesses_depth=new_fitnesses_depth,
            fitnesses_depth_all=new_fitnesses_depth_all,
            descriptors=new_descriptors,
            descriptors_depth=new_descriptors_depth,
            descriptors_depth_all=new_descriptors_depth_all,
            centroids=self.centroids,
            random_key=self.random_key,
            total_evaluations=0,
            maximum_evals_per_generation=self.maximum_evals_per_generation,
        )
