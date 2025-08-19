from __future__ import annotations

from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from qdax.custom_types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Mask,
    ParetoFront,
    RNGKey,
)
from qdax.utils.pareto_front import compute_masked_pareto_front

from core.containers.mome_reprod_repertoire import MOMEReprodRepertoire


class MOMEReprodBiasedRepertoire(MOMEReprodRepertoire):
    """Biased version."""

    @jax.jit
    def _sample_in_masked_pareto_front(
        self,
        pareto_front_genotypes: ParetoFront[Genotype],
        pareto_front_fitnesses: ParetoFront[Fitness],
        pareto_front_descriptors: ParetoFront[Descriptor],
        mask: Mask,
        random_key: RNGKey,
    ) -> Tuple[Genotype, Descriptor, Fitness]:
        """Sample one single genotype and corresponding preference in masked pareto front.
        Selection of genotypes is biased according to the crowding distance, calculated
        in objective space.

        Note: do not retrieve a random key because this function
        is to be vmapped. The public method that uses this function
        will return a random key

        Args:
            pareto_front_genotypes: the genotypes of a pareto front
            pareto_front_fitnesses: the fitnesses of a pareto front
            mask: a mask associated to the front
            random_key: a random key to handle stochastic operations

        Returns:
            A single genotype among the pareto front.
        """
        print("\nBiased _sample_in_masked_pareto_front.\n")

        # mask: 1 if empty, 0 if not
        not_empty_mask = 1.0 - mask
        num_solutions = jnp.sum(not_empty_mask)

        num_objective = pareto_front_fitnesses.shape[1]

        def compute_equal_probabilities(
            mask: Mask,
            fitnesses: Fitness,
        ) -> jnp.array:

            equal_probs = (not_empty_mask) / jnp.sum(not_empty_mask)
            return equal_probs

        def compute_crowding_distances(
            mask: Mask,
            pareto_front_fitnesses: Fitness,
        ) -> jnp.array:

            # mask over each objectice
            mask_dist = jnp.column_stack([mask] * pareto_front_fitnesses.shape[1])

            # calculate min and max of fitnesses, ignoring NaNs
            score_amplitude = jnp.nanmax(
                pareto_front_fitnesses * mask_dist, axis=0
            ) - jnp.nanmin(pareto_front_fitnesses * mask_dist, axis=0)

            dist_fitnesses = (
                pareto_front_fitnesses
                + 3
                * score_amplitude
                * jnp.ones_like(pareto_front_fitnesses)
                * mask_dist
            )

            sorted_index = jnp.argsort(dist_fitnesses, axis=0)
            srt_fitnesses = pareto_front_fitnesses[
                sorted_index, jnp.arange(num_objective)
            ]

            # get the distances
            dists = jnp.row_stack(
                [srt_fitnesses, jnp.full(num_objective, jnp.inf)]
            ) - jnp.row_stack([jnp.full(num_objective, -jnp.inf), srt_fitnesses])

            # Prepare the distance to last and next vectors
            dist_to_last = dists[:-1] / score_amplitude
            dist_to_next = dists[1:] / score_amplitude

            # Replace end point distances of inf, with the distance of NN from other side
            dist_to_last_with_ends = jnp.where(
                dist_to_last == jnp.inf, dist_to_next, dist_to_last
            )
            dist_to_next_with_ends = jnp.where(
                dist_to_next == jnp.inf, dist_to_last, dist_to_next
            )

            # Sum up the distances and reorder
            j = jnp.argsort(sorted_index, axis=0)
            crowding_distances = (
                jnp.sum(
                    (
                        dist_to_last_with_ends[j, jnp.arange(num_objective)]
                        + dist_to_next_with_ends[j, jnp.arange(num_objective)]
                    ),
                    axis=1,
                )
                / num_objective
            )
            # Set empty pf cells probability of selection to 0
            crowding_distances = jnp.where(
                jnp.isnan(crowding_distances), 0, crowding_distances
            )

            # Replace crowding distance of inf
            normalised_crowding_distances = crowding_distances / jnp.nansum(
                crowding_distances
            )

            return normalised_crowding_distances

        selection_probs = jax.lax.cond(
            num_solutions <= 2,
            compute_equal_probabilities,
            compute_crowding_distances,
            *(not_empty_mask, pareto_front_fitnesses),
        )

        genotypes_sample = jax.tree_util.tree_map(
            lambda x: jax.random.choice(random_key, x, shape=(1,), p=selection_probs),
            pareto_front_genotypes,
        )

        descriptor_sample = jax.tree_util.tree_map(
            lambda x: jax.random.choice(random_key, x, shape=(1,), p=selection_probs),
            pareto_front_descriptors,
        )

        fitnesses_sample = jax.tree_util.tree_map(
            lambda x: jax.random.choice(random_key, x, shape=(1,), p=selection_probs),
            pareto_front_fitnesses,
        )

        return genotypes_sample, descriptor_sample, fitnesses_sample

    @jax.jit
    def _update_masked_pareto_front(
        self,
        pareto_front_fitnesses: ParetoFront[Fitness],
        pareto_front_genotypes: ParetoFront[Genotype],
        pareto_front_descriptors: ParetoFront[Descriptor],
        mask: Mask,
        new_batch_of_fitnesses: Fitness,
        new_batch_of_genotypes: Genotype,
        new_batch_of_descriptors: Descriptor,
        new_mask: Mask,
    ) -> Tuple[
        ParetoFront[Fitness], ParetoFront[Genotype], ParetoFront[Descriptor], Mask
    ]:
        """Takes a fixed size pareto front, its mask and new points to add.
        Returns updated front and mask.

        Args:
            pareto_front_fitnesses: fitness of the pareto front
            pareto_front_genotypes: corresponding genotypes
            pareto_front_descriptors: corresponding descriptors
            mask: mask of the front, to hide void parts
            new_batch_of_fitnesses: new batch of fitness that is considered
                to be added to the pareto front
            new_batch_of_genotypes: corresponding genotypes
            new_batch_of_descriptors: corresponding descriptors
            new_mask: corresponding mask (no one is masked)

        Returns:
            The updated pareto front.
        """
        print("\nBiased _update_masked_pareto_front.\n")

        # mask: 1 if fitness  is -inf, 0 otherwise
        # get dimensions
        batch_size = new_batch_of_fitnesses.shape[0]
        num_criteria = new_batch_of_fitnesses.shape[1]

        pareto_front_len = pareto_front_fitnesses.shape[0]  # type: ignore
        descriptors_dim = new_batch_of_descriptors.shape[1]

        # gather all data
        cat_mask = jnp.concatenate([mask, new_mask], axis=-1)
        cat_fitnesses = jnp.concatenate(
            [pareto_front_fitnesses, new_batch_of_fitnesses], axis=0
        )
        cat_genotypes = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x, y], axis=0),
            pareto_front_genotypes,
            new_batch_of_genotypes,
        )
        cat_descriptors = jnp.concatenate(
            [pareto_front_descriptors, new_batch_of_descriptors], axis=0
        )

        # get new front
        cat_bool_front = compute_masked_pareto_front(
            batch_of_criteria=cat_fitnesses, mask=cat_mask
        )
        # get corresponding indices
        # have to add 50 to distinguish whether index 0 is part of front or not
        indices = (
            jnp.arange(start=0, stop=pareto_front_len + batch_size) * cat_bool_front
        )
        indices = indices + ~cat_bool_front * (batch_size + pareto_front_len - 1)
        indices = jnp.sort(indices)

        # get fitnesses, genotypes and descriptors of front (with non-front elements all equal to final element)
        all_front_fitnesses = jnp.take(cat_fitnesses, indices, axis=0)
        all_front_genotypes = jax.tree_util.tree_map(
            lambda x: jnp.take(x, indices, axis=0), cat_genotypes
        )
        all_front_descriptors = jnp.take(cat_descriptors, indices, axis=0)

        # compute new mask (which is to be used on all_front_genotypes etc)
        num_front_elements = jnp.sum(cat_bool_front)
        new_mask_indices = jnp.arange(start=0, stop=batch_size + pareto_front_len)
        new_mask_indices = (num_front_elements - new_mask_indices) > 0
        new_front_mask = jnp.where(
            new_mask_indices,
            jnp.ones(shape=batch_size + pareto_front_len, dtype=bool),
            jnp.zeros(shape=batch_size + pareto_front_len, dtype=bool),
        )

        # get fitnesses, descriptors and genotypes on front, with non-front values set to 0/-inf
        fitness_mask = jnp.repeat(
            jnp.expand_dims(new_front_mask, axis=-1), num_criteria, axis=-1
        )

        # set non-front fitnesses = -inf
        all_front_fitnesses = (
            all_front_fitnesses * fitness_mask
            - jnp.inf * jnp.expand_dims(~new_front_mask, axis=-1)
        )
        all_front_fitnesses = jnp.where(
            jnp.isnan(all_front_fitnesses),
            -jnp.inf * jnp.ones_like(all_front_fitnesses),
            all_front_fitnesses,
        )

        all_front_genotypes = jax.tree_util.tree_map(
            lambda x: x * new_mask_indices[0], all_front_genotypes
        )
        descriptors_mask = jnp.repeat(
            jnp.expand_dims(new_front_mask, axis=-1), descriptors_dim, axis=-1
        )
        all_front_descriptors = all_front_descriptors * descriptors_mask

        # reduce pareto front length to max length by removinf solutions with smallest crowding distnace
        empty_mask = jnp.any(all_front_fitnesses == -jnp.inf, axis=-1)

        # create mask over objective dims
        mask_dist = jnp.column_stack([~empty_mask] * all_front_fitnesses.shape[1])

        # calculate min and max of fitnesses, ignoring NaNs
        score_amplitude = jnp.nanmax(
            all_front_fitnesses * mask_dist, axis=0
        ) - jnp.nanmin(all_front_fitnesses * mask_dist, axis=0)

        dist_fitnesses = (
            all_front_fitnesses
            + 3 * score_amplitude * jnp.ones_like(all_front_fitnesses) * mask_dist
        )

        sorted_index = jnp.argsort(dist_fitnesses, axis=0)
        srt_fitnesses = all_front_fitnesses[sorted_index, jnp.arange(num_criteria)]

        # get the distances
        dists = jnp.row_stack(
            [srt_fitnesses, jnp.full(num_criteria, jnp.inf)]
        ) - jnp.row_stack([jnp.full(num_criteria, -jnp.inf), srt_fitnesses])

        # Prepare the distance to last and next vectors
        dist_to_last = dists[:-1] / score_amplitude
        dist_to_next = dists[1:] / score_amplitude

        # Sum up the distances and reorder
        j = jnp.argsort(sorted_index, axis=0)
        crowding_distances = (
            jnp.sum(
                (
                    dist_to_last[j, jnp.arange(num_criteria)]
                    + dist_to_next[j, jnp.arange(num_criteria)]
                ),
                axis=1,
            )
            / num_criteria
        )

        # replace empty distances with -inf
        crowding_distances = jnp.where(
            jnp.isnan(crowding_distances), -jnp.inf, crowding_distances
        )

        # Get indices of smallest crowding distances
        sorted_distances_index = jnp.argsort(crowding_distances)

        # keep solutions with largest crowding distances
        keep_indices = sorted_distances_index[-pareto_front_len:]

        # turn empty fitnesses back to zero
        all_front_fitnesses = jnp.where(
            all_front_fitnesses == -jnp.inf * jnp.ones_like(all_front_fitnesses),
            jnp.zeros_like(all_front_fitnesses),
            all_front_fitnesses,
        )

        # get new pf with only
        new_front_fitnesses = jnp.take(all_front_fitnesses, keep_indices, axis=0)
        new_front_genotypes = jax.tree_util.tree_map(
            lambda x: jnp.take(x, keep_indices, axis=0), all_front_genotypes
        )
        new_front_descriptors = jnp.take(all_front_descriptors, keep_indices, axis=0)
        new_mask = jnp.take(empty_mask, keep_indices)

        return new_front_fitnesses, new_front_genotypes, new_front_descriptors, new_mask

    @classmethod
    def init(
        cls,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        centroids: Centroid,
        extra_scores: ExtraScores,
        fitness_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        fitness_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        descriptor_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        descriptor_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        pareto_front_max_length: int,
    ) -> MOMEReprodBiasedRepertoire:
        """
        Initialize a Map-Elites repertoire with an initial population of genotypes.
        Requires the definition of centroids that can be computed with any method
        such as CVT or Euclidean mapping.

        Note: this function has been kept outside of the object MapElites, so it can
        be called easily called from other modules.

        Args:
            genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            fitnesses: fitness of the initial genotypes of shape (batch_size,)
            descriptors: descriptors of the initial genotypes
                of shape (batch_size, num_descriptors)
            extra_scores: unused extra_scores of the initial genotypes
            centroids: tesselation centroids of shape (batch_size, num_descriptors)
            pareto_front_max_length

        Returns:
            an initialized MAP-Elite repertoire
        """

        # get dimensions
        num_criteria = 2  # always fitness and reproducibility
        num_descriptors = descriptors.shape[-1]
        num_centroids = centroids.shape[0]

        # create default values
        default_fitnesses = -jnp.inf * jnp.ones(
            shape=(num_centroids, pareto_front_max_length, num_criteria)
        )
        default_genotypes = jax.tree_util.tree_map(
            lambda x: jnp.zeros(
                shape=(
                    num_centroids,
                    pareto_front_max_length,
                )
                + x.shape[1:]
            ),
            genotypes,
        )
        default_descriptors = jnp.zeros(
            shape=(num_centroids, pareto_front_max_length, num_descriptors)
        )

        # create repertoire with default values
        repertoire = MOMEReprodBiasedRepertoire(  # type: ignore
            genotypes=default_genotypes,
            fitnesses=default_fitnesses,
            descriptors=default_descriptors,
            centroids=centroids,
        )

        # add first batch of individuals in the repertoire
        new_repertoire = repertoire.add(
            batch_of_genotypes=genotypes,
            batch_of_all_descriptors=descriptors,
            batch_of_all_fitnesses=fitnesses,
            batch_of_extra_scores=extra_scores,
            fitness_extractor=fitness_extractor,
            fitness_reproducibility_extractor=fitness_reproducibility_extractor,
            descriptor_extractor=descriptor_extractor,
            descriptor_reproducibility_extractor=descriptor_reproducibility_extractor,
        )

        return new_repertoire  # type: ignore

    @jax.jit
    def empty(self) -> MOMEReprodBiasedRepertoire:
        """
        Empty the grid from all existing individuals.

        Returns:
            An empty repertoire
        """

        new_fitnesses = jnp.full_like(self.fitnesses, -jnp.inf)
        new_descriptors = jnp.zeros_like(self.descriptors)
        new_genotypes = jax.tree_map(lambda x: jnp.zeros_like(x), self.genotypes)
        return MOMEReprodBiasedRepertoire(
            genotypes=new_genotypes,
            fitnesses=new_fitnesses,
            descriptors=new_descriptors,
            centroids=self.centroids,
        )
