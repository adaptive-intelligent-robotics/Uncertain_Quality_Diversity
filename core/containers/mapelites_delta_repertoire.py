from __future__ import annotations

from functools import partial
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    get_cells_indices,
)
from qdax.custom_types import Centroid, Descriptor, ExtraScores, Fitness, Genotype


class MapElitesDeltaRepertoire(MapElitesRepertoire):
    """Class for the repertoire in Map Elites when considering the
    fitness-reproducibility trade-off problem.
    """

    reproducibilities: jnp.ndarray
    max_reproducibility: jnp.ndarray
    max_fitness: jnp.ndarray

    def save(self, path: str = "./") -> None:
        """Saves the repertoire on disk in the form of .npy files.

        Flattens the genotypes to store it with .npy format. Supposes that
        a user will have access to the reconstruction function when loading
        the genotypes.

        Args:
            path: Path where the data will be saved. Defaults to "./".
        """

        def flatten_genotype(genotype: Genotype) -> jnp.ndarray:
            flatten_genotype, _ = ravel_pytree(genotype)
            return flatten_genotype

        # flatten all the genotypes
        flat_genotypes = jax.vmap(flatten_genotype)(self.genotypes)

        # save data
        jnp.save(path + "genotypes.npy", flat_genotypes)
        jnp.save(path + "fitnesses.npy", self.fitnesses)
        jnp.save(path + "reproducibilities.npy", self.reproducibilities)
        jnp.save(path + "max_reproducibility.npy", self.max_reproducibility)
        jnp.save(path + "max_fitness.npy", self.max_fitness)
        jnp.save(path + "descriptors.npy", self.descriptors)
        jnp.save(path + "centroids.npy", self.centroids)

    @classmethod
    def load(
        cls, reconstruction_fn: Callable, path: str = "./"
    ) -> MapElitesDeltaRepertoire:
        """Loads a MAP Elites Repertoire.

        Args:
            reconstruction_fn: Function to reconstruct a PyTree
                from a flat array.
            path: Path where the data is saved. Defaults to "./".

        Returns:
            A MAP Elites Repertoire.
        """

        flat_genotypes = jnp.load(path + "genotypes.npy")
        genotypes = jax.vmap(reconstruction_fn)(flat_genotypes)

        fitnesses = jnp.load(path + "fitnesses.npy")
        reproducibilities = jnp.load(path + "reproducibilities.npy")
        max_reproducibility = jnp.load(path + "max_reproducibility.npy")
        max_fitness = jnp.load(path + "max_fitness.npy")
        descriptors = jnp.load(path + "descriptors.npy")
        centroids = jnp.load(path + "centroids.npy")

        return cls(
            genotypes=genotypes,
            fitnesses=fitnesses,
            reproducibilities=reproducibilities,
            max_reproducibility=max_reproducibility,
            max_fitness=max_fitness,
            descriptors=descriptors,
            centroids=centroids,
        )

    @partial(
        jax.jit,
        static_argnames=(
            "fitness_extractor",
            "fitness_reproducibility_extractor",
            "descriptor_extractor",
            "descriptor_reproducibility_extractor",
            "delta_fitness",
            "delta_reproducibility",
        ),
    )
    def add(
        self,
        batch_of_genotypes: Genotype,
        batch_of_descriptors: Descriptor,
        batch_of_fitnesses: Fitness,
        batch_of_extra_scores: ExtraScores,
        fitness_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        fitness_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        descriptor_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        descriptor_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        delta_fitness: float,
        delta_reproducibility: float,
    ) -> MapElitesDeltaRepertoire:
        """
        Add a batch of elements to the repertoire.

        Args:
            batch_of_genotypes: a batch of genotypes to be added to the repertoire.
                Similarly to the self.genotypes argument, this is a PyTree in which
                the leaves have a shape (batch_size, num_features)
            batch_of_descriptors: an array that contains the descriptors of the
                aforementioned genotypes. Its shape is (batch_size, num_descriptors)
            batch_of_fitnesses: an array that contains the fitnesses of the
            batch_of_extra_scores: unused tree that contains the extra_scores of
                aforementioned genotypes. Its shape is (batch_size,)
            batch_of_extra_scores: unused tree that contains the extra_scores of
                aforementioned genotypes.

        Returns:
            The updated MAP-Elites repertoire.
        """
        batch_size, num_samples = batch_of_fitnesses.shape
        num_centroids = self.centroids.shape[0]

        # Compute batch of reproducibility
        # WARNING: in the case of descriptors_reproducibility, take average over dimensions
        batch_of_reproducibilities = descriptor_reproducibility_extractor(
            batch_of_descriptors
        )
        batch_of_reproducibilities = jnp.average(batch_of_reproducibilities, axis=-1)
        batch_of_reproducibilities = jnp.expand_dims(
            batch_of_reproducibilities, axis=-1
        )

        # WARNING: takes the negative as the extractor is actually returning the inverse of reproducibility
        batch_of_reproducibilities = -batch_of_reproducibilities

        # Compute batch of descriptor
        batch_of_descriptors = descriptor_extractor(batch_of_descriptors)

        # Compute batch of fitness
        batch_of_fitnesses = jnp.expand_dims(
            fitness_extractor(batch_of_fitnesses), axis=-1
        )

        # Compute indices
        batch_of_indices = get_cells_indices(batch_of_descriptors, self.centroids)
        batch_of_indices = jnp.expand_dims(batch_of_indices, axis=-1)

        # Compute new maximum reproducibility and fitness
        new_max_reproducibility = jnp.max(batch_of_reproducibilities)
        new_max_reproducibility = jnp.maximum(
            new_max_reproducibility, self.max_reproducibility
        )
        new_max_fitness = jnp.max(batch_of_fitnesses)
        new_max_fitness = jnp.maximum(new_max_fitness, self.max_fitness)
        new_repertoire = self.replace(
            max_reproducibility=new_max_reproducibility,
            max_fitness=new_max_fitness,
        )

        # Add individuals one by one
        def _add_one(
            carry: MapElitesDeltaRepertoire,
            data: Tuple[Genotype, Descriptor, Fitness, jnp.ndarray, jnp.ndarray],
        ) -> Tuple[MapElitesDeltaRepertoire, Any]:

            # unwrap data
            genotype, descriptor, fitness, reproducibility, index = data

            # get cell values
            cell_fitness = carry.fitnesses[index]
            cell_reproducibility = carry.reproducibilities[index]

            # all criterions
            fitness_only_condition = fitness > (cell_fitness + delta_fitness)
            fitness_condition = jnp.logical_and(
                fitness >= cell_fitness,
                reproducibility >= cell_reproducibility,
            )
            reproducibility_condition = jnp.logical_and(
                fitness > (cell_fitness - delta_fitness),
                reproducibility >= (cell_reproducibility + delta_reproducibility),
            )

            # final addition condition
            final_condition = jnp.logical_or(
                fitness_only_condition,
                jnp.logical_or(fitness_condition, reproducibility_condition),
            )

            index = jnp.where(final_condition, index, num_centroids)

            # update grid
            new_genotypes = jax.tree_util.tree_map(
                lambda x, y: x.at[index].set(y), carry.genotypes, genotype
            )
            new_fitnesses = carry.fitnesses.at[index].set(fitness)
            new_descriptors = carry.descriptors.at[index].set(descriptor)
            new_reproducibilities = carry.reproducibilities.at[index].set(
                reproducibility
            )

            carry = carry.replace(  # type: ignore
                genotypes=new_genotypes,
                descriptors=new_descriptors,
                fitnesses=new_fitnesses,
                reproducibilities=new_reproducibilities,
            )

            # return new grid
            return carry, ()

        # scan the addition operation for all the individuals
        new_repertoire, _ = jax.lax.scan(
            _add_one,
            new_repertoire,
            (
                batch_of_genotypes,
                batch_of_descriptors,
                batch_of_fitnesses,
                batch_of_reproducibilities,
                batch_of_indices,
            ),
        )

        return new_repertoire  # type: ignore

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
        delta_fitness: float,
        delta_reproducibility: float,
    ) -> MapElitesRepertoire:
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
            centroids: tesselation centroids of shape (batch_size, num_descriptors)
            extra_scores: unused extra_scores of the initial genotypes

        Returns:
            an initialized MAP-Elite repertoire
        """

        # retrieve one genotype from the population
        first_genotype = jax.tree_util.tree_map(lambda x: x[0], genotypes)

        # create a repertoire with default values
        repertoire = cls.init_default(genotype=first_genotype, centroids=centroids)

        # add initial population to the repertoire
        new_repertoire = repertoire.add(
            batch_of_genotypes=genotypes,
            batch_of_descriptors=descriptors,
            batch_of_fitnesses=fitnesses,
            batch_of_extra_scores=extra_scores,
            fitness_extractor=fitness_extractor,
            fitness_reproducibility_extractor=fitness_reproducibility_extractor,
            descriptor_extractor=descriptor_extractor,
            descriptor_reproducibility_extractor=descriptor_reproducibility_extractor,
            delta_fitness=delta_fitness,
            delta_reproducibility=delta_reproducibility,
        )

        return new_repertoire  # type: ignore

    @classmethod
    def init_default(
        cls,
        genotype: Genotype,
        centroids: Centroid,
    ) -> MapElitesDeltaRepertoire:
        """Initialize a Map-Elites repertoire with an initial population of
        genotypes. Requires the definition of centroids that can be computed
        with any method such as CVT or Euclidean mapping.

        Note: this function has been kept outside of the object MapElites, so
        it can be called easily called from other modules.

        Args:
            genotype: the typical genotype that will be stored.
            centroids: the centroids of the repertoire

        Returns:
            A repertoire filled with default values.
        """

        # get number of centroids
        num_centroids = centroids.shape[0]

        # default fitness is -inf
        default_fitnesses = -jnp.inf * jnp.ones(shape=num_centroids)

        # default reproducibility is -inf
        default_reproducibilities = -jnp.inf * jnp.ones(shape=num_centroids)
        default_max_reproducibility = -jnp.inf * jnp.ones((1))
        default_max_fitness = -jnp.inf * jnp.ones((1))

        # default genotypes is all 0
        default_genotypes = jax.tree_util.tree_map(
            lambda x: jnp.zeros(shape=(num_centroids,) + x.shape, dtype=x.dtype),
            genotype,
        )

        # default descriptor is all zeros
        default_descriptors = jnp.zeros_like(centroids)

        return cls(
            genotypes=default_genotypes,
            fitnesses=default_fitnesses,
            reproducibilities=default_reproducibilities,
            max_reproducibility=default_max_reproducibility,
            max_fitness=default_max_fitness,
            descriptors=default_descriptors,
            centroids=centroids,
        )

    @jax.jit
    def empty(self) -> MapElitesDeltaRepertoire:
        """
        Empty the grid from all existing individuals.

        Returns:
            An empty MapElitesDeltaRepertoire
        """

        new_fitnesses = jnp.full_like(self.fitnesses, -jnp.inf)
        new_descriptors = jnp.zeros_like(self.descriptors)
        new_reproducibilities = jnp.full_like(self.reproducibilities, -jnp.inf)
        new_max_reproducibility = -jnp.inf * jnp.ones_like(self.max_reproducibility)
        new_max_fitness = -jnp.inf * jnp.ones_like(self.max_fitness)
        new_genotypes = jax.tree_map(lambda x: jnp.zeros_like(x), self.genotypes)
        return MapElitesDeltaRepertoire(
            genotypes=new_genotypes,
            fitnesses=new_fitnesses,
            reproducibilities=new_reproducibilities,
            max_reproducibility=new_max_reproducibility,
            max_fitness=new_max_fitness,
            descriptors=new_descriptors,
            centroids=self.centroids,
        )
