from __future__ import annotations

from functools import partial
from typing import Callable, Tuple

import flax
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from qdax.core.containers.mapelites_repertoire import get_cells_indices
from qdax.types import Centroid, Descriptor, ExtraScores, Fitness, Genotype, RNGKey


class DeepMapElitesRepertoire(flax.struct.PyTreeNode):
    """
    Class for the deep repertoire in Map Elites.

    Args:
        genotypes: a PyTree containing the genotypes of the best solutions ordered
            by the centroids. Each leaf has a shape (num_centroids, num_features). The
            PyTree can be a simple Jax array or a more complex nested structure such
            as to represent parameters of neural network in Flax.
        genotypes_depth: a PyTree containing all the genotypes ordered by the centroids.
            Each leaf has a shape (num_centroids * depth, num_features). The PyTree
            can be a simple Jax array or a more complex nested structure such as to
            represent parameters of neural network in Flax.
        fitnesses: an array that contains the fitness of best solutions in each cell of
            the repertoire, ordered by centroids. The array shape is (num_centroids,).
        fitnesses_depth: an array that contains the fitness of all solutions in each
            cell of the repertoire, ordered by centroids. The array shape
            is (num_centroids * depth).
        descriptors: an array that contains the descriptors of best solutions in each
            cell of the repertoire, ordered by centroids. The array shape
            is (num_centroids, num_descriptors).
        descriptors_depth: an array that contains the descriptors of all solutions in
            each cell of the repertoire, ordered by centroids. The array shape
            is (num_centroids * depth, num_descriptors).
        centroids: an array the contains the centroids of the tesselation. The array
            shape is (num_centroids, num_descriptors).
    """

    genotypes: Genotype
    genotypes_depth: Genotype
    fitnesses: Fitness
    fitnesses_depth: Fitness
    descriptors: Descriptor
    descriptors_depth: Descriptor
    centroids: Centroid
    dims: jnp.ndarray

    def save(self, path: str = "./") -> None:
        """Saves the grid on disk in the form of .npy files.

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
        flat_genotypes_depth = jax.vmap(flatten_genotype)(self.genotypes_depth)

        # save data
        jnp.save(path + "genotypes.npy", flat_genotypes)
        jnp.save(path + "genotypes_depth.npy", flat_genotypes_depth)
        jnp.save(path + "fitnesses.npy", self.fitnesses)
        jnp.save(path + "fitnesses_depth.npy", self.fitnesses_depth)
        jnp.save(path + "descriptors.npy", self.descriptors)
        jnp.save(path + "descriptors_depth.npy", self.descriptors_depth)
        jnp.save(path + "centroids.npy", self.centroids)
        jnp.save(path + "dims.npy", self.dims)

    @classmethod
    def load(
        cls, reconstruction_fn: Callable, path: str = "./"
    ) -> DeepMapElitesRepertoire:
        """Loads a MAP Elites Grid.

        Args:
            reconstruction_fn: Function to reconstruct a PyTree
                from a flat array.
            path: Path where the data is saved. Defaults to "./".

        Returns:
            A MAP Elites Repertoire.
        """

        flat_genotypes = jnp.load(path + "genotypes.npy")
        genotypes = jax.vmap(reconstruction_fn)(flat_genotypes)
        flat_genotypes_depth = jnp.load(path + "genotypes_depth.npy")
        genotypes_depth = jax.vmap(reconstruction_fn)(flat_genotypes_depth)

        fitnesses = jnp.load(path + "fitnesses.npy")
        fitnesses_depth = jnp.load(path + "fitnesses_depth.npy")
        descriptors = jnp.load(path + "descriptors.npy")
        descriptors_depth = jnp.load(path + "descriptors_depth.npy")
        centroids = jnp.load(path + "centroids.npy")
        dims = jnp.load(path + "dims.npy")

        return DeepMapElitesRepertoire(
            genotypes=genotypes,
            genotypes_depth=genotypes_depth,
            fitnesses=fitnesses,
            fitnesses_depth=fitnesses_depth,
            descriptors=descriptors,
            descriptors_depth=descriptors_depth,
            centroids=centroids,
            dims=dims,
        )

    @partial(jax.jit, static_argnames=("num_samples",))
    def sample(self, random_key: RNGKey, num_samples: int) -> Tuple[Genotype, RNGKey]:
        """
        Sample elements in the grid. Sample only from the best individuals ("first
        layer of the depth") contained in genotypes, fitnesses and descriptors.

        Args:
            random_key: a jax PRNG random key
            num_samples: the number of elements to be sampled

        Returns:
            samples: a batch of genotypes sampled in the repertoire
            random_key: an updated jax PRNG random key
        """

        random_key, sub_key = jax.random.split(random_key)
        grid_empty = self.fitnesses == -jnp.inf
        p = (1.0 - grid_empty) / jnp.sum(1.0 - grid_empty)

        samples = jax.tree_map(
            lambda x: jax.random.choice(sub_key, x, shape=(num_samples,), p=p),
            self.genotypes,
        )

        return samples, random_key

    @jax.jit
    def _cell_min_fitnesses(
        self,
        cells_indices: jnp.ndarray,
        batch_of_indices: jnp.ndarray,
        batch_of_fitnesses: Fitness,
    ) -> Fitness:
        """
        Sub-method for add(). Give the minimum fitness in each cell of cells_indices,
        given current indivs in cells and new indivs to add in cells.
        !!!WARNING!!! This is strict min fitness and should be used with >, not >=.

        Args:
            cells_indices: the cells to consider
            batch_of_indices: indices of new indivs
            batch_of_fitnesses: fitnesses of new indivs

        Returns: minimum fitness for each cell in cells_indices
        """

        @partial(jax.jit, static_argnames=("depth",))
        def _get_cell_min_fitnesses(
            idx: int,
            fitnesses_depth_reshape: Fitness,
            depth: int,
            batch_of_indices: jnp.ndarray,
            batch_of_fitnesses: Fitness,
        ) -> float:
            """
            Applied using vmap on all cells_indices.
            """
            filter_fitnesses = jnp.where(
                batch_of_indices == idx, batch_of_fitnesses, -jnp.inf
            )
            all_fitnesses = jnp.concatenate(
                [filter_fitnesses, fitnesses_depth_reshape], axis=0
            )
            min_fitnesses, _ = jax.lax.top_k(all_fitnesses, depth + 1)
            return min_fitnesses[depth]  # type: ignore

        get_cell_min_fitnesses_fn = partial(
            _get_cell_min_fitnesses,
            depth=self.dims.shape[0],
            batch_of_indices=batch_of_indices,
            batch_of_fitnesses=batch_of_fitnesses,
        )
        return jax.vmap(get_cell_min_fitnesses_fn)(
            cells_indices,
            jnp.reshape(
                self.fitnesses_depth, (self.centroids.shape[0], self.dims.shape[0])
            )[cells_indices],
        )

    @jax.jit
    def _indices_to_occurence(
        self,
        batch_of_indices: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Sub-method for add(). Return an array similar to the batch_of_indices
        replacing each indice with its occurence number in the batch.

        Args:
            batch_of_indices: indices of new indivs

        Returns: batch_of_occurences: number of occurence for each indice
        """

        @partial(jax.jit, static_argnames=("num_centroids",))
        def _cumulative_count(
            idx: int,
            indices: jnp.ndarray,
            batch_of_indices: jnp.ndarray,
            num_centroids: int,
        ) -> int:
            filter_batch_of_indices = jnp.where(
                indices.ravel() <= idx, batch_of_indices, num_centroids
            )
            count_indices = jnp.bincount(filter_batch_of_indices, length=num_centroids)
            return count_indices.at[batch_of_indices[idx]].get() - 1  # type: ignore

        num_centroids = self.centroids.shape[0]

        # Get occurence
        indices = jnp.arange(0, batch_of_indices.size, step=1)
        cumulative_count = partial(
            _cumulative_count,
            indices=indices,
            batch_of_indices=batch_of_indices,
            num_centroids=num_centroids,
        )
        batch_of_occurence = jax.vmap(cumulative_count)(indices)

        # Filter out-of-bond individuals
        out_of_bound = self.dims.shape[0] * num_centroids
        batch_of_occurence = jnp.where(
            batch_of_indices < out_of_bound, batch_of_occurence, out_of_bound
        )
        return batch_of_occurence

    @jax.jit
    def _place_indivs(
        self,
        batch_of_indices: jnp.ndarray,
        batch_of_fitnesses: Fitness,
    ) -> jnp.ndarray:
        """
        Sub-method for add(). Return indices to place new indiv in the depth grid.

        Args:
            batch_of_indices: indices of new indivs
            batch_of_fitnesses: fitnesses of new indivs

        Returns: indices to place each new indiv
        """

        num_centroids = self.centroids.shape[0]
        depth = self.dims.shape[0]
        out_of_bound = num_centroids * depth  # Index of non-added individuals

        # Get minimum fitness in each cell after addition
        min_fitnesses = self._cell_min_fitnesses(
            jnp.arange(0, num_centroids, step=1),
            batch_of_indices,
            batch_of_fitnesses,
        )

        # Filter individuals and keep those greater than min
        batch_of_indices = jnp.where(
            batch_of_fitnesses > min_fitnesses[batch_of_indices],
            batch_of_indices,
            out_of_bound,
        )

        # Get in-cell indices of individuals
        batch_of_cell_indices = self._indices_to_occurence(batch_of_indices)
        batch_of_cell_indices = jnp.where(
            batch_of_indices < out_of_bound,
            batch_of_indices * depth + batch_of_cell_indices,
            out_of_bound,
        )

        # Filter empty slots using minimum fitness
        @jax.jit
        def _get_empty_slots(
            slots: jnp.ndarray,
            fitness: Fitness,
            min_fitness: Fitness,
            out_of_bound: int,
        ) -> jnp.ndarray:
            return jnp.where(fitness > min_fitness, out_of_bound, slots)

        get_empty_slots = partial(_get_empty_slots, out_of_bound=out_of_bound)
        empty_slots = jax.vmap(get_empty_slots)(
            jnp.reshape(
                jnp.arange(0, num_centroids * depth, step=1), (num_centroids, depth)
            ),
            jnp.reshape(self.fitnesses_depth, (num_centroids, depth)),
            min_fitnesses,
        )

        # Sort the indices in each cell
        empty_slots = jnp.sort(empty_slots, axis=1)

        # Transforms in-cell indices to account for empty slots
        final_batch_of_indices = jnp.where(
            batch_of_cell_indices < out_of_bound,
            empty_slots.ravel()[batch_of_cell_indices],
            out_of_bound,
        )

        return final_batch_of_indices

    @jax.jit
    def add(
        self,
        batch_of_genotypes: Genotype,
        batch_of_descriptors: Descriptor,
        batch_of_fitnesses: Fitness,
        batch_of_extra_scores: ExtraScores,
    ) -> DeepMapElitesRepertoire:
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

        Returns:
            The updated MAP-Elites repertoire.
        """

        out_of_bound = (
            self.dims.shape[0] * self.centroids.shape[0]
        )  # Index of non-added individuals

        # Get indices for given descriptors
        batch_of_indices = get_cells_indices(batch_of_descriptors, self.centroids)

        # Filter dead individuals
        batch_of_indices = jnp.where(
            batch_of_fitnesses > -jnp.inf,
            batch_of_indices,
            out_of_bound,
        )

        # Get final indices of individuals added to top layer of the grid
        # (i.e. best indivs added in: genotypes, fitnesses, descriptors)
        best_fitnesses = jax.ops.segment_max(
            batch_of_fitnesses,
            batch_of_indices,
            num_segments=self.centroids.shape[0],
        )
        filter_fitnesses = jnp.where(
            best_fitnesses[batch_of_indices] == batch_of_fitnesses,
            batch_of_fitnesses,
            -jnp.inf,
        )
        current_fitnesses = jnp.take_along_axis(self.fitnesses, batch_of_indices, 0)
        final_batch_of_max_indices = jnp.where(
            filter_fitnesses > current_fitnesses,
            batch_of_indices,
            out_of_bound,
        )

        # Get final indices of individuals added to the depth of the grid
        # (i.e. indivs in: genotypes_depth, fitnesses_depth, descriptors_depth)
        final_batch_of_indices = self._place_indivs(
            batch_of_indices, batch_of_fitnesses
        )

        # Create new grid
        new_grid_genotypes_depth = jax.tree_map(
            lambda grid_genotypes, new_genotypes: grid_genotypes.at[
                final_batch_of_indices
            ].set(new_genotypes),
            self.genotypes_depth,
            batch_of_genotypes,
        )
        new_grid_genotypes = jax.tree_map(
            lambda grid_genotypes, new_genotypes: grid_genotypes.at[
                final_batch_of_max_indices
            ].set(new_genotypes),
            self.genotypes,
            batch_of_genotypes,
        )

        # Compute new fitness and descriptors
        new_fitnesses = self.fitnesses.at[final_batch_of_max_indices].set(
            batch_of_fitnesses
        )
        new_fitnesses_depth = self.fitnesses_depth.at[final_batch_of_indices].set(
            batch_of_fitnesses
        )
        new_descriptors = self.descriptors.at[final_batch_of_max_indices].set(
            batch_of_descriptors
        )
        new_descriptors_depth = self.descriptors_depth.at[final_batch_of_indices].set(
            batch_of_descriptors
        )

        return DeepMapElitesRepertoire(
            genotypes=new_grid_genotypes,
            genotypes_depth=new_grid_genotypes_depth,
            fitnesses=new_fitnesses.squeeze(),
            fitnesses_depth=new_fitnesses_depth,
            descriptors=new_descriptors.squeeze(),
            descriptors_depth=new_descriptors_depth,
            centroids=self.centroids,
            dims=self.dims,
        )

    @classmethod
    def init(
        cls,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
        centroids: Centroid,
        depth: int,
    ) -> DeepMapElitesRepertoire:
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
            centroids: tesselation centroids of shape (batch_size, num_descriptors)

        Returns:
            an initialized MAP-Elite repertoire
        """

        # Initialize grid with default values
        num_centroids = centroids.shape[0]
        default_fitnesses = -jnp.inf * jnp.ones(shape=num_centroids)
        default_fitnesses_depth = -jnp.inf * jnp.ones(shape=(num_centroids * depth))
        default_genotypes = jax.tree_map(
            lambda x: jnp.zeros(shape=(num_centroids,) + x.shape[1:]),
            genotypes,
        )
        default_genotypes_depth = jax.tree_map(
            lambda x: jnp.zeros(shape=(num_centroids * depth,) + x.shape[1:]),
            genotypes,
        )
        default_descriptors = jnp.zeros(shape=(num_centroids, centroids.shape[-1]))
        default_descriptors_depth = jnp.zeros(
            shape=(num_centroids * depth, centroids.shape[-1])
        )
        dims = jnp.zeros(shape=(depth))

        repertoire = DeepMapElitesRepertoire(
            genotypes=default_genotypes,
            genotypes_depth=default_genotypes_depth,
            fitnesses=default_fitnesses,
            fitnesses_depth=default_fitnesses_depth,
            descriptors=default_descriptors,
            descriptors_depth=default_descriptors_depth,
            centroids=centroids,
            dims=dims,
        )

        # Add initial values to the grid
        new_repertoire = repertoire.add(genotypes, descriptors, fitnesses, extra_scores)

        return new_repertoire  # type: ignore

    @jax.jit
    def empty(self) -> DeepMapElitesRepertoire:
        """
        Empty the grid from all existing individuals.

        Returns:
            An empty DeepMapElitesRepertoire
        """

        new_fitnesses = jnp.full_like(self.fitnesses, -jnp.inf)
        new_fitnesses_depth = jnp.full_like(self.fitnesses_depth, -jnp.inf)
        new_descriptors = jnp.zeros_like(self.descriptors)
        new_descriptors_depth = jnp.zeros_like(self.descriptors_depth)
        new_genotypes = jax.tree_map(lambda x: jnp.zeros_like(x), self.genotypes)
        new_genotypes_depth = jax.tree_map(
            lambda x: jnp.zeros_like(x), self.genotypes_depth
        )
        return DeepMapElitesRepertoire(
            genotypes=new_genotypes,
            genotypes_depth=new_genotypes_depth,
            fitnesses=new_fitnesses,
            fitnesses_depth=new_fitnesses_depth,
            descriptors=new_descriptors,
            descriptors_depth=new_descriptors_depth,
            centroids=self.centroids,
            dims=self.dims,
        )
