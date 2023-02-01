from __future__ import annotations

from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from qdax.core.containers.mapelites_repertoire import get_cells_indices
from qdax.types import Centroid, Descriptor, ExtraScores, Fitness, Genotype, RNGKey

from core.containers.depth_repertoire import DeepMapElitesRepertoire


class DeepGridRepertoire(DeepMapElitesRepertoire):
    """
    Class for the repertoire of deep-grid.
    Redefine the selection and addition from DeepMapElitesRepertoire.
    """

    @classmethod
    def load(cls, reconstruction_fn: Callable, path: str = "./") -> DeepGridRepertoire:
        """Loads a MAP Elites Grid.

        Args:
            reconstruction_fn: Function to reconstruct a PyTree
                from a flat array.
            path: Path where the data is saved. Defaults to "./".

        Returns:
            A MAP Elites Repertoire.
        """

        repertoire = super().load(reconstruction_fn=reconstruction_fn, path=path)

        return DeepGridRepertoire(
            genotypes=repertoire.genotypes,
            genotypes_depth=repertoire.genotypes_depth,
            fitnesses=repertoire.fitnesses,
            fitnesses_depth=repertoire.fitnesses_depth,
            descriptors=repertoire.descriptors,
            descriptors_depth=repertoire.descriptors_depth,
            centroids=repertoire.centroids,
            dims=repertoire.dims,
        )

    @jax.jit
    def _in_cell_sample(
        self,
        idx: jnp.ndarray,
        cells: jnp.ndarray,
        random_keys: RNGKey,
        fitnesses_depth_reshape: Fitness,
    ) -> jnp.ndarray:
        """
        Return the index of an indiv of the cell, fitness proportionally. Applied
        using apply_along_axis on all considered cells.
        """
        cell_fitnesses = fitnesses_depth_reshape[cells[idx[0]]]
        cell_empty = cell_fitnesses == -jnp.inf

        # Normalise all fitnesses in cell
        max_fitness = jnp.max(cell_fitnesses)
        min_fitness = jnp.min(jnp.where(cell_empty, jnp.inf, cell_fitnesses))
        norm_fitnesses = jnp.where(
            cell_empty,
            0,
            (cell_fitnesses - min_fitness) / (max_fitness - min_fitness),
        )

        # Compute probabilities
        p = norm_fitnesses / jnp.sum(norm_fitnesses)

        # Build alternative probability for extreme case
        # (i.e. only 1 indiv in cell that is both min and max fitness)
        p_alt = (1.0 - cell_empty) / jnp.sum(1.0 - cell_empty)

        # Sample taking into account extreme case (p == p fitlers nan)
        indices_indivs = jnp.arange(0, cell_fitnesses.shape[0], step=1)
        final_indices = jnp.where(
            p == p,
            jax.random.choice(random_keys[idx[0]], indices_indivs, shape=(1,), p=p),
            jax.random.choice(random_keys[idx[0]], indices_indivs, shape=(1,), p=p_alt),
        )[idx[0]]
        return final_indices

    @partial(jax.jit, static_argnames=("num_samples",))
    def sample(self, random_key: RNGKey, num_samples: int) -> Tuple[Genotype, RNGKey]:
        """
        Sample elements in the grid.
        For each sample, choose a random cell and return a random individual of this
        cell fitness-proportionally.

        Args:
            random_key: a jax PRNG random key
            num_samples: the number of elements to be sampled

        Returns:
            samples: a batch of genotypes sampled in the repertoire
            random_key: an updated jax PRNG random key
        """

        # Split keys
        random_key, sub_key = jax.random.split(random_key)
        cell_key, indiv_key = jax.random.split(sub_key)  # 1 key per cell sample
        indivs_keys = jax.random.split(
            indiv_key, num=num_samples
        )  # 1 key per in-cell sample

        # Pick num_samples random non-empty cell
        grid_empty = self.fitnesses == -jnp.inf
        p = (1.0 - grid_empty) / jnp.sum(1.0 - grid_empty)
        indices_cells = jnp.arange(0, self.centroids.shape[0], step=1)
        sample_cells = jax.random.choice(
            cell_key, indices_cells, shape=(num_samples,), p=p
        )

        # Pick one indiv per sampled cell fitness-proportionaly
        in_cell_sample = partial(
            self._in_cell_sample,
            cells=sample_cells,
            random_keys=indivs_keys,
            fitnesses_depth_reshape=jnp.reshape(
                self.fitnesses_depth, (self.centroids.shape[0], self.dims.shape[0])
            ),
        )
        sample_indivs = jnp.apply_along_axis(
            in_cell_sample,
            1,
            jnp.transpose(jnp.expand_dims(jnp.arange(0, num_samples, step=1), axis=0)),
        )
        sample_indivs = sample_indivs.ravel()

        # Get corresponding genotypes
        sample_indices = sample_cells * self.dims.shape[0] + sample_indivs
        samples = jax.tree_map(
            lambda x: x[sample_indices],
            self.genotypes_depth,
        )

        return samples, random_key

    @partial(jax.jit, static_argnames=("num_samples",))
    def sample_all_cells(
        self, random_key: RNGKey, num_samples: int
    ) -> Tuple[Genotype, Fitness, Descriptor, RNGKey]:
        """
        Sample num_samples indivs from each cells using in-cell selector.
        Args:
           random_key
           num_samples: number of samples per cell.
        Returns:
            genotypes: genotypes of samples indivs.
            fitnesses: associated fitnesses.
            descriptor: associated descriptors.
            random_key
        """

        # Split keys (1 per in-cell samples)
        random_key, sub_key = jax.random.split(random_key)
        total_indivs = self.centroids.shape[0] * num_samples
        indivs_keys = jax.random.split(sub_key, num=total_indivs)

        # Pick num_samples indivs per cell fitness-proportionaly
        cells = jnp.repeat(
            jnp.arange(0, self.centroids.shape[0], step=1), num_samples, axis=0
        )
        in_cell_sample = partial(
            self._in_cell_sample,
            cells=cells,
            random_keys=indivs_keys,
            fitnesses_depth_reshape=jnp.reshape(
                self.fitnesses_depth, (self.centroids.shape[0], self.dims.shape[0])
            ),
        )
        sample_indivs = jnp.apply_along_axis(
            in_cell_sample,
            1,
            jnp.transpose(jnp.expand_dims(jnp.arange(0, total_indivs, step=1), axis=0)),
        )
        sample_indivs = sample_indivs.ravel()

        # Get corresponding genotypes, fitnesses and descriptors
        sample_indices = cells * self.dims.shape[0] + sample_indivs
        genotypes = jax.tree_map(
            lambda x: x.at[sample_indices].get(),
            self.genotypes_depth,
        )
        fitnesses = self.fitnesses_depth.at[sample_indices].get()
        descriptors = self.descriptors_depth.at[sample_indices].get()

        return genotypes, fitnesses, descriptors, random_key

    @jax.jit
    def _place_indivs(
        self,
        batch_of_indices: jnp.ndarray,
        random_key: RNGKey,
    ) -> Tuple[jnp.ndarray, RNGKey]:
        """
        Sub-method for add(). Return indices to place each new indiv in the grid.
        WARNING: batch_of_indices should have already been filtered to contain
        only indivs that should be added.

        Args:
            batch_of_indices: indices of new indivs
            random_key

        Returns: indices to place each new indiv and a new random_key
        """

        random_key, sub_key = jax.random.split(random_key)

        num_centroids = self.centroids.shape[0]
        depth = self.dims.shape[0]
        out_of_bound = num_centroids * depth  # Index of non-added individuals

        # Add a maximum of depth individuals per cell
        batch_of_occurence = self._indices_to_occurence(batch_of_indices)
        batch_of_indices = jnp.where(
            batch_of_occurence < depth,
            batch_of_indices,
            out_of_bound,
        )

        # Get in-cell indices of individuals
        batch_of_cell_indices = jnp.where(
            batch_of_indices < out_of_bound,
            batch_of_indices * depth + batch_of_occurence,
            out_of_bound,
        )

        # Choose slots to fill (empty first, then random)
        @partial(jax.jit, static_argnames=("depth",))
        def _get_slots(
            slots: jnp.ndarray,
            fitness: Fitness,
            key: RNGKey,
            depth: int,
        ) -> jnp.ndarray:
            slots_number = jnp.where(
                fitness == -jnp.inf, 0, jax.random.randint(key, (depth,), 0, depth)
            )
            return slots[jnp.argsort(slots_number)]

        sorted_slots = jax.vmap(partial(_get_slots, depth=self.dims.shape[0]))(
            jnp.reshape(
                jnp.arange(0, num_centroids * depth, step=1), (num_centroids, depth)
            ),
            jnp.reshape(self.fitnesses_depth, (num_centroids, depth)),
            jax.random.split(sub_key, num=num_centroids),
        ).ravel()

        # Place individuals
        final_batch_of_indices = jnp.where(
            batch_of_cell_indices < out_of_bound,
            sorted_slots[batch_of_cell_indices],
            out_of_bound,
        )

        return final_batch_of_indices, random_key

    @jax.jit
    def add(
        self,
        batch_of_genotypes: Genotype,
        batch_of_descriptors: Descriptor,
        batch_of_fitnesses: Fitness,
        batch_of_extra_scores: ExtraScores,
        random_key: RNGKey,
    ) -> Tuple[DeepGridRepertoire, RNGKey]:
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
            random_key

        Returns:
            The updated MAP-Elites repertoire.
        """

        num_centroids = self.centroids.shape[0]
        depth = self.dims.shape[0]
        out_of_bound = num_centroids * depth  # Index of non-added individuals

        # Get indices for given descriptors
        batch_of_indices = get_cells_indices(batch_of_descriptors, self.centroids)

        # Filter dead individuals
        batch_of_indices = jnp.where(
            batch_of_fitnesses > -jnp.inf, batch_of_indices, out_of_bound
        )

        # Get final indices of individuals added to the depth of the grid
        # (i.e. indivs in: genotypes_depth, fitnesses_depth, descriptors_depth)
        final_batch_of_indices, random_key = self._place_indivs(
            batch_of_indices, random_key
        )

        # Create new depth grid, fitness and descriptors
        new_grid_genotypes_depth = jax.tree_map(
            lambda grid_genotypes, new_genotypes: grid_genotypes.at[
                final_batch_of_indices
            ].set(new_genotypes),
            self.genotypes_depth,
            batch_of_genotypes,
        )
        new_fitnesses_depth = self.fitnesses_depth.at[final_batch_of_indices].set(
            batch_of_fitnesses
        )
        new_descriptors_depth = self.descriptors_depth.at[final_batch_of_indices].set(
            batch_of_descriptors
        )

        # Create top layer of the grid from depth
        # (i.e. best indivs added in: genotypes, fitnesses, descriptors)
        new_fitnesses_depth_reshape = jnp.reshape(
            new_fitnesses_depth, (num_centroids, depth)
        )
        final_batch_of_max_indices = jnp.arange(0, num_centroids, step=1)
        depth_indices = jnp.nanargmax(new_fitnesses_depth_reshape, axis=1)
        depth_indices = final_batch_of_max_indices * depth + depth_indices
        final_batch_of_max_indices = jnp.where(
            new_fitnesses_depth[depth_indices] == -jnp.inf,
            out_of_bound,
            final_batch_of_max_indices,
        )

        # Create new grid, fitness and descriptors
        new_fitnesses = self.fitnesses.at[final_batch_of_max_indices].set(
            new_fitnesses_depth[depth_indices]
        )
        new_descriptors = self.descriptors.at[final_batch_of_max_indices].set(
            new_descriptors_depth[depth_indices]
        )

        new_grid_genotypes = jax.tree_map(
            lambda genotypes, genotypes_depth: genotypes.at[
                final_batch_of_max_indices
            ].set(genotypes_depth[depth_indices]),
            self.genotypes,
            new_grid_genotypes_depth,
        )

        return (
            DeepGridRepertoire(
                genotypes=new_grid_genotypes,
                genotypes_depth=new_grid_genotypes_depth,
                fitnesses=new_fitnesses.squeeze(),
                fitnesses_depth=new_fitnesses_depth,
                descriptors=new_descriptors.squeeze(),
                descriptors_depth=new_descriptors_depth,
                centroids=self.centroids,
                dims=self.dims,
            ),
            random_key,
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
        random_key: RNGKey,
    ) -> Tuple[DeepGridRepertoire, RNGKey]:
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

        repertoire = super().init(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
            centroids=centroids,
            depth=depth,
        )
        repertoire = DeepGridRepertoire(
            genotypes=repertoire.genotypes,
            genotypes_depth=repertoire.genotypes_depth,
            fitnesses=repertoire.fitnesses,
            fitnesses_depth=repertoire.fitnesses_depth,
            descriptors=repertoire.descriptors,
            descriptors_depth=repertoire.descriptors_depth,
            centroids=repertoire.centroids,
            dims=repertoire.dims,
        )

        # Add initial values to the grid
        new_repertoire, random_key = repertoire.add(
            genotypes, descriptors, fitnesses, extra_scores, random_key
        )

        return new_repertoire, random_key

    @jax.jit
    def empty(self) -> DeepGridRepertoire:
        """
        Empty the grid from all existing individuals.

        Returns:
            An empty DeepGridRepertoire
        """

        repertoire = super().empty()
        return DeepGridRepertoire(
            genotypes=repertoire.genotypes,
            genotypes_depth=repertoire.genotypes_depth,
            fitnesses=repertoire.fitnesses,
            fitnesses_depth=repertoire.fitnesses_depth,
            descriptors=repertoire.descriptors,
            descriptors_depth=repertoire.descriptors_depth,
            centroids=repertoire.centroids,
            dims=repertoire.dims,
        )
