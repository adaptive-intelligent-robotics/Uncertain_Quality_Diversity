from __future__ import annotations

from functools import partial
from typing import Callable, Tuple

import flax
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from qdax.core.containers.mapelites_repertoire import get_cells_indices
from qdax.types import Centroid, Descriptor, ExtraScores, Fitness, Genotype, RNGKey


class ArchiveSamplingRepertoire(flax.struct.PyTreeNode):
    """
    Class for a deep repertoire that also stores all past evaluations of each indiv.

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
        fitnesses_depth_all: an array that contains the fitness of all solutions in each
            cell of the repertoire, ordered by centroids. The array shape
            is (num_centroids * depth, num_evals).
        descriptors: an array that contains the descriptors of best solutions in each
            cell of the repertoire, ordered by centroids. The array shape
            is (num_centroids, num_descriptors).
        descriptors_depth_all: an array that contains the descriptors of all solutions in
            each cell of the repertoire, ordered by centroids. The array shape
            is (num_centroids * depth, num_evals, num_descriptors).
        evaluations_depth: an array that contains the number of evaluations of
            individuals in the grid.
        total_evaluations: the total number of evaluations spent.
        centroids: an array the contains the centroids of the tesselation. The array
            shape is (num_centroids, num_descriptors).
        dims
    """

    genotypes: Genotype
    genotypes_depth: Genotype
    fitnesses: Fitness
    fitnesses_depth: Fitness
    fitnesses_depth_all: Fitness
    descriptors: Descriptor
    descriptors_depth_all: Descriptor
    evaluations_depth: jnp.ndarray
    total_evaluations: int
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
        jnp.save(path + "fitnesses_depth_all.npy", self.fitnesses_depth_all)
        jnp.save(path + "descriptors.npy", self.descriptors)
        jnp.save(path + "descriptors_depth_all.npy", self.descriptors_depth_all)
        jnp.save(path + "evaluations_depth.npy", self.evaluations_depth)
        jnp.save(path + "total_evaluations.npy", self.total_evaluations)
        jnp.save(path + "centroids.npy", self.centroids)
        jnp.save(path + "dims.npy", self.dims)

    @classmethod
    def load(
        cls, reconstruction_fn: Callable, path: str = "./"
    ) -> ArchiveSamplingRepertoire:
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
        fitnesses_depth_all = jnp.load(path + "fitnesses_depth_all.npy")
        descriptors = jnp.load(path + "descriptors.npy")
        descriptors_depth_all = jnp.load(path + "descriptors_depth_all.npy")
        centroids = jnp.load(path + "centroids.npy")
        evaluations_depth = jnp.load(path + "evaluations_depth.npy")
        total_evaluations = jnp.load(path + "total_evaluations.npy")
        dims = jnp.load(path + "dims.npy")

        return ArchiveSamplingRepertoire(
            genotypes=genotypes,
            genotypes_depth=genotypes_depth,
            fitnesses=fitnesses,
            fitnesses_depth=fitnesses_depth,
            fitnesses_depth_all=fitnesses_depth_all,
            descriptors=descriptors,
            descriptors_depth_all=descriptors_depth_all,
            evaluations_depth=evaluations_depth,
            total_evaluations=total_evaluations,
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

    @partial(jax.jit, static_argnames=("num_samples",))
    def sample_with_descs(
        self, random_key: RNGKey, num_samples: int
    ) -> Tuple[Genotype, Descriptor, RNGKey]:
        """Sample elements in the repertoire and return both their
        genotypes, descriptors and fitnesses.

        Args:
            random_key: a jax PRNG random key
            num_samples: the number of elements to be sampled

        Returns:
            samples: a batch of genotypes sampled in the repertoire
            descriptors: the corresponding descriptors
            random_key: an updated jax PRNG random key
        """

        repertoire_empty = self.fitnesses == -jnp.inf
        p = (1.0 - repertoire_empty) / jnp.sum(1.0 - repertoire_empty)

        random_key, subkey = jax.random.split(random_key)
        samples = jax.tree_util.tree_map(
            lambda x: jax.random.choice(subkey, x, shape=(num_samples,), p=p),
            self.genotypes,
        )
        descriptors = jax.random.choice(
            subkey, self.descriptors, shape=(num_samples,), p=p
        )

        return samples, descriptors, random_key

    @jax.jit
    def _cell_min_comparison_metrics(
        self,
        cells_indices: jnp.ndarray,
        batch_of_indices: jnp.ndarray,
        batch_of_comparison_metrics: Fitness,
        comparison_metrics_depth: Fitness,
    ) -> Fitness:
        """
        Sub-method for add(). Give the minimum fitness in each cell of cells_indices,
        given current indivs in cells and new indivs to add in cells.
        !!!WARNING!!! This is strict min fitness and should be used with >, not >=.

        Args:
            cells_indices: the cells to consider
            batch_of_indices: indices of new indivs
            batch_of_comparison_metrics: comparison_metrics of new indivs
            comparison_metrics_depth: existing comparison_metrics depth

        Returns: minimum fitness for each cell in cells_indices
        """

        @partial(jax.jit, static_argnames=("depth",))
        def _get_cell_min_comparison_metrics(
            idx: int,
            comparison_metrics_depth_reshape: Fitness,
            depth: int,
            batch_of_indices: jnp.ndarray,
            batch_of_comparison_metrics: Fitness,
        ) -> float:
            """
            Applied using vmap on all cells_indices.
            """
            filter_comparison_metrics = jnp.where(
                batch_of_indices == idx, batch_of_comparison_metrics, -jnp.inf
            )
            all_comparison_metrics = jnp.concatenate(
                [filter_comparison_metrics, comparison_metrics_depth_reshape], axis=0
            )
            min_comparison_metrics, _ = jax.lax.top_k(all_comparison_metrics, depth + 1)
            return min_comparison_metrics[depth]  # type: ignore

        get_cell_min_comparison_metrics_fn = partial(
            _get_cell_min_comparison_metrics,
            depth=self.dims.shape[0],
            batch_of_indices=batch_of_indices,
            batch_of_comparison_metrics=batch_of_comparison_metrics,
        )
        return jax.vmap(get_cell_min_comparison_metrics_fn)(
            cells_indices,
            jnp.reshape(
                comparison_metrics_depth, (self.centroids.shape[0], self.dims.shape[0])
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
        batch_of_comparison_metrics: Fitness,
        comparison_metrics_depth: Fitness,
    ) -> jnp.ndarray:
        """
        Sub-method for add(). Return indices to place new indiv in the depth grid.

        Args:
            batch_of_indices: indices of new indivs
            batch_of_comparison_metrics: comparison_metrics of new indivs
            comparison_metrics_depth: existing comparison_metrics depth

        Returns: indices to place each new indiv
        """

        num_centroids = self.centroids.shape[0]
        depth = self.dims.shape[0]
        out_of_bound = num_centroids * depth  # Index of non-added individuals

        # Get minimum comparison_metrics in each cell after addition
        min_comparison_metrics = self._cell_min_comparison_metrics(
            cells_indices=jnp.arange(0, num_centroids, step=1),
            batch_of_indices=batch_of_indices,
            batch_of_comparison_metrics=batch_of_comparison_metrics,
            comparison_metrics_depth=comparison_metrics_depth,
        )

        # Filter individuals and keep those greater than min
        batch_of_indices = jnp.where(
            batch_of_comparison_metrics > min_comparison_metrics[batch_of_indices],
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
            jnp.reshape(comparison_metrics_depth, (num_centroids, depth)),
            min_comparison_metrics,
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

    @partial(
        jax.jit,
        static_argnames=(
            "fitness_extractor",
            "fitness_reproducibility_extractor",
            "descriptor_extractor",
            "descriptor_reproducibility_extractor",
        ),
    )
    def add(
        self,
        batch_of_genotypes: Genotype,
        batch_of_all_descriptors: Descriptor,
        batch_of_all_fitnesses: Fitness,
        batch_of_extra_scores: ExtraScores,
        fitness_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        fitness_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        descriptor_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        descriptor_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray],
    ) -> ArchiveSamplingRepertoire:
        """
        Add a batch of elements to the repertoire.
        WARNING: This addition makes the hypothesis that batch_of_all_descriptors
        and batch_of_all_fitnesses are already dimensions num_evals.

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

        # Compute batch of descriptor
        batch_of_descriptors = descriptor_extractor(batch_of_all_descriptors)

        # Compute batch of fitness
        batch_of_fitnesses = fitness_extractor(batch_of_all_fitnesses)

        # Compute indices
        batch_of_indices = get_cells_indices(batch_of_descriptors, self.centroids)

        # Filter dead individuals
        batch_of_indices = jnp.where(
            batch_of_fitnesses > -jnp.inf,
            batch_of_indices,
            out_of_bound,
        )

        # Get final indices of individuals addded to top layer of the grid
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
        # (i.e. indivs in: genotypes_depth, fitnesses_depth_all, descriptors_depth_all)
        final_batch_of_indices = self._place_indivs(
            batch_of_indices=batch_of_indices,
            batch_of_comparison_metrics=batch_of_fitnesses,
            comparison_metrics_depth=self.fitnesses_depth,
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
        new_fitnesses_depth_all = self.fitnesses_depth_all.at[
            final_batch_of_indices
        ].set(batch_of_all_fitnesses)
        new_descriptors = self.descriptors.at[final_batch_of_max_indices].set(
            batch_of_descriptors
        )
        new_descriptors_depth_all = self.descriptors_depth_all.at[
            final_batch_of_indices
        ].set(batch_of_all_descriptors)

        # Compute new evaluations
        batch_of_evaluations = batch_of_extra_scores["num_evaluations"]
        new_evaluations_depth = self.evaluations_depth.at[final_batch_of_indices].set(
            batch_of_evaluations
        )
        new_total_evaluations = self.total_evaluations + jnp.sum(batch_of_evaluations)

        return ArchiveSamplingRepertoire(
            genotypes=new_grid_genotypes,
            genotypes_depth=new_grid_genotypes_depth,
            fitnesses=new_fitnesses.squeeze(),
            fitnesses_depth=new_fitnesses_depth,
            fitnesses_depth_all=new_fitnesses_depth_all,
            descriptors=new_descriptors.squeeze(),
            descriptors_depth_all=new_descriptors_depth_all,
            evaluations_depth=new_evaluations_depth,
            total_evaluations=new_total_evaluations,
            centroids=self.centroids,
            dims=self.dims,
        )

    @jax.jit
    def set_total_evaluations(
        self, total_evaluations: int
    ) -> ArchiveSamplingRepertoire:
        """Set up current number of evaluations."""
        return self.replace(total_evaluations=total_evaluations)  # type: ignore

    @classmethod
    def init(
        cls,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
        centroids: Centroid,
        depth: int,
        num_evals: int,
        fitness_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        fitness_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        descriptor_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        descriptor_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray],
    ) -> ArchiveSamplingRepertoire:
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
            depth
            num_evals

        Returns:
            an initialized MAP-Elite repertoire
        """

        # Initialize grid with default values
        num_centroids = centroids.shape[0]
        default_fitnesses = -jnp.inf * jnp.ones(shape=num_centroids)
        default_fitnesses_depth = -jnp.inf * jnp.ones(shape=(num_centroids * depth))
        default_fitnesses_depth_all = jnp.nan * jnp.ones(
            shape=(num_centroids * depth, num_evals)
        )
        default_genotypes = jax.tree_map(
            lambda x: jnp.zeros(shape=(num_centroids,) + x.shape[1:]),
            genotypes,
        )
        default_genotypes_depth = jax.tree_map(
            lambda x: jnp.zeros(shape=(num_centroids * depth,) + x.shape[1:]),
            genotypes,
        )
        default_descriptors = jnp.zeros(shape=(num_centroids, centroids.shape[-1]))
        default_descriptors_depth_all = jnp.nan * jnp.ones(
            shape=(num_centroids * depth, num_evals, centroids.shape[-1])
        )
        default_evaluations_depth = jnp.zeros(shape=num_centroids * depth)
        default_total_evaluations = 0
        dims = jnp.zeros(shape=(depth, num_evals))

        repertoire = ArchiveSamplingRepertoire(
            genotypes=default_genotypes,
            genotypes_depth=default_genotypes_depth,
            fitnesses=default_fitnesses,
            fitnesses_depth=default_fitnesses_depth,
            fitnesses_depth_all=default_fitnesses_depth_all,
            descriptors=default_descriptors,
            descriptors_depth_all=default_descriptors_depth_all,
            evaluations_depth=default_evaluations_depth,
            total_evaluations=default_total_evaluations,
            centroids=centroids,
            dims=dims,
        )

        # Add initial values to the grid
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
    def empty(self) -> ArchiveSamplingRepertoire:
        """
        Empty the grid from all existing individuals.

        Returns:
            An empty ArchiveSamplingRepertoire
        """

        new_fitnesses = jnp.full_like(self.fitnesses, -jnp.inf)
        new_fitnesses_depth = jnp.full_like(self.fitnesses_depth, -jnp.inf)
        new_fitnesses_depth_all = jnp.full_like(self.fitnesses_depth_all, jnp.nan)
        new_descriptors = jnp.zeros_like(self.descriptors)
        new_descriptors_depth_all = jnp.full_like(self.descriptors_depth_all, jnp.nan)
        new_genotypes = jax.tree_map(lambda x: jnp.zeros_like(x), self.genotypes)
        new_genotypes_depth = jax.tree_map(
            lambda x: jnp.zeros_like(x), self.genotypes_depth
        )
        new_evaluations_depth = jnp.zeros_like(self.evaluations_depth)
        new_total_evaluations = 0
        return ArchiveSamplingRepertoire(
            genotypes=new_genotypes,
            genotypes_depth=new_genotypes_depth,
            fitnesses=new_fitnesses,
            fitnesses_depth=new_fitnesses_depth,
            fitnesses_depth_all=new_fitnesses_depth_all,
            descriptors=new_descriptors,
            descriptors_depth_all=new_descriptors_depth_all,
            evaluations_depth=new_evaluations_depth,
            total_evaluations=new_total_evaluations,
            centroids=self.centroids,
            dims=self.dims,
        )

    @jax.jit
    def added_repertoire(
        self,
        genotypes: Genotype,
        descriptors: Descriptor,
    ) -> jnp.ndarray:
        """Compute if the given genotypes have been added to the repertoire in
        corresponding cell.

        Args:
            genotypes: genotypes candidate to addition
            descriptors: corresponding descriptors
        Returns:
            boolean for each genotype
        """
        cells = get_cells_indices(descriptors, self.centroids)
        repertoire_genotypes = jax.tree_util.tree_map(
            lambda x: x[cells], self.genotypes_depth
        )
        added = jax.tree_util.tree_map(
            lambda x, y: jnp.equal(x, y), genotypes, repertoire_genotypes
        )
        added = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (descriptors.shape[0], -1)), added
        )
        added = jax.tree_util.tree_map(lambda x: jnp.all(x, axis=1), added)
        final_added = jnp.array(jax.tree_util.tree_leaves(added))
        final_added = jnp.all(final_added, axis=0)
        return final_added
