from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.standard_emitters import EmitterState, MixingEmitter
from qdax.custom_types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey


class RandomEmitter(MixingEmitter):
    def __init__(
        self,
        batch_size: int,
        genotypes: Genotype,
        min_genotype: float,
        max_genotype: float,
        behavior_descriptor_length: float,
    ) -> None:
        self._batch_size = batch_size
        self._genotypes = genotypes
        self._min_genotype = min_genotype
        self._max_genotype = max_genotype
        self._behavior_descriptor_length = behavior_descriptor_length

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def emit(
        self,
        repertoire: Repertoire,
        emitter_state: EmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, ExtraScores, RNGKey]:
        """
        Emitter that generate batch-size random individuals.

        Note: this emitter has no state. A fake none state must be added
        through a function redefinition to make this emitter usable with MAP-Elites.

        Params:
            repertoire: the MAP-Elites repertoire to sample from
            emitter_state: void
            random_key: a jax PRNG random key

        Returns:
            a batch of offsprings
            a new jax PRNG key
        """

        # Create new offsprings
        random_key, subkey = jax.random.split(random_key)
        offspring = jax.tree_map(
            lambda x: jnp.repeat(jnp.expand_dims(x, axis=0), self._batch_size, axis=0),
            self._genotypes,
        )
        offspring = jax.tree_map(
            lambda x: jax.random.uniform(
                subkey,
                shape=x.shape,
                minval=self._min_genotype,
                maxval=self._max_genotype,
            ),
            offspring,
        )

        return offspring, {}, random_key

    @partial(jax.jit, static_argnames=("self",))
    def state_update(
        self,
        emitter_state: EmitterState,
        repertoire: Repertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
    ) -> EmitterState:

        # Update metrics
        usage = jnp.sum(repertoire.added_repertoire(genotypes, descriptors))
        parents_bd_distance = jnp.average(
            jnp.sqrt(
                jnp.sum(
                    jnp.square(emitter_state.parents_descriptors - descriptors),
                    axis=1,
                )
            ),
            axis=0,
        )
        genotype_distance = jax.tree_util.tree_map(
            lambda x, y: jnp.sqrt(
                jnp.sum(jnp.reshape(jnp.square(x - y), (self._batch_size, -1)), axis=1)
            ),
            emitter_state.parents,
            genotypes,
        )
        parents_genotype_distance = jnp.average(
            jnp.array(
                jax.tree_util.tree_leaves(
                    genotype_distance,
                )
            ),
            axis=0,
        )

        return emitter_state.replace(  # type: ignore
            usage=usage,
            parents=genotypes,
            parents_descriptors=descriptors,
            parents_bd_distance=parents_bd_distance,
            parents_genotype_distance=parents_genotype_distance,
        )
