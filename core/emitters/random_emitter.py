from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.types import Genotype, RNGKey


class RandomEmitter(Emitter):
    def __init__(
        self,
        batch_size: int,
        init_genotype: Genotype,
        min_genotype: float,
        max_genotype: float,
    ) -> None:
        self._batch_size = batch_size
        self._init_genotype = init_genotype
        self._min_genotype = min_genotype
        self._max_genotype = max_genotype

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def emit(
        self,
        repertoire: Repertoire,
        unused_emitter_state: Optional[EmitterState],
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """
        Emitter that generate batch-size random individuals.

        Note: this emitter has no state. A fake none state must be added
        through a function redefinition to make this emitter usable with MAP-Elites.

        Params:
            repertoire: the MAP-Elites repertoire to sample from
            unused_emitter_state: void
            random_key: a jax PRNG random key

        Returns:
            a batch of offsprings
            a new jax PRNG key
        """
        random_key, subkey = jax.random.split(random_key)

        offspring = jax.tree_map(
            lambda x: jnp.repeat(jnp.expand_dims(x, axis=0), self._batch_size, axis=0),
            self._init_genotype,
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

        return offspring, random_key
