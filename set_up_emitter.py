from functools import partial
from typing import Any, Tuple

import jax
from qdax.core.containers.archive import score_euclidean_novelty
from qdax.core.emitters.dcrl_me_emitter import DCRLMEConfig, DCRLMEEmitter
from qdax.core.emitters.dpg_emitter import DiversityPGConfig
from qdax.core.emitters.emitter import Emitter
from qdax.core.emitters.mutation_operators import isoline_variation, polynomial_mutation
from qdax.core.emitters.pga_me_emitter import PGAMEConfig, PGAMEEmitter
from qdax.core.emitters.qdpg_emitter import QDPGEmitter, QDPGEmitterConfig
from qdax.core.emitters.qpg_emitter import QualityPGConfig
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.custom_types import Genotype, RNGKey

from core.emitters.random_emitter import RandomEmitter

# Emitter list
EMITTER_LIST = [
    "Random",
    "Mixing",
    "PGA",
    "DCG",
    "QDPG",
]

# Metrics type returned by each emitter
USAGE_EMITTER = [
    "Random",
    "Mixing",
]

# Mutation list
MUTATION_LIST = [
    "isoline",
    "polynomial",
]


def get_evals_per_offspring(args: Any) -> int:
    """Get the number of samples spent on each offspring."""

    # Base is the number of samples
    evals_per_offspring = args.num_samples

    # PGA, DCG and QDPG case
    if args.emitter == "PGA" or args.emitter == "DCG" or args.emitter == "QDPG":
        # Those appraoch cannot have a batch-size greater than 256
        evals_per_offspring = max(args.num_samples, args.sampling_size // 256)

    return evals_per_offspring  # type: ignore


def set_up_emitter(
    emitter_name: str,
    num_iterations: int,
    effective_batch_size: int,
    env: Any,
    num_descriptors: int,
    num_centroids: int,
    hard_limit_genotype: bool,
    min_genotype: float,
    max_genotype: float,
    policy_structure: Any,
    policy_dc_structure: Any,
    init_policies: Genotype,
    mutation: str,
    iso_sigma: float,
    line_sigma: float,
    proportion_mutation: float,
    eta: float,
    pg_num_critic_training_steps: int,
    pg_critic_hidden_layer_sizes: Tuple,
    pg_replay_buffer_size: int,
    random_key: RNGKey,
) -> Tuple[Emitter, RNGKey]:

    # Input check
    assert emitter_name in EMITTER_LIST, "\n!!!ERROR!!! Invalid emitter:" + emitter_name
    assert mutation in MUTATION_LIST, "\n!!!ERROR!!! Invalid mutation:" + mutation
    if emitter_name == "PGA" or emitter_name == "DCG" or emitter_name == "QDPG":
        assert (
            mutation == "isoline"
        ), "\n!!!ERROR!!! PGA, DCG and QDPG only defined for isoline for now."

    # Define mutation
    if mutation == "isoline":
        variation_fn = partial(
            isoline_variation,
            iso_sigma=iso_sigma,
            line_sigma=line_sigma,
            minval=min_genotype if hard_limit_genotype else None,
            maxval=max_genotype if hard_limit_genotype else None,
        )
        mutation_fn = None
        variation_percentage = 1.0
    elif mutation == "polynomial":
        variation_fn = None
        mutation_fn = partial(
            polynomial_mutation,
            proportion_to_mutate=proportion_mutation,
            eta=eta,
            minval=min_genotype,
            maxval=max_genotype,
        )
        variation_percentage = 0.0
    else:
        assert 0, "!!!ERROR!!! Undefined mutation."

    # Define emitter
    if emitter_name == "Random":
        emitter = RandomEmitter(
            batch_size=effective_batch_size,
            genotypes=jax.tree_map(lambda x: x[0], init_policies),
            min_genotype=min_genotype,
            max_genotype=max_genotype,
            behavior_descriptor_length=num_descriptors,
        )
    elif emitter_name == "Mixing":
        emitter = MixingEmitter(
            mutation_fn=mutation_fn,
            variation_fn=variation_fn,
            variation_percentage=variation_percentage,
            batch_size=effective_batch_size,
        )
    elif emitter_name == "PGA":
        pga_config = PGAMEConfig(
            env_batch_size=effective_batch_size,
            num_critic_training_steps=pg_num_critic_training_steps,
            critic_hidden_layer_size=pg_critic_hidden_layer_sizes,
            replay_buffer_size=pg_replay_buffer_size,
        )
        emitter = PGAMEEmitter(
            config=pga_config,
            policy_network=policy_structure,
            env=env,
            variation_fn=variation_fn,
        )

    elif emitter_name == "DCG":
        ga_batch_size = effective_batch_size // 2
        qpg_batch_size = ga_batch_size // 2
        ai_batch_size = int(effective_batch_size - ga_batch_size - qpg_batch_size)
        dcg_emitter_config = DCRLMEConfig(
            ga_batch_size=ga_batch_size,
            qpg_batch_size=qpg_batch_size,
            ai_batch_size=ai_batch_size,
            num_critic_training_steps=pg_num_critic_training_steps,
            critic_hidden_layer_size=pg_critic_hidden_layer_sizes,
            replay_buffer_size=pg_replay_buffer_size,
        )
        emitter = DCRLMEEmitter(
            config=dcg_emitter_config,
            policy_network=policy_structure,
            actor_network=policy_dc_structure,
            env=env,
            variation_fn=variation_fn,
        )

    elif emitter_name == "QDPG":
        qdpg_diversity_pg_batch_size = effective_batch_size // 3
        qdpg_ga_batch_size = qdpg_diversity_pg_batch_size
        qdpg_quality_pg_batch_size = int(
            effective_batch_size - qdpg_diversity_pg_batch_size - qdpg_ga_batch_size
        )

        assert (
            qdpg_quality_pg_batch_size
            + qdpg_diversity_pg_batch_size
            + qdpg_ga_batch_size
            == effective_batch_size
        ), "!!!ERROR!!! Incorrect splitting between quality diversity and GA in QDPG."

        print("Running QDPG with batch-sizes:")
        print(
            f"Quality {qdpg_quality_pg_batch_size}, Diversity {qdpg_diversity_pg_batch_size}, GA {qdpg_ga_batch_size}."
        )

        # Define the Quality PG emitter config
        qpg_emitter_config = QualityPGConfig(
            env_batch_size=qdpg_quality_pg_batch_size,
            replay_buffer_size=pg_replay_buffer_size,
            num_critic_training_steps=pg_num_critic_training_steps,
            critic_hidden_layer_size=pg_critic_hidden_layer_sizes,
        )

        # Define the Diversity PG emitter config
        dpg_emitter_config = DiversityPGConfig(
            env_batch_size=qdpg_diversity_pg_batch_size,
            replay_buffer_size=pg_replay_buffer_size,
            num_critic_training_steps=pg_num_critic_training_steps,
            critic_hidden_layer_size=pg_critic_hidden_layer_sizes,
        )

        # Define the QDPG Emitter config
        qdpg_emitter_config = QDPGEmitterConfig(
            qpg_config=qpg_emitter_config,
            dpg_config=dpg_emitter_config,
            iso_sigma=iso_sigma,
            line_sigma=line_sigma,
            ga_batch_size=qdpg_ga_batch_size,
        )
        score_novelty = jax.jit(
            partial(
                score_euclidean_novelty,
                num_nearest_neighb=5,
                scaling_ratio=1.0,
            )
        )

        # define the QDPG emitter
        emitter = QDPGEmitter(
            config=qdpg_emitter_config,
            policy_network=policy_structure,
            env=env,
            score_novelty=score_novelty,
        )

    return emitter, random_key
