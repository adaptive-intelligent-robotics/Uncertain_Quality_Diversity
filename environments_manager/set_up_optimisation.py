from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from qdax.core.containers.repertoire import Repertoire
from qdax.custom_types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey

from core.sampling import dummy_extra_scores_extractor
from environments_manager.uncertainty_in_cell_metrics import (
    incell_reevaluation_function,
)
from environments_manager.uncertainty_metrics import reevaluation_function
from tasks.arm import (
    ArmBimodalGaussianDesc,
    ArmBimodalGaussianFitness,
    ArmGaussianDescBiVarianceNoise,
    ArmGaussianDescFitPropVarianceNoise,
    ArmGaussianNoise,
    ArmSelectedJointsNoise,
    arm_scoring_function,
)
from tasks.direct_mapping import (
    deceptive_direct_mapping_scoring_function,
    direct_mapping_scoring_function,
    no_trade_off_direct_mapping_scoring_function,
    perfect_trade_off_direct_mapping_scoring_function,
    sharp_peak_direct_mapping_scoring_function,
)
from tasks.optimisation_problems import (
    rastrigin_scoring_function,
    sphere_scoring_function,
)


def reevaluation_function_ground_truth(
    repertoire: Repertoire,
    random_key: RNGKey,
    metric_repertoire: Repertoire,
    scoring_fn: Callable[
        [Genotype, RNGKey],
        Tuple[Fitness, Descriptor, ExtraScores, RNGKey],
    ],
    num_reevals: int,
    scan_size: int,
    batch_size: int,
    fitness_extractor: Callable[[jnp.ndarray], jnp.ndarray],
    fitness_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray],
    descriptor_extractor: Callable[[jnp.ndarray], jnp.ndarray],
    descriptor_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray],
    extra_scores_extractor: Callable[[ExtraScores, int], ExtraScores],
    ground_truth_scoring_fn: Callable[
        [Genotype, RNGKey],
        Tuple[Fitness, Fitness, Descriptor, Fitness, ExtraScores, RNGKey],
    ],
) -> Tuple[
    Repertoire,
    Repertoire,
    Repertoire,
    Repertoire,
    Repertoire,
    Repertoire,
    Repertoire,
    RNGKey,
]:
    """
    Use ground truth instead of truly performing reevaluation.
    WARNING: most arguments have no effect here, this is just to preserve interface.

    Args:
        repertoire: repertoire to reevaluate.
        metric_repertoire: repertoire used to compute reeval stats, allow to use a
            different type of container than the one from the algorithm (in most cases
            just set to the same as repertoire).
        random_key: JAX random key.
        scoring_fn: scoring function used for evaluation.
        num_reevals: number of samples to generate for each individual.
        scan_size: allow to split the reevaluations in multiple batch in case the
            memory is limited.
        fitness_extractor: function to extract the final fitness from
            multiple samples of the same policy.
        fitness_reproducibility_extractor: function to extract the fitness
            reproducibility from multiple samples of the same policy.
        descriptor_extractor: function to extract the final descriptor from
            multiple samples of the same policy.
        descriptor_reproducibility_extractor: function to extract the descriptor
            reproducibility from multiple samples of the same policy.
        extra_scores_extractor: function to extract the extra_scores from
            multiple samples of the same policy.
    Returns:
        A container with reevaluated fitness and descriptor.
        A container with reevaluated fitness only.
        A container with reevaluated descriptor only.
        A non-reevaluated container with reproducibility in fitness.
        A reevaluated container with reproducibility in fitness.
        A non-reevaluated container with reproducibility in descriptor.
        A reevaluated container with reproducibility in descriptor.
        A random key.
    """

    # If no reevaluations, return copies of the original container
    if num_reevals == 0:
        return (
            repertoire,
            repertoire,
            repertoire,
            repertoire,
            repertoire,
            repertoire,
            repertoire,
            random_key,
        )

    # Else evaluate through ground truth function
    (
        fitnesses,
        fitnesses_var,
        descriptors,
        descriptors_var,
        extra_scores,
        random_key,
    ) = ground_truth_scoring_fn(repertoire.genotypes, random_key)

    # Fill-in reeval repertoire
    reeval_repertoire = metric_repertoire.empty()
    reeval_repertoire = reeval_repertoire.add(
        repertoire.genotypes,
        descriptors,
        fitnesses,
        {},
    )

    # Fill-in fit_reeval repertoire
    fit_reeval_repertoire = metric_repertoire.empty()
    fit_reeval_repertoire = fit_reeval_repertoire.add(
        repertoire.genotypes,
        repertoire.descriptors,
        fitnesses,
        {},
    )

    # Fill-in desc_reeval repertoire
    desc_reeval_repertoire = metric_repertoire.empty()
    desc_reeval_repertoire = desc_reeval_repertoire.add(
        repertoire.genotypes,
        descriptors,
        repertoire.fitnesses,
        {},
    )

    # Fill-in fit_var repertoire
    fit_reproducibility_repertoire = metric_repertoire.empty()
    fit_reproducibility_repertoire = fit_reproducibility_repertoire.add(
        repertoire.genotypes,
        repertoire.descriptors,
        fitnesses_var,
        {},
    )

    # Fill-in reeval_fit_var repertoire
    reeval_fit_reproducibility_repertoire = metric_repertoire.empty()
    reeval_fit_reproducibility_repertoire = reeval_fit_reproducibility_repertoire.add(
        repertoire.genotypes,
        descriptors,
        fitnesses_var,
        {},
    )

    # Fill-in desc_var repertoire
    desc_reproducibility_repertoire = metric_repertoire.empty()
    desc_reproducibility_repertoire = desc_reproducibility_repertoire.add(
        repertoire.genotypes,
        repertoire.descriptors,
        descriptors_var,
        {},
    )

    # Fill-in reeval_desc_var repertoire
    reeval_desc_reproducibility_repertoire = metric_repertoire.empty()
    reeval_desc_reproducibility_repertoire = reeval_desc_reproducibility_repertoire.add(
        repertoire.genotypes,
        descriptors,
        descriptors_var,
        {},
    )

    return (
        reeval_repertoire,
        fit_reeval_repertoire,
        desc_reeval_repertoire,
        fit_reproducibility_repertoire,
        reeval_fit_reproducibility_repertoire,
        desc_reproducibility_repertoire,
        reeval_desc_reproducibility_repertoire,
        random_key,
    )


def incell_reevaluation_function_ground_truth(
    repertoire: Repertoire,
    random_key: RNGKey,
    metric_repertoire: Repertoire,
    scoring_fn: Callable[
        [Genotype, RNGKey],
        Tuple[Fitness, Descriptor, ExtraScores, RNGKey],
    ],
    depth: int,
    num_reevals: int,
    scan_size: int,
    batch_size: int,
    fitness_extractor: Callable[[jnp.ndarray], jnp.ndarray],
    fitness_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray],
    descriptor_extractor: Callable[[jnp.ndarray], jnp.ndarray],
    descriptor_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray],
    extra_scores_extractor: Callable[[ExtraScores, int], ExtraScores],
    ground_truth_scoring_fn: Callable[
        [Genotype, RNGKey],
        Tuple[Fitness, Fitness, Descriptor, Fitness, ExtraScores, RNGKey],
    ],
) -> Tuple[
    Repertoire,
    Repertoire,
    Repertoire,
    Repertoire,
    Repertoire,
    Repertoire,
    Repertoire,
    Repertoire,
    RNGKey,
]:
    """
    Use ground truth instead of truly performing reevaluation.
    WARNING: most arguments have no effect here, this is just to preserve interface.

    Args:
        repertoire: repertoire to reevaluate.
        metric_repertoire: repertoire used to compute reeval stats, allow to use a
            different type of container than the one from the algorithm (in most cases
            just set to the same as repertoire).
        random_key: JAX random key.
        scoring_fn: scoring function used for evaluation.
        num_reevals: number of samples to generate for each individual.
        scan_size: allow to split the reevaluations in multiple batch in case the
            memory is limited.
                fitness_extractor: function to extract the final fitness from
            multiple samples of the same policy.
                fitness_reproducibility_extractor: function to extract the fitness
            reproducibility from multiple samples of the same policy.
        descriptor_extractor: function to extract the final descriptor from
            multiple samples of the same policy.
                descriptor_reproducibility_extractor: function to extract the descriptor
            reproducibility from multiple samples of the same policy.
        extra_scores_extractor: function to extract the extra_scores from
            multiple samples of the same policy.
    Returns:
        A container with reevaluated fitness and descriptor.
        A container with reevaluated fitness only.
        A container with reevaluated descriptor only.
        A non-reevaluated container with reproducibility in fitness.
        A reevaluated container with reproducibility in fitness.
        A non-reevaluated container with reproducibility in descriptor.
        A reevaluated container with reproducibility in descriptor.
        A random key.
    """

    depth = repertoire.fitnesses_depth.shape[1]
    num_centroids = repertoire.fitnesses_depth.shape[0]

    # Sample depth fitnesses and descriptors for each cell
    (
        _,
        all_repertoire_fitnesses,
        all_repertoire_descriptors,
        random_key,
    ) = repertoire.sample_all_cells(random_key, depth)
    repertoire_genotypes = repertoire.genotypes
    repertoire_fitnesses = fitness_extractor(all_repertoire_fitnesses)
    repertoire_descriptors = descriptor_extractor(all_repertoire_descriptors)

    # Set -inf fitness for all unexisting indivs
    mask = repertoire.fitnesses == -jnp.inf
    repertoire_fitnesses = jnp.where(mask, -jnp.inf, repertoire_fitnesses)

    # Build the incell_repertoire
    incell_repertoire = metric_repertoire.empty()
    incell_repertoire = incell_repertoire.add(
        repertoire_genotypes,
        repertoire_descriptors,
        repertoire_fitnesses,
        {},
    )

    # If no reevaluations, return copies of the original container
    if num_reevals == 0:
        return (
            incell_repertoire,
            incell_repertoire,
            incell_repertoire,
            incell_repertoire,
            incell_repertoire,
            incell_repertoire,
            incell_repertoire,
            incell_repertoire,
            random_key,
        )

    # Resample num_reevals from all cells
    all_repertoire_genotypes, _, _, random_key = repertoire.sample_all_cells(
        random_key, num_reevals
    )
    all_repertoire_genotypes = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (num_centroids * num_reevals,) + x.shape[2:]),
        all_repertoire_genotypes,
    )

    # Else evaluate through ground truth function
    (
        all_fitnesses,
        all_fitnesses_var,
        all_descriptors,
        all_descriptors_var,
        _,
        random_key,
    ) = ground_truth_scoring_fn(all_repertoire_genotypes, random_key)

    # Get correct output shape
    all_fitnesses = jnp.reshape(all_fitnesses, (num_centroids, num_reevals))
    all_descriptors = jnp.reshape(all_descriptors, (num_centroids, num_reevals, -1))
    all_fitnesses_var = jnp.reshape(all_fitnesses_var, (num_centroids, num_reevals))
    all_descriptors_var = jnp.reshape(all_descriptors_var, (num_centroids, num_reevals))

    # Extract the final scores
    fitnesses = fitness_extractor(all_fitnesses)
    descriptors = descriptor_extractor(all_descriptors)

    # WARNING: in the case of reproducibility, take average
    fitnesses_reproducibility = jnp.average(all_fitnesses_var, axis=-1)
    descriptors_reproducibility = jnp.average(all_descriptors_var, axis=-1)
    descriptors_reproducibility = jnp.average(descriptors_reproducibility, axis=-1)

    # Set -inf fitness for all unexisting indivs
    fitnesses = jnp.where(mask, -jnp.inf, fitnesses)
    fitnesses_reproducibility = jnp.where(mask, -jnp.inf, fitnesses_reproducibility)
    descriptors_reproducibility = jnp.where(mask, -jnp.inf, descriptors_reproducibility)

    # Fill-in reeval repertoire
    reeval_repertoire = metric_repertoire.empty()
    reeval_repertoire = reeval_repertoire.add(
        repertoire_genotypes,
        descriptors,
        fitnesses,
        {},
    )

    # Fill-in fit_reeval repertoire
    fit_reeval_repertoire = metric_repertoire.empty()
    fit_reeval_repertoire = fit_reeval_repertoire.add(
        repertoire_genotypes,
        repertoire_descriptors,
        fitnesses,
        {},
    )

    # Fill-in desc_reeval repertoire
    desc_reeval_repertoire = metric_repertoire.empty()
    desc_reeval_repertoire = desc_reeval_repertoire.add(
        repertoire_genotypes,
        descriptors,
        repertoire_fitnesses,
        {},
    )

    # Fill-in fit_var repertoire
    fit_reproducibility_repertoire = metric_repertoire.empty()
    fit_reproducibility_repertoire = fit_reproducibility_repertoire.add(
        repertoire_genotypes,
        repertoire_descriptors,
        fitnesses_reproducibility,
        {},
    )

    # Fill-in reeval_fit_var repertoire
    reeval_fit_reproducibility_repertoire = metric_repertoire.empty()
    reeval_fit_reproducibility_repertoire = reeval_fit_reproducibility_repertoire.add(
        repertoire_genotypes,
        descriptors,
        fitnesses_reproducibility,
        {},
    )

    # Fill-in desc_var repertoire
    desc_reproducibility_repertoire = metric_repertoire.empty()
    desc_reproducibility_repertoire = desc_reproducibility_repertoire.add(
        repertoire_genotypes,
        repertoire_descriptors,
        descriptors_reproducibility,
        {},
    )

    # Fill-in reeval_desc_var repertoire
    reeval_desc_reproducibility_repertoire = metric_repertoire.empty()
    reeval_desc_reproducibility_repertoire = reeval_desc_reproducibility_repertoire.add(
        repertoire.genotypes,
        descriptors,
        descriptors_reproducibility,
        {},
    )

    return (
        incell_repertoire,
        reeval_repertoire,
        fit_reeval_repertoire,
        desc_reeval_repertoire,
        fit_reproducibility_repertoire,
        reeval_fit_reproducibility_repertoire,
        desc_reproducibility_repertoire,
        reeval_desc_reproducibility_repertoire,
        random_key,
    )


def set_up_optimisation(
    env_name: str,
    batch_size: int,
    policy_hidden_layer_sizes: Tuple,
    random_key: RNGKey,
    deterministic: bool = False,
    fit_std: float = 0.0,
    desc_std: float = 0.0,
    params_std: float = 0.0,
    delta_fitness: float = 0.0,
    delta_reproducibility: float = 0.0,
) -> Tuple:
    """
    Build the optimisation tasks.
    """

    if len(policy_hidden_layer_sizes) > 1:
        print(
            "\n!!!WARNING!!! For optimisation functions,",
            "only the first element of policy_hidden_layer_sizes:",
            policy_hidden_layer_sizes[0],
            "is used as genotype dimension.",
        )

    # If deterministic, set all noise params to 0
    if deterministic:
        fit_std = 0
        desc_std = 0
        params_std = 0

    # Sphere
    if env_name == "sphere":

        scoring_fn = partial(
            sphere_scoring_function,
            fit_std=fit_std,
            desc_std=desc_std,
            params_std=params_std,
        )
        qd_offset = 50 * policy_hidden_layer_sizes[0]

        def ground_truth_scoring_fn(
            params: Genotype, random_key: RNGKey
        ) -> Tuple[Fitness, Fitness, Descriptor, Fitness, ExtraScores, RNGKey]:

            fitnesses, descriptors, extra_scores, random_key = sphere_scoring_function(
                params=params,
                random_key=random_key,
                fit_std=0,
                desc_std=0,
                params_std=0,
            )
            fitnesses_var = fit_std * jnp.ones_like(fitnesses)
            descriptors_var = desc_std * jnp.ones_like(fitnesses)
            return (
                fitnesses,
                fitnesses_var,
                descriptors,
                descriptors_var,
                extra_scores,
                random_key,
            )

        def init_policies_fn(
            size: int, random_key: RNGKey
        ) -> Tuple[jnp.ndarray, RNGKey]:
            random_key, subkey = jax.random.split(random_key)
            init_policies = jax.random.uniform(
                random_key,
                shape=(size, policy_hidden_layer_sizes[0]),
                minval=0,
                maxval=1,
            )
            return init_policies, random_key

    # Rastrigin
    elif env_name == "rastrigin":

        scoring_fn = partial(
            rastrigin_scoring_function,
            fit_std=fit_std,
            desc_std=desc_std,
            params_std=params_std,
        )
        qd_offset = 50 + 50 * policy_hidden_layer_sizes[0]

        def ground_truth_scoring_fn(
            params: Genotype, random_key: RNGKey
        ) -> Tuple[Fitness, Fitness, Descriptor, Fitness, ExtraScores, RNGKey]:

            (
                fitnesses,
                descriptors,
                extra_scores,
                random_key,
            ) = rastrigin_scoring_function(
                params,
                random_key,
                fit_std=0,
                desc_std=0,
                params_std=0,
            )
            fitnesses_var = fit_std * jnp.ones_like(fitnesses)
            descriptors_var = desc_std * jnp.ones_like(fitnesses)
            return (
                fitnesses,
                fitnesses_var,
                descriptors,
                descriptors_var,
                extra_scores,
                random_key,
            )

        def init_policies_fn(
            size: int, random_key: RNGKey
        ) -> Tuple[jnp.ndarray, RNGKey]:
            random_key, subkey = jax.random.split(random_key)
            init_policies = jax.random.uniform(
                random_key,
                shape=(size, policy_hidden_layer_sizes[0]),
                minval=0,
                maxval=1,
            )
            return init_policies, random_key

    # Direct mapping with no trade-off
    elif env_name == "direct_mapping_no_trade_off":

        if policy_hidden_layer_sizes[0] != 3:
            print("!!!WARNING!!! Genotype is always dimension 3 for direct_mapping.")
            policy_hidden_layer_sizes = (3,)

        fit_std = 0
        params_std = 0
        scoring_fn = partial(
            no_trade_off_direct_mapping_scoring_function,
            desc_std=desc_std,
        )
        qd_offset = 0

        def ground_truth_scoring_fn(
            params: Genotype, random_key: RNGKey
        ) -> Tuple[Fitness, Fitness, Descriptor, Fitness, ExtraScores, RNGKey]:

            (
                fitnesses,
                descriptors,
                extra_scores,
                random_key,
            ) = direct_mapping_scoring_function(params, random_key)
            fitnesses_var = jnp.zeros_like(fitnesses)
            descriptors_var = jnp.ones_like(fitnesses)
            return (
                fitnesses,
                fitnesses_var,
                descriptors,
                descriptors_var,
                extra_scores,
                random_key,
            )

        def init_policies_fn(
            size: int, random_key: RNGKey
        ) -> Tuple[jnp.ndarray, RNGKey]:
            random_key, subkey = jax.random.split(random_key)
            init_policies = jnp.abs(
                jax.random.normal(
                    random_key, shape=(size, policy_hidden_layer_sizes[0])
                )
                * 0.1
            )
            return init_policies, random_key

    # Direct mapping with perfect trade-off
    elif "direct_mapping_perfect_trade_off" in env_name:

        if policy_hidden_layer_sizes[0] != 3:
            print("!!!WARNING!!! Genotype is always dimension 3 for direct_mapping.")
            policy_hidden_layer_sizes = (3,)

        fit_std = 0
        params_std = 0
        if env_name == "direct_mapping_perfect_trade_off_0.02":
            print("!!!WARNING!!! For direct_mapping_perfect_trade_off_0.02:")
            print("- desc_std is set to 0.2.")
            print("- delta_fitness and delta_reproducibility are set to 0.02.")
            delta_fitness = 0.02
            delta_reproducibility = 0.02
            desc_std = 0.2

        scoring_fn = partial(
            perfect_trade_off_direct_mapping_scoring_function,
            desc_std=desc_std,
        )
        qd_offset = 0

        def ground_truth_scoring_fn(
            params: Genotype, random_key: RNGKey
        ) -> Tuple[Fitness, Fitness, Descriptor, Fitness, ExtraScores, RNGKey]:

            (
                fitnesses,
                descriptors,
                extra_scores,
                random_key,
            ) = direct_mapping_scoring_function(params, random_key)
            fitnesses_var = jnp.zeros_like(fitnesses)
            descriptors_var = fitnesses
            return (
                fitnesses,
                fitnesses_var,
                descriptors,
                descriptors_var,
                extra_scores,
                random_key,
            )

        def init_policies_fn(
            size: int, random_key: RNGKey
        ) -> Tuple[jnp.ndarray, RNGKey]:
            random_key, subkey = jax.random.split(random_key)
            init_policies = jnp.abs(
                jax.random.normal(
                    random_key, shape=(size, policy_hidden_layer_sizes[0])
                )
                * 0.1
            )
            return init_policies, random_key

    # Direct mapping with sharp peak
    elif "direct_mapping_sharp_peak" in env_name:

        if policy_hidden_layer_sizes[0] != 3:
            print("!!!WARNING!!! Genotype is always dimension 3 for direct_mapping.")
            policy_hidden_layer_sizes = (3,)

        fit_std = 0
        params_std = 0
        if env_name == "direct_mapping_sharp_peak_bigger_0.2":
            print("!!!WARNING!!! For direct_mapping_sharp_peak_bigger_0.2:")
            print("- desc_std is set to 0.05.")
            print("- delta_fitness to 0.2 and delta_reproducibility to 0.02.")
            delta_fitness = 0.2
            delta_reproducibility = 0.02
            desc_std = 0.05
        elif env_name == "direct_mapping_sharp_peak_smaller_0.02":
            print("!!!WARNING!!! For direct_mapping_sharp_peak_smaller_0.02:")
            print("- desc_std is set to 0.05.")
            print("- delta_fitness to 0.02 and delta_reproducibility to 0.02.")
            delta_fitness = 0.02
            delta_reproducibility = 0.02
            desc_std = 0.05

        scoring_fn = partial(
            sharp_peak_direct_mapping_scoring_function,
            desc_std=desc_std,
        )
        qd_offset = 0

        def ground_truth_scoring_fn(
            params: Genotype, random_key: RNGKey
        ) -> Tuple[Fitness, Fitness, Descriptor, Fitness, ExtraScores, RNGKey]:

            (
                fitnesses,
                descriptors,
                extra_scores,
                random_key,
            ) = direct_mapping_scoring_function(params, random_key)
            fitnesses_var = jnp.zeros_like(fitnesses)
            descriptors_var = jnp.where(fitnesses < 0.9, 0.0, 1.0)
            return (
                fitnesses,
                fitnesses_var,
                descriptors,
                descriptors_var,
                extra_scores,
                random_key,
            )

        def init_policies_fn(
            size: int, random_key: RNGKey
        ) -> Tuple[jnp.ndarray, RNGKey]:
            random_key, subkey = jax.random.split(random_key)
            init_policies = jnp.abs(
                jax.random.normal(
                    random_key, shape=(size, policy_hidden_layer_sizes[0])
                )
                * 0.1
            )
            return init_policies, random_key

    # Direct mapping with deceptive
    elif "direct_mapping_deceptive" in env_name:

        if policy_hidden_layer_sizes[0] != 3:
            print("!!!WARNING!!! Genotype is always dimension 3 for direct_mapping.")
            policy_hidden_layer_sizes = (3,)

        fit_std = 0
        params_std = 0
        if env_name == "direct_mapping_deceptive_0.1":
            print("!!!WARNING!!! For direct_mapping_deceptive_0.1:")
            print("- desc_std is set to 0.1.")
            print("- elta_fitness and delta_reproducibility to 0.05.")
            delta_fitness = 0.05
            delta_reproducibility = 0.05
            desc_std = 0.1

        scoring_fn = partial(
            deceptive_direct_mapping_scoring_function,
            desc_std=desc_std,
        )
        qd_offset = 0

        def ground_truth_scoring_fn(
            params: Genotype, random_key: RNGKey
        ) -> Tuple[Fitness, Fitness, Descriptor, Fitness, ExtraScores, RNGKey]:

            (
                fitnesses,
                descriptors,
                extra_scores,
                random_key,
            ) = direct_mapping_scoring_function(params, random_key)
            fitnesses_var = jnp.zeros_like(fitnesses)
            descriptors_var = jnp.where(
                jnp.logical_or(fitnesses < 0.6, fitnesses > 0.7), 0.0, 1.0
            )
            return (
                fitnesses,
                fitnesses_var,
                descriptors,
                descriptors_var,
                extra_scores,
                random_key,
            )

        def init_policies_fn(
            size: int, random_key: RNGKey
        ) -> Tuple[jnp.ndarray, RNGKey]:
            random_key, subkey = jax.random.split(random_key)
            init_policies = jnp.abs(
                jax.random.normal(
                    random_key, shape=(size, policy_hidden_layer_sizes[0])
                )
                * 0.1
            )
            return init_policies, random_key

    # Arm with Gaussian fit and desc noise
    elif env_name == "arm_gaussian":

        if fit_std <= 0 or desc_std <= 0:
            print("\n!!!WARNING!!! Invalid std, using default values.")
            fit_std = 0.01
            desc_std = 0.01

        print(
            f"Using Gaussian noise on fitness with var {fit_std} and descriptor with var {desc_std}"
        )

        params_std = 0
        env = ArmGaussianNoise(
            fit_std=fit_std,
            desc_std=desc_std,
            params_std=0.0,
        )
        scoring_fn = env.scoring_fn  # type: ignore
        qd_offset = 1 + 2 * fit_std

        if delta_fitness == 0.0 and delta_reproducibility == 0.0:
            delta_fitness = 1.0
            delta_reproducibility = 0.0

        def ground_truth_scoring_fn(
            params: Genotype, random_key: RNGKey
        ) -> Tuple[Fitness, Fitness, Descriptor, Fitness, ExtraScores, RNGKey]:

            fitnesses, descriptors, extra_scores, random_key = arm_scoring_function(
                params, random_key
            )

            fitnesses_var = fit_std * jnp.ones_like(fitnesses)
            descriptors_var = jnp.zeros_like(fitnesses)
            return (
                fitnesses,
                fitnesses_var,
                descriptors,
                descriptors_var,
                extra_scores,
                random_key,
            )

        def init_policies_fn(
            size: int, random_key: RNGKey
        ) -> Tuple[jnp.ndarray, RNGKey]:
            random_key, subkey = jax.random.split(random_key)
            init_policies = jax.random.uniform(
                random_key,
                shape=(size, policy_hidden_layer_sizes[0]),
                minval=0,
                maxval=1,
            )
            return init_policies, random_key

    # Arm with Gaussian fit noise
    elif env_name == "arm_gaussian_fit":

        if fit_std <= 0:
            print("\n!!!WARNING!!! Invalid std, using default values.")
            fit_std = 0.1

        print(f"Using Gaussian noise on fitness with var {fit_std}")

        params_std = 0
        desc_std = 0
        env = ArmGaussianNoise(
            fit_std=fit_std,
            desc_std=0.0,
            params_std=0.0,
        )
        scoring_fn = env.scoring_fn  # type: ignore
        qd_offset = 1 + 2 * fit_std

        if delta_fitness == 0.0 and delta_reproducibility == 0.0:
            delta_fitness = 1.0
            delta_reproducibility = 0.0

        def ground_truth_scoring_fn(
            params: Genotype, random_key: RNGKey
        ) -> Tuple[Fitness, Fitness, Descriptor, Fitness, ExtraScores, RNGKey]:

            fitnesses, descriptors, extra_scores, random_key = arm_scoring_function(
                params, random_key
            )

            fitnesses_var = fit_std * jnp.ones_like(fitnesses)
            descriptors_var = jnp.zeros_like(fitnesses)
            return (
                fitnesses,
                fitnesses_var,
                descriptors,
                descriptors_var,
                extra_scores,
                random_key,
            )

        def init_policies_fn(
            size: int, random_key: RNGKey
        ) -> Tuple[jnp.ndarray, RNGKey]:
            random_key, subkey = jax.random.split(random_key)
            init_policies = jax.random.uniform(
                random_key,
                shape=(size, policy_hidden_layer_sizes[0]),
                minval=0,
                maxval=1,
            )
            return init_policies, random_key

    # Arm with Gaussian desc noise
    elif env_name == "arm_gaussian_desc":

        if desc_std <= 0:
            print("\n!!!WARNING!!! Invalid std, using default values.")
            desc_std = 0.01

        print(f"Using Gaussian noise on des with var {desc_std}")

        params_std = 0
        fit_std = 0
        env = ArmGaussianNoise(
            fit_std=0.0,
            desc_std=desc_std,
            params_std=0.0,
        )
        scoring_fn = env.scoring_fn  # type: ignore
        qd_offset = 1

        if delta_fitness == 0.0 and delta_reproducibility == 0.0:
            delta_fitness = 1.0
            delta_reproducibility = 0.0

        def ground_truth_scoring_fn(
            params: Genotype, random_key: RNGKey
        ) -> Tuple[Fitness, Fitness, Descriptor, Fitness, ExtraScores, RNGKey]:

            fitnesses, descriptors, extra_scores, random_key = arm_scoring_function(
                params, random_key
            )
            fitnesses_var = jnp.zeros_like(fitnesses)
            descriptors_var = desc_std * jnp.ones_like(fitnesses)
            return (
                fitnesses,
                fitnesses_var,
                descriptors,
                descriptors_var,
                extra_scores,
                random_key,
            )

        def init_policies_fn(
            size: int, random_key: RNGKey
        ) -> Tuple[jnp.ndarray, RNGKey]:
            random_key, subkey = jax.random.split(random_key)
            init_policies = jax.random.uniform(
                random_key,
                shape=(size, policy_hidden_layer_sizes[0]),
                minval=0,
                maxval=1,
            )
            return init_policies, random_key

    # Arm with Multimodal fit noise
    elif env_name == "arm_multi_modal_fit":

        if fit_std <= 0 or desc_std <= 0:
            print("\n!!!WARNING!!! Invalid std, using default values.")
            fit_std = 0.01
            desc_std = 0.01

        # Values from UQD Benchmark paper to simplify
        proba_mode_1_fit = 0.85
        mean_fitness_2 = -1
        print("Using multi-modal Gaussian noise on fitness with")
        print(f"proba_mode_1 {proba_mode_1_fit}")
        print(f"var {fit_std}")
        print(f"mean_fitness_2 {mean_fitness_2}")

        params_std = 0
        env = ArmBimodalGaussianFitness(  # type: ignore
            proba_mode_1=proba_mode_1_fit,
            fit_std_1=fit_std,
            fit_std_2=desc_std,
            mean_fitness_2=mean_fitness_2,
        )
        scoring_fn = env.scoring_fn  # type: ignore
        qd_offset = 1 - mean_fitness_2 + 2 * fit_std

        if delta_fitness == 0.0 and delta_reproducibility == 0.0:
            delta_fitness = 1.0
            delta_reproducibility = 0.0

        def ground_truth_scoring_fn(
            params: Genotype, random_key: RNGKey
        ) -> Tuple[Fitness, Fitness, Descriptor, Fitness, ExtraScores, RNGKey]:

            fitnesses, descriptors, extra_scores, random_key = arm_scoring_function(
                params, random_key
            )
            fitnesses_var = fit_std * jnp.ones_like(fitnesses)
            descriptors_var = jnp.zeros_like(fitnesses)
            return (
                fitnesses,
                fitnesses_var,
                descriptors,
                descriptors_var,
                extra_scores,
                random_key,
            )

        def init_policies_fn(
            size: int, random_key: RNGKey
        ) -> Tuple[jnp.ndarray, RNGKey]:
            random_key, subkey = jax.random.split(random_key)
            init_policies = jax.random.uniform(
                random_key,
                shape=(size, policy_hidden_layer_sizes[0]),
                minval=0,
                maxval=1,
            )
            return init_policies, random_key

    # Arm with Multimodal desc noise
    elif env_name == "arm_multi_modal_desc":

        if fit_std <= 0 or desc_std <= 0:
            print("\n!!!WARNING!!! Invalid std, using default values.")
            fit_std = 0.01
            desc_std = 0.01

        # Values from UQD Benchmark paper to simplify
        params_std = 0
        proba_mode_1_desc = 0.85
        mean_desc_2 = [1.0, 1.0]
        desc_std_1 = [desc_std, desc_std]
        desc_std_2 = [fit_std, fit_std]
        print("Using multi-modal Gaussian noise on descriptor")
        print(f"proba_mode_1 {proba_mode_1_desc}")
        print(f"var {desc_std_1}")
        print(f"mean_desc_2 {mean_desc_2}")
        print(f"var {desc_std_2}")

        if delta_fitness == 0.0 and delta_reproducibility == 0.0:
            delta_fitness = 1.0
            delta_reproducibility = 0.0

        def ground_truth_scoring_fn(
            params: Genotype, random_key: RNGKey
        ) -> Tuple[Fitness, Fitness, Descriptor, Fitness, ExtraScores, RNGKey]:

            fitnesses, descriptors, extra_scores, random_key = arm_scoring_function(
                params, random_key
            )
            fitnesses_var = jnp.zeros_like(fitnesses)
            descriptors_var = desc_std * jnp.ones_like(fitnesses)
            return (
                fitnesses,
                fitnesses_var,
                descriptors,
                descriptors_var,
                extra_scores,
                random_key,
            )

        env = ArmBimodalGaussianDesc(  # type: ignore
            proba_mode_1=proba_mode_1_desc,
            desc_std_1=desc_std_1,
            desc_std_2=desc_std_2,
            mean_desc_2=mean_desc_2,
        )
        scoring_fn = env.scoring_fn  # type: ignore
        qd_offset = 1

        def init_policies_fn(
            size: int, random_key: RNGKey
        ) -> Tuple[jnp.ndarray, RNGKey]:
            random_key, subkey = jax.random.split(random_key)
            init_policies = jax.random.uniform(
                random_key,
                shape=(size, policy_hidden_layer_sizes[0]),
                minval=0,
                maxval=1,
            )
            return init_policies, random_key

    # Arm with Gaussian param noise
    elif env_name == "arm_selected_gaussian_params":

        if params_std <= 0:
            print("\n!!!WARNING!!! Invalid std, using default values.")
            params_std = 0.1

        # Values from UQD Benchmark paper to simplify
        selected_indexes_noise = [6]
        print("Using Gaussian noise on selected params")
        print(f"noise with var {params_std} on indexes {selected_indexes_noise}")

        desc_std = 0
        fit_std = 0
        env = ArmSelectedJointsNoise(  # type: ignore
            selected_indexes=jnp.asarray(selected_indexes_noise),
            params_std=params_std,
        )
        scoring_fn = env.scoring_fn  # type: ignore
        qd_offset = 5
        ground_truth_scoring_fn = None  # type: ignore

        if delta_fitness == 0.0 and delta_reproducibility == 0.0:
            delta_fitness = 1.0
            delta_reproducibility = 0.0

        def init_policies_fn(
            size: int, random_key: RNGKey
        ) -> Tuple[jnp.ndarray, RNGKey]:
            random_key, subkey = jax.random.split(random_key)
            init_policies = jax.random.uniform(
                random_key,
                shape=(size, policy_hidden_layer_sizes[0]),
                minval=0,
                maxval=1,
            )
            return init_policies, random_key

    # Arm with bi-std Gaussian desc noise
    elif env_name == "arm_gaussian_desc_bi_variance":

        if fit_std <= 0 or desc_std <= 0:
            print("\n!!!WARNING!!! Invalid std, using default values.")
            fit_std = 0.01
            desc_std = 0.1

        desc_std_1 = [fit_std, fit_std]
        desc_std_2 = [desc_std, desc_std]
        print("Using Gaussian noise on descriptor with 2 choices of std")
        print(f"{desc_std_1} and {desc_std_2}")
        print("Using fitness of 0 for this task")

        if delta_fitness == 0.0 and delta_reproducibility == 0.0:
            delta_fitness = 0.0
            delta_reproducibility = 0.2

        params_std = 0
        env = ArmGaussianDescBiVarianceNoise(  # type: ignore
            desc_std_1=desc_std_1,
            desc_std_2=desc_std_2,
        )
        scoring_fn = env.scoring_fn  # type: ignore
        qd_offset = 0

        def ground_truth_scoring_fn(
            params: Genotype, random_key: RNGKey
        ) -> Tuple[Fitness, Fitness, Descriptor, Fitness, ExtraScores, RNGKey]:

            fitnesses, descriptors, extra_scores, random_key = arm_scoring_function(
                params, random_key
            )
            fitnesses = jnp.zeros_like(fitnesses)
            fitnesses_var = jnp.zeros_like(fitnesses)
            descriptors_var = (
                jax.vmap(env.get_std)(params, fitnesses, descriptors).at[:, 0].get()
            )
            return (
                fitnesses,
                fitnesses_var,
                descriptors,
                descriptors_var,
                extra_scores,
                random_key,
            )

        def init_policies_fn(
            size: int, random_key: RNGKey
        ) -> Tuple[jnp.ndarray, RNGKey]:
            random_key, subkey = jax.random.split(random_key)
            init_policies = jax.random.uniform(
                random_key,
                shape=(size, policy_hidden_layer_sizes[0]),
                minval=0,
                maxval=1,
            )
            return init_policies, random_key

    # Arm with prop-std Gaussian desc noise
    elif env_name == "arm_gaussian_desc_fitprop_variance":

        # Values from UQD Benchmark paper to simplify
        prop_factors = [0.1, 0.1]
        print("Using Gaussian noise on descriptor with fitness-proportional std")
        print(f"prop_factors {prop_factors}")
        print("Using fitness of 0 for this task")

        params_std = 0
        env = ArmGaussianDescFitPropVarianceNoise(prop_factors=prop_factors)  # type: ignore
        scoring_fn = env.scoring_fn  # type: ignore
        qd_offset = 0

        if delta_fitness == 0.0 and delta_reproducibility == 0.0:
            delta_fitness = 0.0
            delta_reproducibility = 0.2

        def ground_truth_scoring_fn(
            params: Genotype, random_key: RNGKey
        ) -> Tuple[Fitness, Fitness, Descriptor, Fitness, ExtraScores, RNGKey]:

            fitnesses, descriptors, extra_scores, random_key = arm_scoring_function(
                params, random_key
            )
            fitnesses = jnp.zeros_like(fitnesses)
            fitnesses_var = jnp.zeros_like(fitnesses)
            descriptors_var = (
                jax.vmap(env.get_std)(params, fitnesses, descriptors).at[:, 0].get()
            )
            return (
                fitnesses,
                fitnesses_var,
                descriptors,
                descriptors_var,
                extra_scores,
                random_key,
            )

        def init_policies_fn(
            size: int, random_key: RNGKey
        ) -> Tuple[jnp.ndarray, RNGKey]:
            random_key, subkey = jax.random.split(random_key)
            init_policies = jax.random.uniform(
                random_key,
                shape=(size, policy_hidden_layer_sizes[0]),
                minval=0,
                maxval=1,
            )
            return init_policies, random_key

    else:
        print("!!!ERROR!!! Cannot find this env in optimisation.")
        assert 0

    # Build reeval functions
    if params_std == 0:
        reevaluation_function_fn = partial(
            reevaluation_function_ground_truth,
            batch_size=batch_size,
            extra_scores_extractor=dummy_extra_scores_extractor,
            ground_truth_scoring_fn=ground_truth_scoring_fn,
        )
        incell_reevaluation_function_fn = partial(
            incell_reevaluation_function_ground_truth,
            batch_size=batch_size,
            extra_scores_extractor=dummy_extra_scores_extractor,
            ground_truth_scoring_fn=ground_truth_scoring_fn,
        )
    else:
        reevaluation_function_fn = reevaluation_function
        incell_reevaluation_function_fn = incell_reevaluation_function

    # Get number descriptor dimensions
    num_descriptors = 2

    # Get min and max bd
    min_bd = jnp.array([0.0, 0.0])
    max_bd = jnp.array([1.0, 1.0])

    # Get min and max genotypes
    hard_limit_genotype = True
    min_genotype = 0
    max_genotype = 1

    # Add noise values to name
    new_env_name = env_name
    if fit_std == 0 and desc_std == 0 and params_std == 0:
        new_env_name += "_nonoise"
    else:
        new_env_name += f"_fit{fit_std}" f"_desc{desc_std}" f"_params{params_std}"

    # Return of optimisation env
    return (
        new_env_name,
        None,
        None,
        None,
        scoring_fn,
        reevaluation_function_fn,
        incell_reevaluation_function_fn,
        None,
        None,
        init_policies_fn,
        hard_limit_genotype,
        min_genotype,
        max_genotype,
        min_bd,
        max_bd,
        qd_offset,
        num_descriptors,
        delta_fitness,
        delta_reproducibility,
        random_key,
    )
