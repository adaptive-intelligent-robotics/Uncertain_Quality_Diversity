import logging
import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.4'
import pickle
import time
from dataclasses import dataclass
from functools import partial
from typing import Dict, Optional, Tuple

import hydra
import jax
import jax.numpy as jnp
from hydra.core.config_store import ConfigStore

from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.tasks.brax_envs import create_default_brax_task_components
from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids

from core.map_elites import MAPElites
from core.containers.mapelites_repertoire import MapElitesRepertoire
from core.stochasticity_utils import reevaluation_function
from core.sampling import sampling, sampling_reproducibility
from tasks.uqd_benchmark import (
    ArmBimodalGaussianDesc,
    ArmBimodalGaussianFitness,
    ArmGaussianNoise,
    ArmSelectedJointsNoise,
    ArmGaussianDescBiVarianceNoise,
    ArmGaussianDescFitPropVarianceNoise,
)


@dataclass
class ExperimentConfig:
    """Configuration from this experiment script"""

    alg_name: str

    # Env config
    seed: int
    env_name: str
    episode_length: int
    policy_hidden_layer_sizes: Tuple[int, ...]
    arm_dofs: int

    # ME config
    num_evaluations: int
    num_iterations: int
    batch_size: int
    fixed_init_state: bool
    discard_dead: bool

    # Grid config
    grid_shape: Tuple[int, ...]

    # Emitter config
    iso_sigma: float
    line_sigma: float
    crossover_percentage: float

    # Log config
    log_period: int
    store_repertoire: bool
    store_repertoire_log_period: int

    # Noise config
    noise_type: str 

    # Stochasticity config
    fit_variance: float
    desc_variance: float
    num_reevals: int
    use_median: bool
    log_period_reevals: int

    # Multi-modal fitness noise config
    proba_mode_1_fit: float
    fit_variance_1: float
    fit_variance_2: float
    mean_fitness_2: float

    # Multi-modal descriptor noise config
    proba_mode_1_desc: float
    desc_variances_1: Tuple[float, ...]
    desc_variances_2: Tuple[float, ...]
    mean_desc_2: Tuple[float, ...]

    # Selected Params noise config
    params_variance: float
    selected_indexes_noise: Tuple[int, ...]

    # Variance-proportional Gaussian noise
    prop_factors: Tuple[float, ...]

    # Naive sampling config (equals None for ME only)
    num_samples: Optional[int] = None


def get_env(config: ExperimentConfig):
    noise_type = config.noise_type
    if noise_type == "gaussian_fit":
        print("Using Gaussian noise on fitness")
        env = ArmGaussianNoise(
            fit_variance=config.fit_variance,
            desc_variance=0.0,
            params_variance=0.0,
        )
    elif noise_type == "gaussian_desc":
        print("Using Gaussian noise on descriptor")
        env = ArmGaussianNoise(
            fit_variance=0.0,
            desc_variance=config.desc_variance,
            params_variance=0.0,
        )
    elif noise_type == "multi_modal_fit":
        print("Using multi-modal Gaussian noise on fitness")
        env = ArmBimodalGaussianFitness(
            proba_mode_1=config.proba_mode_1_fit,
            fit_variance_1=config.fit_variance_1,
            fit_variance_2=config.fit_variance_2,
            mean_fitness_2=config.mean_fitness_2,
        )
    elif noise_type == "multi_modal_desc":
        print("Using multi-modal Gaussian noise on descriptor")
        env = ArmBimodalGaussianDesc(
            proba_mode_1=config.proba_mode_1_desc,
            desc_variances_1=config.desc_variances_1,
            desc_variances_2=config.desc_variances_2,
            mean_desc_2=config.mean_desc_2,
        )
    elif noise_type == "selected_gaussian_params":
        print("Using Gaussian noise on selected params")
        env = ArmSelectedJointsNoise(
            selected_indexes=jnp.asarray(config.selected_indexes_noise),
            params_variance=config.params_variance,
        )
    elif noise_type == "gaussian_desc_bi_variance":
        print("Using Gaussian noise on descritpor with 2 choices of variance")
        print("Using fitness of 0 for this task")
        env = ArmGaussianDescBiVarianceNoise(
            desc_variances_1=config.desc_variances_1,
            desc_variances_2=config.desc_variances_2,
        )
    elif noise_type == "gaussian_desc_fitprop_variance":
        print("Using Gaussian noise on descriptor with fitness-proportional variance")
        print("Using fitness of 0 for this task")
        env = ArmGaussianDescFitPropVarianceNoise(
            prop_factors=config.prop_factors,
        )
    else:
        raise ValueError(f"Noise type {noise_type} not supported")
    return env


@hydra.main(config_path="configs", config_name="config")
def train(config: ExperimentConfig) -> None:

    # Setup logging
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().handlers[0].setLevel(logging.INFO)
    logger = logging.getLogger(f"{__name__}")
    output_dir = "./"

    # Setup metrics checkpoint save
    _last_metrics_dir = os.path.join(output_dir, "checkpoints", "last_metrics")
    os.makedirs(_last_metrics_dir, exist_ok=True)

    # Setup repertoire checkpoint save
    _last_grid_dir = os.path.join(output_dir, "checkpoints", "last_grid")
    os.makedirs(_last_grid_dir, exist_ok=True)
    _last_reeval_grid_dir = os.path.join(output_dir, "checkpoints", "last_reeval_grid")
    os.makedirs(_last_reeval_grid_dir, exist_ok=True)
    _last_fit_reeval_grid_dir = os.path.join(
        output_dir, "checkpoints", "last_fit_reeval_grid"
    )
    os.makedirs(_last_fit_reeval_grid_dir, exist_ok=True)
    _last_desc_reeval_grid_dir = os.path.join(
        output_dir, "checkpoints", "last_desc_reeval_grid"
    )
    os.makedirs(_last_desc_reeval_grid_dir, exist_ok=True)
    _last_fit_var_grid_dir = os.path.join(
        output_dir, "checkpoints", "last_fit_var_grid"
    )
    os.makedirs(_last_fit_var_grid_dir, exist_ok=True)
    _last_desc_var_grid_dir = os.path.join(
        output_dir, "checkpoints", "last_desc_var_grid"
    )
    os.makedirs(_last_desc_var_grid_dir, exist_ok=True)

    # Choose stopping criteria
    if config.num_iterations > 0 and config.num_evaluations > 0:
        print(
            "!!!WARNING!!! Both num_iterations and num_evaluations are set",
            "choosing num_iterations over num_evaluations",
        )
    if config.num_iterations > 0:
        num_iterations = config.num_iterations
    elif config.num_evaluations > 0:
        num_iterations = config.num_evaluations // config.batch_size + 1

    # Init a random key
    random_key = jax.random.PRNGKey(config.seed)

    # Init environment and population of controllers
    print("Env name: ", config.env_name)
    if config.env_name == "arm":
        env = get_env(config)
        random_key, subkey = jax.random.split(random_key)
        init_variables = jax.random.uniform(
            random_key, shape=(config.batch_size, config.arm_dofs), minval=0, maxval=1
        )
        scoring_fn = env.scoring_fn

    else:
        (
            env,
            policy_network,
            scoring_fn,
            random_key,
        ) = create_default_brax_task_components(
            config.env_name,
            random_key,
            config.episode_length,
            config.policy_hidden_layer_sizes,
            deterministic=config.fixed_init_state,
        )
        random_key, subkey = jax.random.split(random_key)
        keys = jax.random.split(subkey, num=config.batch_size)
        fake_batch = jnp.zeros(shape=(config.batch_size, env.observation_size))
        init_variables = jax.vmap(policy_network.init)(keys, fake_batch)

    # Compute the centroids
    logger.warning("--- Compute the CVT centroids ---")
    minval, maxval = env.behavior_descriptor_limits
    init_time = time.time()
    centroids = compute_euclidean_centroids(
        grid_shape=config.grid_shape,
        minval=minval,
        maxval=maxval,
    )
    duration = time.time() - init_time
    logger.warning(f"--- Duration for CVT centroids computation : {duration:.2f}s")

    # Define emitter
    variation_fn = partial(
        isoline_variation, iso_sigma=config.iso_sigma, line_sigma=config.line_sigma
    )
    mixing_emitter = MixingEmitter(
        mutation_fn=None,
        variation_fn=variation_fn,
        variation_percentage=1.0,
        batch_size=config.batch_size,
    )

    # Define a metrics function
    def metrics_fn(repertoire: MapElitesRepertoire) -> Dict:
        grid_empty = repertoire.fitnesses == -jnp.inf
        qd_score = jnp.sum(repertoire.fitnesses, where=~grid_empty)
        coverage = 100 * jnp.mean(1.0 - grid_empty)
        max_fitness = jnp.max(repertoire.fitnesses)
        min_fitness = jnp.min(
            jnp.where(repertoire.fitnesses > -jnp.inf, repertoire.fitnesses, jnp.inf)
        )
        return {
            "qd_score": jnp.array([qd_score]),
            "max_fitness": jnp.array([max_fitness]),
            "min_fitness": jnp.array([min_fitness]),
            "coverage": jnp.array([coverage]),
        }

    # Define a reeval metrics function
    metric_repertoire = MapElitesRepertoire.init(
        genotypes=init_variables,
        fitnesses=jnp.zeros(config.batch_size),
        descriptors=jnp.zeros((config.batch_size, env.behavior_descriptor_length)),
        extra_scores={},
        centroids=centroids,
    )

    def reeval_metrics_fn(
        reeval_repertoire: MapElitesRepertoire,
        fit_reeval_repertoire: MapElitesRepertoire,
        desc_reeval_repertoire: MapElitesRepertoire,
        fit_var_repertoire: MapElitesRepertoire,
        desc_var_repertoire: MapElitesRepertoire,
    ) -> Dict:
        reeval_metrics = metrics_fn(reeval_repertoire)
        fit_reeval_metrics = metrics_fn(fit_reeval_repertoire)
        desc_reeval_metrics = metrics_fn(desc_reeval_repertoire)
        fit_var_metrics = metrics_fn(fit_var_repertoire)
        desc_var_metrics = metrics_fn(desc_var_repertoire)
        return {
            "reeval_qd_score": reeval_metrics["qd_score"],
            "reeval_max_fitness": reeval_metrics["max_fitness"],
            "reeval_min_fitness": reeval_metrics["min_fitness"],
            "reeval_coverage": reeval_metrics["coverage"],
            "fit_reeval_qd_score": fit_reeval_metrics["qd_score"],
            "fit_reeval_max_fitness": fit_reeval_metrics["max_fitness"],
            "fit_reeval_min_fitness": fit_reeval_metrics["min_fitness"],
            "fit_reeval_coverage": fit_reeval_metrics["coverage"],
            "desc_reeval_qd_score": desc_reeval_metrics["qd_score"],
            "desc_reeval_max_fitness": desc_reeval_metrics["max_fitness"],
            "desc_reeval_min_fitness": desc_reeval_metrics["min_fitness"],
            "desc_reeval_coverage": desc_reeval_metrics["coverage"],
            "fit_var_qd_score": fit_var_metrics["qd_score"],
            "fit_var_max_fitness": fit_var_metrics["max_fitness"],
            "fit_var_min_fitness": fit_var_metrics["min_fitness"],
            "fit_var_coverage": fit_var_metrics["coverage"],
            "desc_var_qd_score": desc_var_metrics["qd_score"],
            "desc_var_max_fitness": desc_var_metrics["max_fitness"],
            "desc_var_min_fitness": desc_var_metrics["min_fitness"],
            "desc_var_coverage": desc_var_metrics["coverage"],
        }

    print("Algorithm Name: ", config.alg_name)
    if config.alg_name == "me":
        me_scoring_fn = scoring_fn
    elif config.alg_name == "naive":
        me_scoring_fn = partial(
            sampling,
            scoring_fn=scoring_fn,
            num_samples=config.num_samples,
        )
    elif config.alg_name == "naive_reproducibility":
        me_scoring_fn = partial(
            sampling_reproducibility,
            scoring_fn=scoring_fn,
            num_samples=config.num_samples,
        )
    else:
        raise ValueError("Unknown algorithm name")

    # Instantiate MAP-Elites
    map_elites = MAPElites(
        scoring_function=me_scoring_fn,
        emitter=mixing_emitter,
        metrics_function=metrics_fn,
    )

    # Init algorithm
    logger.warning("--- Algorithm initialisation ---")
    total_training_time = 0.0
    start_time = time.time()
    repertoire, emitter_state, random_key = map_elites.init(
        init_variables, centroids, random_key
    )
    init_time = time.time() - start_time
    total_training_time += init_time
    logger.warning("--- Initialised ---")
    logger.warning("--- Starting the algorithm main process ---")

    # Init metrics
    full_metrics = {
        "coverage": jnp.array([0.0]),
        "max_fitness": jnp.array([0.0]),
        "min_fitness": jnp.array([0.0]),
        "qd_score": jnp.array([0.0]),
    }
    full_reeval_metrics = {
        "reeval_coverage": jnp.array([0.0]),
        "reeval_max_fitness": jnp.array([0.0]),
        "reeval_min_fitness": jnp.array([0.0]),
        "reeval_qd_score": jnp.array([0.0]),
        "fit_reeval_coverage": jnp.array([0.0]),
        "fit_reeval_max_fitness": jnp.array([0.0]),
        "fit_reeval_min_fitness": jnp.array([0.0]),
        "fit_reeval_qd_score": jnp.array([0.0]),
        "desc_reeval_coverage": jnp.array([0.0]),
        "desc_reeval_max_fitness": jnp.array([0.0]),
        "desc_reeval_min_fitness": jnp.array([0.0]),
        "desc_reeval_qd_score": jnp.array([0.0]),
        "fit_var_coverage": jnp.array([0.0]),
        "fit_var_max_fitness": jnp.array([0.0]),
        "fit_var_min_fitness": jnp.array([0.0]),
        "fit_var_qd_score": jnp.array([0.0]),
        "desc_var_coverage": jnp.array([0.0]),
        "desc_var_max_fitness": jnp.array([0.0]),
        "desc_var_min_fitness": jnp.array([0.0]),
        "desc_var_qd_score": jnp.array([0.0]),
    }

    # Main QD Loop
    update_fn = jax.jit(partial(map_elites.update))
    for iteration in range(num_iterations):
        logger.warning(
            f"--- Iteration indice : {iteration} out of {num_iterations} ---"
        )

        start_time = time.time()
        (repertoire, emitter_state, metrics, random_key,) = update_fn(
            repertoire,
            emitter_state,
            random_key,
        )
        time_duration = time.time() - start_time
        total_training_time += time_duration

        logger.warning(f"--- Current QD Score: {metrics['qd_score'][-1]:.2f}")
        logger.warning(f"--- Current Coverage: {metrics['coverage'][-1]:.2f}%")
        logger.warning(f"--- Current Max Fitness: {metrics['max_fitness'][-1]}")

        # Save metrics
        full_metrics = {
            key: jnp.concatenate((full_metrics[key], metrics[key]))
            for key in full_metrics
        }

        # Write metrics
        if iteration % config.log_period == 0:
            with open(
                os.path.join(_last_metrics_dir, "metrics.pkl"), "wb"
            ) as file_to_save:
                pickle.dump(full_metrics, file_to_save)

        # Compute reeval metrics
        if config.num_reevals > 0 and iteration % config.log_period_reevals == 0:
            (
                reeval_repertoire,
                fit_reeval_repertoire,
                desc_reeval_repertoire,
                fit_var_repertoire,
                desc_var_repertoire,
                random_key,
            ) = reevaluation_function(
                repertoire=repertoire,
                random_key=random_key,
                metric_repertoire=metric_repertoire,
                scoring_fn=scoring_fn,
                num_reevals=config.num_reevals,
                use_median=config.use_median,
            )
            reeval_metrics = reeval_metrics_fn(
                reeval_repertoire,
                fit_reeval_repertoire,
                desc_reeval_repertoire,
                fit_var_repertoire,
                desc_var_repertoire,
            )

            logger.warning(
                f"--- Current Reeval QD Score: {reeval_metrics['reeval_qd_score'][-1]:.2f}"
            )
            logger.warning(
                f"--- Current Reeval Coverage: {reeval_metrics['reeval_coverage'][-1]:.2f}%"
            )
            logger.warning(
                f"--- Current Reeval Max Fitness: {reeval_metrics['reeval_max_fitness'][-1]}"
            )

            # Save reeval metrics
            full_reeval_metrics = {
                key: jnp.concatenate((full_reeval_metrics[key], reeval_metrics[key]))
                for key in full_reeval_metrics
            }
            with open(
                os.path.join(_last_metrics_dir, "reeval_metrics.pkl"), "wb"
            ) as file_to_save:
                pickle.dump(full_reeval_metrics, file_to_save)

            # Store the latest controllers of the reeval repertoires
            if (
                config.store_repertoire
                and iteration % config.store_repertoire_log_period == 0
            ):
                reeval_repertoire.save(path=_last_reeval_grid_dir + "/")
                fit_reeval_repertoire.save(path=_last_fit_reeval_grid_dir + "/")
                desc_reeval_repertoire.save(path=_last_desc_reeval_grid_dir + "/")
                fit_var_repertoire.save(path=_last_fit_var_grid_dir + "/")
                desc_var_repertoire.save(path=_last_desc_var_grid_dir + "/")

        # Store the latest controllers of the repertoire
        if (
            config.store_repertoire
            and iteration % config.store_repertoire_log_period == 0
        ):
            repertoire.save(path=_last_grid_dir + "/")

    duration = time.time() - init_time

    logger.warning("--- Final metrics ---")
    logger.warning(f"Duration: {duration:.2f}s")
    logger.warning(f"Training duration: {total_training_time:.2f}s")
    logger.warning(f"QD Score: {metrics['qd_score'][-1]:.2f}")
    logger.warning(f"Coverage: {metrics['coverage'][-1]:.2f}%")

    # Save final metrics
    with open(os.path.join(_last_metrics_dir, "metrics.pkl"), "wb") as file_to_save:
        pickle.dump(full_metrics, file_to_save)

    # Save final repertoire
    repertoire.save(path=_last_grid_dir + "/")

    # Reeval final repertoire
    if config.num_reevals > 0:
        (
            reeval_repertoire,
            fit_reeval_repertoire,
            desc_reeval_repertoire,
            fit_var_repertoire,
            desc_var_repertoire,
            random_key,
        ) = reevaluation_function(
            repertoire=repertoire,
            random_key=random_key,
            metric_repertoire=metric_repertoire,
            scoring_fn=scoring_fn,
            num_reevals=config.num_reevals,
            use_median=config.use_median,
        )
        reeval_metrics = reeval_metrics_fn(
            reeval_repertoire,
            fit_reeval_repertoire,
            desc_reeval_repertoire,
            fit_var_repertoire,
            desc_var_repertoire,
        )

        logger.warning(
            f"--- Reeval QD Score: {reeval_metrics['reeval_qd_score'][-1]:.2f}"
        )
        logger.warning(
            f"--- Reeval Coverage: {reeval_metrics['reeval_coverage'][-1]:.2f}%"
        )
        logger.warning(
            f"--- Reeval Max Fitness: {reeval_metrics['reeval_max_fitness'][-1]}"
        )

        # Save reeval metrics
        full_reeval_metrics = {
            key: jnp.concatenate((full_reeval_metrics[key], reeval_metrics[key]))
            for key in full_reeval_metrics
        }
        with open(
            os.path.join(_last_metrics_dir, "reeval_metrics.pkl"), "wb"
        ) as file_to_save:
            pickle.dump(full_reeval_metrics, file_to_save)

        # Store the latest controllers of the reeval repertoires
        reeval_repertoire.save(path=_last_reeval_grid_dir + "/")
        fit_reeval_repertoire.save(path=_last_fit_reeval_grid_dir + "/")
        desc_reeval_repertoire.save(path=_last_desc_reeval_grid_dir + "/")
        fit_var_repertoire.save(path=_last_fit_var_grid_dir + "/")
        desc_var_repertoire.save(path=_last_desc_var_grid_dir + "/")


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="validate_experiment_config", node=ExperimentConfig)
    train()
