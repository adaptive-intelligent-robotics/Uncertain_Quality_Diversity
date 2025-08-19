import argparse
import os
import time
from functools import reduce
from math import ceil

import jax
import jax.numpy as jnp
import numpy as np
from numpy.random import randint

from core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    compute_cvt_centroids,
    compute_euclidean_centroids,
)
from environments_manager.set_up_environment import (
    ENV_RL_LIST,
    ENV_TIMESTEP_LIST,
    set_up_environment,
)
from metrics_manager.constructor import get_metrics_manager
from set_up_container import get_batch_size, get_sampling_size, set_up_container
from set_up_emitter import get_evals_per_offspring, set_up_emitter

# Limit CPU usage for HPC
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false " "intra_op_parallelism_threads=4"
)

# Uncomment this to debug if time is spent rejiting functions.
# import logging
# logging.basicConfig(level=logging.DEBUG)


############
# 0. Input #
############

parser = argparse.ArgumentParser()

# Run
parser.add_argument("--results", default="results", type=str)
parser.add_argument("--suffixe", default="", type=str)
parser.add_argument("--seed", default=0, type=int, help="Sampled if 0.")
parser.add_argument("--deterministic", action="store_true")

# Metrics
parser.add_argument("--log-period", default=50, type=int)
parser.add_argument("--archive-log-period", default=500, type=int)
parser.add_argument("--size-log-period", action="store_true")
parser.add_argument("--num-reevals", default=512, type=int)
parser.add_argument("--reeval-scan-size", default=0, type=int, help="Not used if 0.")
parser.add_argument("--reeval-fitness-extractor", default="Average", type=str)
parser.add_argument("--reeval-lighter", action="store_true")
parser.add_argument(
    "--reeval-fitness-reproducibility-extractor", default="STD", type=str
)
parser.add_argument("--reeval-descriptor-extractor", default="Average", type=str)
parser.add_argument(
    "--reeval-descriptor-reproducibility-extractor", default="STD", type=str
)

# Stopping criterion
parser.add_argument("--num-iterations", default=0, type=int)
parser.add_argument("--num-evaluations", default=0, type=int)
parser.add_argument("--num-timesteps", default=0, type=int)

# Compare size
parser.add_argument("--batch-size", default=0, type=int)
parser.add_argument("--sampling-size", default=0, type=int)

# Environment
parser.add_argument("--env-name", default="arm_gaussian_fit", type=str)
parser.add_argument("--episode-length", default=1000, type=int)
parser.add_argument("--fit-std", default=0.0, type=float)
parser.add_argument("--desc-std", default=0.0, type=float)
parser.add_argument("--params-std", default=0.0, type=float)
parser.add_argument("--policy-hidden-layer-sizes", default="8", type=str)

# Archive
parser.add_argument("--num-centroids", default=1024, type=int)
parser.add_argument("--num-init-cvt-samples", default=50000, type=int)
parser.add_argument("--euclidean-centroids", action="store_true")
parser.add_argument("--euclidean-grid-shape", default="5_5_5_5_5_5", type=str)

# Algorithm
parser.add_argument("--container", default="MAP-Elites", type=str)
parser.add_argument("--emitter", default="Mixing", type=str)
parser.add_argument("--emitter-batch", default=0, type=int, help="No effect if 0.")
parser.add_argument("--num-samples", default=1, type=int)
parser.add_argument("--fitness-extractor", default="Average", type=str)
parser.add_argument("--fitness-reproducibility-extractor", default="STD", type=str)
parser.add_argument("--descriptor-extractor", default="Average", type=str)
parser.add_argument("--descriptor-reproducibility-extractor", default="STD", type=str)
parser.add_argument("--depth", default=1, type=int)
parser.add_argument(
    "--max-number-evals",
    default=500,
    type=int,
    help="4th dimension of the grid (number of resamples) to be jax-compatible",
)

# Mutation
parser.add_argument("--mutation", default="isoline", type=str)
parser.add_argument("--iso-sigma", default=0.005, type=float)  # 0.05
parser.add_argument("--line-sigma", default=0.05, type=float)  # 0.1
parser.add_argument("--proportion-mutation", default=0.1, type=float)
parser.add_argument("--eta", default=10, type=float)

# Archive-Sampling family
parser.add_argument("--as-repertoire-num-samples", default=1, type=int)
parser.add_argument("--pas-max-samples", default=0, type=int)
parser.add_argument("--pas-use-evals", default="median", type=str)

# Extract-ME family
parser.add_argument("--extract-proportion-resample", default=0.25, type=float)
parser.add_argument("--extract-cap-resample", default=2048, type=int)
parser.add_argument("--extract-type", default="proportional", type=str)

# PG family
parser.add_argument("--pg-num-critic-training-steps", default=5000, type=int)
parser.add_argument("--pg-replay-buffer-size", default=1000000, type=int)
parser.add_argument("--pg-critic-hidden-layer-sizes", default="128_128", type=str)

# MER family
parser.add_argument("--mer-delta-fitness", default=0.0, type=float)
parser.add_argument("--mer-delta-reproducibility", default=0.0, type=float)
parser.add_argument("--mer-rho", default=0.0001, type=float)

# MOME family
parser.add_argument("--mome-max-pareto-front-length", default=50, type=int)
parser.add_argument("--mome-biased-sampling", action="store_true")
parser.add_argument("--mome-projected-delta-fitness", default="fit", type=str)
parser.add_argument("--mome-projected-delta-reproducibility", default="fit", type=str)

args = parser.parse_args()

##################
# 0. Input check #
##################

# Batch-size / sampling-size
assert not (
    args.batch_size != 0 and args.sampling_size != 0
), "\n!!!ERROR!!! Cannot use both batch_size and sampling_size."
assert not (
    args.batch_size == 0 and args.sampling_size == 0
), "\n!!!ERROR!!! No --sampling-size nor --batch-size."

# Stopping criterion
assert not (
    (args.num_iterations != 0 and args.num_evaluations != 0)
    or (args.num_iterations != 0 and args.num_timesteps != 0)
    or (args.num_timesteps != 0 and args.num_evaluations != 0)
), "\n!!!ERROR!!! Multiple stopping criterions."
assert not (
    args.num_iterations == 0 and args.num_evaluations == 0 and args.num_timesteps == 0
), "\n!!!ERROR!!! No stopping criterion."

# Some additional edge cases
if args.emitter == "PGA" or args.emitter == "DCG" or args.emitter == "QDPG":
    assert args.env_name in ENV_RL_LIST, "\n!!!ERROR!!! QD-RL only for RL tasks."

####################
# I. Configuration #
####################

# Set random seed
args.seed = randint(1000000) if args.seed == 0 else args.seed

# Process grid structure
if args.euclidean_centroids:
    grid_shape = tuple([int(x) for x in args.euclidean_grid_shape.split("_")])
    args.num_centroids = reduce(lambda x, y: x * y, grid_shape)
    print(
        f"Using euclidean centroids with grid-shape {grid_shape} and num_centroids {args.num_centroids}."
    )

# Process NN structure
args.policy_hidden_layer_sizes = tuple(
    [int(x) for x in args.policy_hidden_layer_sizes.split("_")]
)
args.pg_critic_hidden_layer_sizes = tuple(
    [int(x) for x in args.pg_critic_hidden_layer_sizes.split("_")]
)

# Compute needed evaluations in each iter
evals_per_offspring = get_evals_per_offspring(args=args)
if args.sampling_size != 0:
    # Compute batch_size from sampling_size
    (
        args.batch_size,
        args.init_batch_size,
        args.effective_batch_size,
        args.real_evals_per_iter,
    ) = get_batch_size(
        sampling_size=args.sampling_size,
        evals_per_offspring=evals_per_offspring,
        args=args,
    )
else:
    # Compute sampling_size from batch_size
    (
        args.sampling_size,
        args.init_batch_size,
        args.effective_batch_size,
        args.real_evals_per_iter,
    ) = get_sampling_size(
        batch_size=args.batch_size,
        evals_per_offspring=evals_per_offspring,
        args=args,
    )
print(f"Using batch-size: {args.batch_size} and sampling-size: {args.sampling_size}.")
print(f"With real_evals_per_iter: {args.real_evals_per_iter}")

# Compute number of timesteps per evaluations
timesteps_per_eval = 1
if args.env_name in ENV_TIMESTEP_LIST:
    timesteps_per_eval = args.episode_length

# Compute run length from num_evaluations
if args.num_evaluations > 0:
    args.num_iterations = args.num_evaluations // args.sampling_size
elif args.num_timesteps > 0:
    assert (
        args.env_name in ENV_TIMESTEP_LIST
    ), "!!!ERROR!!! Iterations as stopping criterion only for timesteps-based tasks."
    args.num_iterations = (
        args.num_timesteps // timesteps_per_eval
    ) // args.sampling_size

# Compute log period from compare size
if args.size_log_period:
    print("\n!!!WARNING!!! Assuming given log_period is for 256, 3-root decay:")
    compare_size = args.sampling_size if args.sampling_size > 0 else args.batch_size
    factor = pow(compare_size / 256, 1 / 3) if compare_size > 256 else 1
    args.log_period = max(1, ceil(args.log_period / factor))
    args.archive_log_period = max(1, ceil(args.archive_log_period / factor))
    print(
        f"Using log_period {args.log_period} and archive log_period {args.archive_log_period}."
    )

# Compute log period from timestep evaluation
log_period_ratio = 1
args.log_period = args.log_period * log_period_ratio
args.archive_log_period = args.archive_log_period * log_period_ratio
print(
    f"Using log_period {args.log_period} and archive log_period {args.archive_log_period}."
)

######################
# II. Initialisation #
######################

print("\n\nEntering initialisation.\n")
step_t = time.time()

# Init a random key
np.random.seed(args.seed)
random_key = jax.random.PRNGKey(args.seed)

# Set up the environment
(
    args.env_name,
    env,
    play_reset_fn,
    play_step_fn,
    scoring_fn,
    reevaluation_fn,
    reevaluation_in_cell_fn,
    policy_structure,
    policy_dc_structure,
    init_policies_fn,
    hard_limit_genotype,
    min_genotype,
    max_genotype,
    min_bd,
    max_bd,
    qd_offset,
    num_descriptors,
    new_mer_delta_fitness,
    new_mer_delta_reproducibility,
    random_key,
) = set_up_environment(
    deterministic=args.deterministic,
    env_name=args.env_name,
    episode_length=args.episode_length,
    fit_std=args.fit_std,
    desc_std=args.desc_std,
    params_std=args.params_std,
    batch_size=args.batch_size,
    policy_hidden_layer_sizes=args.policy_hidden_layer_sizes,
    delta_fitness=args.mer_delta_fitness,
    delta_reproducibility=args.mer_delta_reproducibility,
    random_key=random_key,
)

# Add infos from env creation to args
if args.mer_delta_fitness == 0 and args.mer_delta_reproducibility == 0:
    args.mer_delta_fitness = new_mer_delta_fitness
    args.mer_delta_reproducibility = new_mer_delta_reproducibility
elif (
    args.mer_delta_fitness != new_mer_delta_fitness
    or args.mer_delta_reproducibility != new_mer_delta_reproducibility
):
    print(
        f"\n!!!WARNING!!! Not using task delta values: {new_mer_delta_fitness} and {new_mer_delta_reproducibility}."
    )
if (
    args.mome_projected_delta_fitness == "0.0"
    and args.mome_projected_delta_reproducibility == "0.0"
):
    args.mome_projected_delta_fitness = f"{new_mer_delta_fitness}"
    args.mome_projected_delta_reproducibility = f"{new_mer_delta_reproducibility}"
elif (
    args.mome_projected_delta_fitness != f"{new_mer_delta_fitness}"
    or args.mome_projected_delta_reproducibility != f"{new_mer_delta_reproducibility}"
):
    print(
        f"\n!!!WARNING!!! Not using task delta values: {new_mer_delta_fitness} and {new_mer_delta_reproducibility}."
    )
args.min_bd = min_bd
args.max_bd = max_bd
args.qd_offset = qd_offset

print("\n  -> Environment initialised.")

# Sample a set of initial solutions
init_policies, random_key = init_policies_fn(args.init_batch_size, random_key)
printing_size = jax.tree_util.tree_leaves(init_policies)
printing_size = sum([leaf.size for leaf in printing_size]) / args.init_batch_size
print(f"\nPolicy size: {printing_size}")
print(jax.tree_util.tree_map(lambda x: x.shape, init_policies))

# Set up the algo
emitter, random_key = set_up_emitter(
    emitter_name=args.emitter,
    num_iterations=args.num_iterations,
    effective_batch_size=args.effective_batch_size,
    env=env,
    num_descriptors=num_descriptors,
    num_centroids=args.num_centroids,
    hard_limit_genotype=hard_limit_genotype,
    min_genotype=min_genotype,
    max_genotype=max_genotype,
    policy_structure=policy_structure,
    policy_dc_structure=policy_dc_structure,
    init_policies=init_policies,
    mutation=args.mutation,
    iso_sigma=args.iso_sigma,
    line_sigma=args.line_sigma,
    proportion_mutation=args.proportion_mutation,
    eta=args.eta,
    pg_num_critic_training_steps=args.pg_num_critic_training_steps,
    pg_critic_hidden_layer_sizes=args.pg_critic_hidden_layer_sizes,
    pg_replay_buffer_size=args.pg_replay_buffer_size,
    random_key=random_key,
)
map_elites, random_key = set_up_container(
    container_name=args.container,
    emitter=emitter,
    scoring_fn=scoring_fn,
    batch_size=args.batch_size,
    effective_batch_size=args.effective_batch_size,
    sampling_size=args.sampling_size,
    num_samples=args.num_samples,
    num_descriptors=num_descriptors,
    depth=args.depth,
    max_number_evals=args.max_number_evals,
    extract_type=args.extract_type,
    as_repertoire_num_samples=args.as_repertoire_num_samples,
    pas_max_samples=args.pas_max_samples,
    pas_use_evals=args.pas_use_evals,
    mer_delta_fitness=args.mer_delta_fitness,
    mer_delta_reproducibility=args.mer_delta_reproducibility,
    mer_rho=args.mer_rho,
    mome_pareto_front_max_length=args.mome_max_pareto_front_length,
    mome_biased_sampling=args.mome_biased_sampling,
    fitness_extractor=args.fitness_extractor,
    fitness_reproducibility_extractor=args.fitness_reproducibility_extractor,
    descriptor_extractor=args.descriptor_extractor,
    descriptor_reproducibility_extractor=args.descriptor_reproducibility_extractor,
    random_key=random_key,
)
print("\n  -> Algorithm initialised.")

# Compute the centroids
if args.euclidean_centroids:
    centroids = compute_euclidean_centroids(
        grid_shape=grid_shape,
        minval=args.min_bd,
        maxval=args.max_bd,
    )
else:
    centroids, random_key = compute_cvt_centroids(
        num_descriptors=num_descriptors,
        num_init_cvt_samples=args.num_init_cvt_samples,
        num_centroids=args.num_centroids,
        minval=args.min_bd,
        maxval=args.max_bd,
        random_key=random_key,
    )
print("\n  -> Centroid initialised.")

# Set up the metric manager
name = args.container + "-" + args.emitter + args.suffixe
empty_metric_repertoire = MapElitesRepertoire.init(
    genotypes=init_policies,
    fitnesses=jnp.zeros(args.init_batch_size),
    descriptors=jnp.zeros((args.init_batch_size, num_descriptors)),
    extra_scores={},
    centroids=centroids,
)
metrics_manager = get_metrics_manager(
    args=args,
    name=name,
    scoring_fn=scoring_fn,
    reevaluation_fn=reevaluation_fn,
    reevaluation_in_cell_fn=reevaluation_in_cell_fn,
    metric_repertoire=empty_metric_repertoire,
)
print(f"\n  -> Metrics Manager initialised with name {name}.")

init_t = time.time() - step_t
print(f"\nFinished initialisation in {init_t} seconds.")

############
# III. Run #
############

print("\n\nEntering run.\n")
step_t = time.time()

# First iteration of the algorithm
repertoire, emitter_state, random_key = map_elites.init(
    init_policies, centroids, random_key
)
jax.tree_util.tree_map(
    lambda x: x.block_until_ready(), repertoire.genotypes
)  # ensure timing accuracy

# Initialise all counters
current_t = time.time() - step_t
total_metrics_t = 0.0
total_reeval_t = 0.0
total_write_t = 0.0
epoch = 0
evals = args.sampling_size
real_evals = args.sampling_size
timesteps = evals * timesteps_per_eval
real_timesteps = real_evals * timesteps_per_eval

# Compute and write initial metrics
metrics_t, reeval_t, write_t = metrics_manager.write_all_metrics(
    epoch=epoch,
    evals=evals,
    real_evals=real_evals,
    timesteps=timesteps,
    real_timesteps=real_timesteps,
    current_time=current_t,
    repertoire=repertoire,
    projected_repertoire=repertoire,
    emitter_state=emitter_state,
    random_key=random_key,
)
total_metrics_t += metrics_t
total_reeval_t += reeval_t
total_write_t += write_t

# main loop
while epoch < args.num_iterations:

    ########
    # Loop #

    (
        repertoire,
        emitter_state,
        _,
        random_key,
    ) = map_elites.update(repertoire, emitter_state, random_key)
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready(), repertoire.genotypes
    )  # ensure timing accuracy

    # Update all counters
    epoch += 1
    evals += args.sampling_size
    real_evals += args.real_evals_per_iter
    timesteps = evals * timesteps_per_eval
    real_timesteps = real_evals * timesteps_per_eval

    # These are the only algo that adapts live num_iterations
    # So take it into account for stopping criterion based on evaluations
    if (
        args.container == "Parallel-Adaptive-Sampling"
        or args.container == "Adaptive-Sampling"
    ):
        real_evals = repertoire.total_evaluations
        real_timesteps = real_evals * timesteps_per_eval

        # If using evaluation termination, check that it is not done
        if args.num_evaluations > 0:
            if real_evals >= args.num_evaluations:
                args.num_iterations = epoch

    ###########
    # Metrics #

    # Check log period
    if epoch % args.log_period != 0:
        continue

    # Compute effective running time (remove all metric time)
    current_t = time.time() - step_t - total_reeval_t - total_write_t - total_metrics_t
    print(
        f"\n    Epoch: {epoch} / {args.num_iterations}",
        f"-- Evals: {evals}",
        f"-- Real evals: {real_evals}",
        f"-- Timesteps: {timesteps}",
        f"-- Real Timesteps: {real_timesteps}",
        f"-- Time: {ceil(current_t)}",
        f"-- Total Time: {ceil(time.time() - step_t)}",
    )

    # Compute and write metrics
    metrics_t, reeval_t, write_t = metrics_manager.write_all_metrics(
        epoch=epoch,
        evals=evals,
        real_evals=real_evals,
        timesteps=timesteps,
        real_timesteps=real_timesteps,
        current_time=current_t,
        repertoire=repertoire,
        projected_repertoire=repertoire,
        emitter_state=emitter_state,
        random_key=random_key,
    )
    total_metrics_t += metrics_t
    total_reeval_t += reeval_t
    total_write_t += write_t


#################
# Final metrics #

# Compute effective running time (remove all metric time)
current_t = time.time() - step_t - total_reeval_t - total_write_t - total_metrics_t
print(
    f"\n    Ended at epoch: {epoch} / {args.num_iterations} -- evals: {evals} -- timesteps: {timesteps} -- time: {current_t}"
)

# Compute and write metrics
_, _, _ = metrics_manager.write_all_metrics(
    epoch=epoch,
    evals=evals,
    real_evals=real_evals,
    timesteps=timesteps,
    real_timesteps=real_timesteps,
    current_time=current_t,
    repertoire=repertoire,
    projected_repertoire=repertoire,
    emitter_state=emitter_state,
    random_key=random_key,
    final=True,
)

run_t = time.time() - step_t
print(f"\nFinished run in {run_t} seconds.")
