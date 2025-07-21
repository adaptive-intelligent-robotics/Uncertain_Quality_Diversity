import argparse
import os
import time
from functools import partial
from math import ceil

import jax
import jax.numpy as jnp
import numpy as np
from numpy.random import randint

from core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    compute_cvt_centroids,
)
from core.sampling import (
    multi_sample_scoring_function,
    sampling,
    sampling_descriptor_reproducibility,
)
from metrics_manager.constructor import create_algo_name, get_metrics_manager
from set_up_container import (
    CONTAINER_NO_SAMPLES,
    CONTAINER_REEVAL_ARCHIVE,
    CONTAINER_REQUIRE_ALL_SAMPLES,
    EXTRACTOR_LIST,
    get_add_evals_per_iter,
    set_up_container,
)
from set_up_emitter import get_evals_per_offspring, set_up_emitter
from set_up_environment import set_up_environment

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

# Compare size
parser.add_argument("--batch-size", default=0, type=int)
parser.add_argument("--sampling-size", default=0, type=int)
parser.add_argument("--archive-out-sampling", action="store_true")

# Environment
parser.add_argument("--env-name", default="arm_gaussian_fit", type=str)
parser.add_argument("--episode-length", default=250, type=int)
parser.add_argument("--gaussian-vel", action="store_true")
parser.add_argument("--gaussian-pos", action="store_true")
parser.add_argument("--fit-std", default=0.0, type=float)
parser.add_argument("--desc-std", default=0.0, type=float)
parser.add_argument("--params-std", default=0.0, type=float)
parser.add_argument("--policy-hidden-layer-sizes", default="8", type=str)

# Archive
parser.add_argument("--num-centroids", default=1024, type=int)
parser.add_argument("--num-init-cvt-samples", default=50000, type=int)

# Algorithm
parser.add_argument("--container", default="MAP-Elites", type=str)
parser.add_argument("--emitter", default="Mixing", type=str)
parser.add_argument("--num-samples", default=0, type=int)
parser.add_argument("--fitness-extractor", default="Average", type=str)
parser.add_argument("--fitness-reproducibility-extractor", default="STD", type=str)
parser.add_argument("--descriptor-extractor", default="Average", type=str)
parser.add_argument("--descriptor-reproducibility-extractor", default="STD", type=str)
parser.add_argument("--depth", default=1, type=int)
parser.add_argument("--reproducibility-objective", action="store_true")

# Mutation
parser.add_argument("--mutation", default="isoline", type=str)
parser.add_argument("--iso-sigma", default=0.005, type=float)  # 0.05
parser.add_argument("--line-sigma", default=0.05, type=float)  # 0.1
parser.add_argument("--proportion-mutation", default=0.1, type=float)
parser.add_argument("--eta", default=10, type=float)

# Archive-Sampling family
parser.add_argument("--eas-max-samples", default=0, type=int)
parser.add_argument("--eas-use-evals", default="median", type=str)

# MER family
parser.add_argument("--mer-delta-fitness", default=0.01, type=float)
parser.add_argument("--mer-delta-reproducibility", default=0.01, type=float)
parser.add_argument("--mer-rho", default=0.0001, type=float)

# MOME family
parser.add_argument("--mome-max-pareto-front-length", default=50, type=int)
parser.add_argument("--mome-biased-sampling", default=True, type=bool)
parser.add_argument(
    "--mome-projected-delta-fitness", default="0.01", type=str
)  # 0.0_0.4_0.1
parser.add_argument(
    "--mome-projected-delta-reproducibility", default="0.01", type=str
)  # 0.4_0.0_0.1

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
if args.archive_out_sampling:
    if args.sampling_size == 0:
        args.archive_out_sampling = False
    else:
        print("\n!!!WARNING!!! Not considering archive evaluation in sampling budget.")

# Stopping criterion
assert not (
    args.num_iterations != 0 and args.num_evaluations != 0
), "\n!!!ERROR!!! Cannot use both num_iterations and num_evaluations."
assert not (
    args.num_iterations == 0 and args.num_evaluations == 0
), "\n!!!ERROR!!! No stopping criterion."

# Some additional edge cases
if args.reproducibility_objective:
    assert args.num_samples > 0, "\n!!!ERROR!!! reproducibility_obj require sampling."


####################
# I. Configuration #
####################

# Set random seed
args.seed = randint(1000000) if args.seed == 0 else args.seed

# Process policy structure
args.policy_hidden_layer_sizes = tuple(
    [int(x) for x in args.policy_hidden_layer_sizes.split("_")]
)

# Compute batch_size from sampling_size
evals_per_offspring = get_evals_per_offspring(args=args)
add_evals_per_iter = get_add_evals_per_iter(args=args)
if args.sampling_size != 0:
    assert args.sampling_size >= add_evals_per_iter + evals_per_offspring, (
        "!!!ERROR!!! Missing sampling credit for evaluation, got "
        + str(args.sampling_size)
        + " left, require at least "
        + str(evals_per_offspring)
        + " per offspring and "
        + str(add_evals_per_iter)
        + " for the rest (archive reevaluation, etc)."
    )
    left_sampling_size = args.sampling_size - add_evals_per_iter
    args.batch_size = int(left_sampling_size // evals_per_offspring)
    print(
        f"\nWith evals_per_offspring: {evals_per_offspring}, and add_evals_per_iter: {add_evals_per_iter}."
    )
    print(f"Using batch-size: {args.batch_size}.")
base_evals_per_iter = args.batch_size * evals_per_offspring

# Compute number of evals per iteration
evals_per_iter = base_evals_per_iter + add_evals_per_iter

# Compute run length from num_evaluations
if args.num_evaluations > 0:
    args.num_iterations = args.num_evaluations // evals_per_iter

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

# Create algo name for metrics and analysis files
name = create_algo_name(args)
print(f"\n\nRunning with name {name}")

######################
# II. Initialisation #
######################

print("\n\nEntering initialisation\n")
step_t = time.time()

# Init a random key
np.random.seed(args.seed)
random_key = jax.random.PRNGKey(args.seed)

# Set up the environment
(
    args.env_name,
    env,
    scoring_fn,
    policy_structure,
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
    gaussian_vel=args.gaussian_vel,
    gaussian_pos=args.gaussian_pos,
    delta_fitness=args.mer_delta_fitness,
    delta_reproducibility=args.mer_delta_reproducibility,
    random_key=random_key,
)
if (
    new_mer_delta_fitness != args.mer_delta_fitness
    or new_mer_delta_reproducibility != args.mer_delta_reproducibility
):
    args.mome_projected_delta_fitness = f"{new_mer_delta_fitness}"
    args.mome_projected_delta_reproducibility = f"{new_mer_delta_reproducibility}"
    args.mer_delta_fitness = new_mer_delta_fitness
    args.mer_delta_reproducibility = new_mer_delta_reproducibility

print("\n  -> Environment initialised.")

# Set up sampling
if args.container in CONTAINER_NO_SAMPLES:
    # Do not use sampling
    sampling_scoring_fn = scoring_fn
elif args.num_samples > 0 and args.container in CONTAINER_REQUIRE_ALL_SAMPLES:
    # Use sampling and requires all the evaluations
    sampling_scoring_fn = partial(
        multi_sample_scoring_function,
        scoring_fn=scoring_fn,
        num_samples=args.num_samples,
    )
elif args.reproducibility_objective and args.container not in CONTAINER_REEVAL_ARCHIVE:
    # Use sampling but uses only the reproducibility estimate
    sampling_scoring_fn = partial(
        sampling_descriptor_reproducibility,
        scoring_fn=scoring_fn,
        num_samples=args.num_samples,
        descriptor_extractor=EXTRACTOR_LIST[args.descriptor_extractor],
        descriptor_reproducibility_extractor=EXTRACTOR_LIST[
            args.descriptor_reproducibility_extractor
        ],
    )
elif args.num_samples > 0 and args.container not in CONTAINER_REEVAL_ARCHIVE:
    # Use sampling and uses only the fitness estimate
    sampling_scoring_fn = partial(
        sampling,
        scoring_fn=scoring_fn,
        num_samples=args.num_samples,
        fitness_extractor=EXTRACTOR_LIST[args.fitness_extractor],
        descriptor_extractor=EXTRACTOR_LIST[args.descriptor_extractor],
    )
else:
    # Do not use sampling
    sampling_scoring_fn = scoring_fn
print("\n  -> Sampling initialised.")

# Sample a set of initial solutions
init_policies, random_key = init_policies_fn(args.batch_size, random_key)

# Set up the algo
emitter = set_up_emitter(
    emitter_name=args.emitter,
    container_name=args.container,
    num_iterations=args.num_iterations,
    batch_size=args.batch_size,
    env=env,
    scoring_fn=scoring_fn,
    num_descriptors=num_descriptors,
    num_centroids=args.num_centroids,
    hard_limit_genotype=hard_limit_genotype,
    min_genotype=min_genotype,
    max_genotype=max_genotype,
    policy_structure=policy_structure,
    init_policies=init_policies,
    mutation=args.mutation,
    iso_sigma=args.iso_sigma,
    line_sigma=args.line_sigma,
    proportion_mutation=args.proportion_mutation,
    eta=args.eta,
)
map_elites = set_up_container(
    container_name=args.container,
    emitter_name=args.emitter,
    emitter=emitter,
    num_iterations=args.num_iterations,
    batch_size=args.batch_size,
    init_policies_fn=init_policies_fn,
    sampling_size=args.sampling_size,
    scoring_fn=sampling_scoring_fn,
    num_samples=args.num_samples,
    depth=args.depth,
    mutation=args.mutation,
    line_sigma=args.line_sigma,
    eta=args.eta,
    eas_max_samples=args.eas_max_samples,
    eas_use_evals=args.eas_use_evals,
    eas_archive_out_sampling=args.archive_out_sampling,
    fitness_extractor=args.fitness_extractor,
    fitness_reproducibility_extractor=args.fitness_reproducibility_extractor,
    descriptor_extractor=args.descriptor_extractor,
    descriptor_reproducibility_extractor=args.descriptor_reproducibility_extractor,
    mer_delta_fitness=args.mer_delta_fitness,
    mer_delta_reproducibility=args.mer_delta_reproducibility,
    mer_rho=args.mer_rho,
    mome_pareto_front_max_length=args.mome_max_pareto_front_length,
    mome_biased_sampling=args.mome_biased_sampling,
)
print("\n  -> Algorithm initialised.")

# Compute the centroids
centroids, random_key = compute_cvt_centroids(
    num_descriptors=num_descriptors,
    num_init_cvt_samples=args.num_init_cvt_samples,
    num_centroids=args.num_centroids,
    minval=min_bd,
    maxval=max_bd,
    random_key=random_key,
)
print("\n  -> Centroid initialised.")

# Set up the metric manager
empty_metric_repertoire = MapElitesRepertoire.init(
    genotypes=init_policies,
    fitnesses=jnp.zeros(args.batch_size),
    descriptors=jnp.zeros((args.batch_size, num_descriptors)),
    extra_scores={},
    centroids=centroids,
)
metrics_manager = get_metrics_manager(
    args=args,
    name=name,
    scoring_fn=scoring_fn,
    metric_repertoire=empty_metric_repertoire,
    evals_per_iter=evals_per_iter,
    min_bd=min_bd,
    max_bd=max_bd,
    qd_offset=qd_offset,
)
print("\n  -> Metrics Manager initialised.")

init_t = time.time() - step_t
print(f"\nFinished initialisation in {init_t} seconds. \n\nEntering run\n")

############
# III. Run #
############

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
evals = args.batch_size * max(args.num_samples, 1)
previous_evals = 0

# Compute and write initial metrics
metrics_t, reeval_t, write_t = metrics_manager.write_all_metrics(
    epoch=epoch,
    evals=evals,
    current_time=current_t,
    evals_per_offspring=evals_per_offspring,
    evals_per_iter=evals_per_iter,
    batch_size=args.batch_size,
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
    previous_evals = evals
    evals += evals_per_iter

    # Parallel-Adaptive-Sampling is the only algo that adapts live num_iterations
    # So take it into account for stopping criterion based on evaluations
    if "Parallel-Adaptive-Sampling" in args.container:
        evals = repertoire.total_evaluations
        if args.num_evaluations > 0 and evals >= args.num_evaluations:
            args.num_iterations = epoch

    ###########
    # Metrics #

    # Check log period
    if epoch % args.log_period != 0:
        continue

    # Compute effective running time (remove all metric time)
    current_t = time.time() - step_t - total_reeval_t - total_write_t - total_metrics_t
    print(
        f"\n    Epoch: {epoch} / {args.num_iterations} -- evals: {evals} -- time: {current_t}"
    )

    # Get additional run metrics
    metrics_t = time.time()
    if "Parallel-Adaptive-Sampling" in args.container:
        num_samples = map_elites.num_samples
        batch_size = map_elites.batch_size
    else:
        num_samples = evals_per_offspring
        batch_size = args.batch_size

    metrics_t = time.time() - metrics_t
    total_metrics_t += metrics_t

    # Compute and write metrics
    metrics_t, reeval_t, write_t = metrics_manager.write_all_metrics(
        epoch=epoch,
        evals=evals,
        current_time=current_t,
        evals_per_offspring=num_samples,
        evals_per_iter=evals - previous_evals,
        batch_size=batch_size,
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
    f"\n    Ended at epoch: {epoch} / {args.num_iterations} -- evals: {evals} -- time: {current_t}"
)

# Get additional run metrics
if "Parallel-Adaptive-Sampling" in args.container:
    num_samples = map_elites.num_samples
    batch_size = map_elites.batch_size
else:
    num_samples = evals_per_offspring
    batch_size = args.batch_size

# Compute and write metrics
_, _, _ = metrics_manager.write_all_metrics(
    epoch=epoch,
    evals=evals,
    current_time=current_t,
    evals_per_offspring=num_samples,
    evals_per_iter=evals - previous_evals,
    batch_size=batch_size,
    repertoire=repertoire,
    projected_repertoire=repertoire,
    emitter_state=emitter_state,
    random_key=random_key,
)

print("\nFinished run:", time.time() - step_t)
