import argparse
import os
import time
from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.random import randint
from set_up_algo import set_up_algo

from analysis.save_metrics import save_config
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey
from set_up_environment import ENV_LIST, ENV_OPTIMISATION, set_up_environment

# Limit CPU usage for HPC
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false " "intra_op_parallelism_threads=4"
)

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
parser.add_argument("--size-log-period", action="store_true")

# Stopping criterion
parser.add_argument("--num-iterations", default=0, type=int)

# Compare size
parser.add_argument("--batch-size", default=0, type=int)
parser.add_argument("--sampling-size", default=0, type=int)

# Environment
parser.add_argument("--env-name", default="ant_omni", type=str)
parser.add_argument("--episode-length", default=250, type=int)
parser.add_argument("--gaussian-vel", action="store_true")
parser.add_argument("--gaussian-pos", action="store_true")
parser.add_argument("--optimisation-fit-variance", default=0.0, type=float)
parser.add_argument("--optimisation-desc-variance", default=0.0, type=float)
parser.add_argument("--params-variance", default=0.0, type=float)
parser.add_argument("--policy-hidden-layer-sizes", default="8", type=str)

# Archive
parser.add_argument("--num-centroids", default=512, type=int)
parser.add_argument("--num-init-cvt-samples", default=50000, type=int)

# Num reeval tunning
parser.add_argument("--scan-batch", default=1024, type=int, help="Not used if 0.")
parser.add_argument("--min-replications", default=16, type=int)
parser.add_argument("--max-replications", default=16384, type=int)

args = parser.parse_args()

# Check that inputs are valid
if args.batch_size != 0 and args.sampling_size != 0:
    print("\n!!!WARNING!!! Considering sampling size over batch size.")
    args.batch_size = args.sampling_size
elif args.batch_size != 0:
    args.sampling_size = args.batch_size
elif args.sampling_size != 0:
    args.batch_size = args.sampling_size
elif args.batch_size == 0 and args.sampling_size == 0:
    assert 0, "\n!!!ERROR!!! No --sampling-size nor --batch-size."
assert args.env_name in ENV_LIST, "\n!!!ERROR!!! Invalid env name:" + args.env_name
assert args.num_iterations != 0, "\n!!!ERROR!!! No stopping criterion."

####################
# I. Configuration #
####################

# Set random seed
args.seed = randint(1000000) if args.seed == 0 else args.seed

# Process policy structure
args.policy_hidden_layer_sizes = tuple(
    [int(x) for x in args.policy_hidden_layer_sizes.split("_")]
)

# Create algo name for metrics and analysis files
name = "Vanilla-MAP-Elites" + args.suffixe
if args.env_name not in ENV_OPTIMISATION:
    if args.gaussian_vel:
        name += "_velnormal"
    if args.gaussian_pos:
        name += "_posnormal"
if args.deterministic:
    name += "_deterministic"

# Print
print("\n\nParameters:")
print("  Name:", name)
print("  Run:")
print("    -> seed:", args.seed)
print("    -> results:", args.results)
print("  Env:")
print("    -> env_name:", args.env_name)
if args.params_variance > 0:
    print("    -> params_variance:", args.params_variance)
if args.env_name in ENV_OPTIMISATION:
    print("    -> optimisation_fit_variance:", args.optimisation_fit_variance)
    print("    -> optimisation_desc_variance:", args.optimisation_desc_variance)
else:
    print("    -> episode_length:", args.episode_length)
    print("    -> gaussian_vel" if args.gaussian_vel else "    -> uniform_vel")
    print("    -> gaussian_pos" if args.gaussian_pos else "    -> uniform_pos")
print("    -> policy_hidden_layer_sizes:", args.policy_hidden_layer_sizes)
print("    -> num_iterations:", args.num_iterations)
print("    -> batch_size:", args.batch_size)


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
    env,
    scoring_fn,
    policy_structure,
    init_policies,
    min_genotype,
    max_genotype,
    min_bd,
    max_bd,
    qd_offset,
    num_descriptors,
    random_key,
) = set_up_environment(
    deterministic=args.deterministic,
    env_name=args.env_name,
    episode_length=args.episode_length,
    optimisation_fit_variance=args.optimisation_fit_variance,
    optimisation_desc_variance=args.optimisation_desc_variance,
    params_variance=args.params_variance,
    batch_size=args.batch_size,
    policy_hidden_layer_sizes=args.policy_hidden_layer_sizes,
    gaussian_vel=args.gaussian_vel,
    gaussian_pos=args.gaussian_pos,
    random_key=random_key,
)
if args.env_name in ENV_OPTIMISATION:
    # Add noise values to name
    if (
        args.optimisation_fit_variance == 0
        and args.optimisation_desc_variance == 0
        and args.params_variance == 0
    ):
        args.env_name += "_nonoise"
    else:
        args.env_name += (
            f"_fit{args.optimisation_fit_variance}"
            f"_desc{args.optimisation_desc_variance}"
            f"_params{args.params_variance}"
        )
else:
    if args.gaussian_vel:
        args.env_name += "_velnormal"
    if args.gaussian_pos:
        args.env_name += "_posnormal"

# Define algo
_, map_elites = set_up_algo(
    container_name="MAP-Elites",
    emitter_name="Mixing",
    num_iterations=args.num_iterations,
    batch_size=args.batch_size,
    sampling_size=args.sampling_size,
    env=env,
    scoring_fn=scoring_fn,
    num_descriptors=num_descriptors,
    min_genotype=min_genotype,
    max_genotype=max_genotype,
    policy_structure=policy_structure,
    init_policies=init_policies,
    depth=0,
    eas_max_samples=0,
    eas_use_evals="",
    eas_archive_out_sampling=0,
    es_num_gradient_samples=0,
    es_no_mirror=0,
    es_num_optimizer_steps=0,
    es_no_adam=0,
    es_no_explore=0,
    es_scan_batch_size=0,
    es_no_novelty_archive=0,
    esga_num_ga=0,
    pg_num_critic_training_steps=0,
    pg_replay_buffer_size=0,
    qd_offset=qd_offset,
    use_median=0,
)

# Compute the centroids
centroids, random_key = compute_cvt_centroids(
    num_descriptors=num_descriptors,
    num_init_cvt_samples=args.num_init_cvt_samples,
    num_centroids=args.num_centroids,
    minval=min_bd,
    maxval=max_bd,
    random_key=random_key,
)

############
# III. Run #
############

init_t = time.time() - step_t
print("\nFinished initialisation:", time.time() - step_t, "\n\nEntering run\n")
step_t = time.time()

# Compute initial repertoire
repertoire, emitter_state, random_key = map_elites.init(
    init_policies, centroids, random_key
)
jax.tree_util.tree_map(lambda x: x.block_until_ready(), repertoire.genotypes)

# Create results folder
results_repertoire = f"{args.results}/repertoire_{name}_{args.seed}/"
file_reevals_name = f"{args.results}/reevals_tunning_{name}_{args.seed}.csv"
if not os.path.exists(args.results):
    os.mkdir(args.results)
if not os.path.exists(results_repertoire):
    os.mkdir(results_repertoire)

# Create config
config_file = save_config(
    args.results,
    name,
    args.seed,
    args.env_name,
    args.episode_length,
    args.params_variance,
    min_bd,
    max_bd,
    args.batch_size,
    args.sampling_size,
    args.batch_size,
    args.num_iterations,
    args.policy_hidden_layer_sizes,
    args.num_init_cvt_samples,
    args.num_centroids,
    0,
    0,
    0,
    file_reevals_name,
    "",
    results_repertoire,
    "",
    "",
    "",
    "",
    "",
)
print("  -> Config saved in", config_file)

# main loop
map_elites_scan_update = map_elites.scan_update
(repertoire, emitter_state, random_key), metrics = jax.lax.scan(
    map_elites_scan_update,
    (repertoire, emitter_state, random_key),
    (),
    length=args.num_iterations,
)

# Write repertoire
repertoire.save(path=results_repertoire)
print("  -> Final repertoire saved in", results_repertoire)


######################
# Num reeval tunning #

print("\nFinished run:", time.time() - step_t, "\n\nEntering reevals tunning\n")
step_t = time.time()

# Get evaluations
repertoire_fill = repertoire.fitnesses > -jnp.inf
num_indivs = jnp.sum(repertoire_fill)
policies_params = repertoire.genotypes
print("  -> Got", num_indivs, "valid individuals in archive")


# Clearing GPU
def clear_cache_all() -> None:
    backend = jax.lib.xla_bridge.get_backend()
    for buf in backend.live_buffers():
        buf.delete()


map_elites_scan_update._clear_cache()
map_elites.scan_update._clear_cache()
map_elites.update._clear_cache()
map_elites.init._clear_cache

# Get all replications values
replications_value = args.min_replications - args.min_replications % 2
replications_values = [replications_value]
while replications_value < args.max_replications:
    replications_value *= 2
    replications_values.append(replications_value)
print("  -> Replications values:", replications_values)
min_replications_value = replications_values[0]
max_replications_value = replications_values[-1]


@partial(
    jax.jit,
    static_argnames=(
        "batch_size",
        "scoring_fn",
        "num_samples",
        "max_replications_batch",
        "scan_batch",
    ),
)
def sampling(
    policies_params: Genotype,
    batch_size: int,
    random_key: RNGKey,
    scoring_fn: Callable[
        [Genotype, RNGKey],
        Tuple[Fitness, Descriptor, ExtraScores, RNGKey],
    ],
    num_samples: int,
    max_replications_batch: int = 256,
    scan_batch: int = 0,
) -> Tuple[Fitness, Descriptor, Fitness, Descriptor, Fitness, Descriptor, RNGKey]:
    """
    Wrap scoring_function to perform sampling.

    Args:
        policies_params: policies to evaluate
        random_key
        scoring_fn: scoring function used for evaluation
        num_samples

    Returns:
        The closer median and average fitness and descriptor of the individuals
    """

    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(subkey, num=num_samples)

    sample_scoring_fn = jax.vmap(scoring_fn, (None, 0), 1)

    # evaluate
    if (
        scan_batch == 0
        or num_samples <= max_replications_batch
        or batch_size <= scan_batch
    ):
        all_fitnesses, all_descriptors, all_extra_scores, keys = sample_scoring_fn(
            policies_params, keys
        )
    else:
        num_scan = batch_size // scan_batch
        policies_params = jax.tree_map(
            lambda x: x.reshape((num_scan, scan_batch) + x.shape[1:]),
            policies_params,
        )

        def scoring_scan(
            carry: int,
            unused: Tuple[()],
        ) -> Tuple[int, Tuple[Descriptor, Fitness]]:
            fitnesses, descriptors, extra_scores, key = sample_scoring_fn(
                jax.tree_map(lambda x: x[carry], policies_params),
                keys,
            )
            return (carry + 1), (fitnesses, descriptors)

        (_), (all_fitnesses, all_descriptors) = jax.lax.scan(
            scoring_scan, (0), (), length=num_scan
        )
        all_fitnesses = all_fitnesses.reshape((batch_size, num_samples))
        all_descriptors = all_descriptors.reshape(
            (batch_size, num_samples) + all_descriptors.shape[3:]
        )

    # compute closer stats
    def one_dim_closer_median(values: jnp.ndarray) -> jnp.ndarray:
        def distance(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            return jnp.sqrt(jnp.sum(jnp.square(x - y)))

        distances = jax.vmap(
            jax.vmap(partial(distance), in_axes=(None, 0)), in_axes=(0, None)
        )(values, values)
        return values[jnp.argmin(jnp.mean(distances, axis=0))]

    closer_descriptors = jax.vmap(one_dim_closer_median)(all_descriptors)
    closer_fitnesses = jax.vmap(one_dim_closer_median)(all_fitnesses)

    # compute median stats
    median_descriptors = jnp.median(all_descriptors, axis=1)
    median_fitnesses = jnp.median(all_fitnesses, axis=1)

    # compute average stats
    average_descriptors = jnp.average(all_descriptors, axis=1)
    average_fitnesses = jnp.average(all_fitnesses, axis=1)

    return (
        average_fitnesses,
        average_descriptors,
        median_fitnesses,
        median_descriptors,
        closer_fitnesses,
        closer_descriptors,
        random_key,
    )


# Create the file to save values
file_reevals = open(file_reevals_name, "w")
file_reevals.write("env,algo,seed,replications,label,descriptor_dist,fitness_dist\n")
file_reevals.flush()
file_reevals.close()

# Do sampling once for maximal replications value
max_replications_value = replications_values[-1]
print("    -> Running replications:", max_replications_value)
(
    max_average_fitnesses,
    max_average_descriptors,
    max_median_fitnesses,
    max_median_descriptors,
    max_closer_fitnesses,
    max_closer_descriptors,
    random_key,
) = sampling(
    policies_params=policies_params,
    batch_size=args.num_centroids,
    random_key=random_key,
    scoring_fn=scoring_fn,
    num_samples=max_replications_value,
    max_replications_batch=256,
    scan_batch=args.scan_batch,
)
max_average_fitnesses = max_average_fitnesses[repertoire_fill]
max_average_descriptors = max_average_descriptors[repertoire_fill]
max_median_fitnesses = max_median_fitnesses[repertoire_fill]
max_median_descriptors = max_median_descriptors[repertoire_fill]
max_closer_fitnesses = max_closer_fitnesses[repertoire_fill]
max_closer_descriptors = max_closer_descriptors[repertoire_fill]

for replications_value in replications_values:
    row_prefixe = f"{args.env_name},{name},{args.seed},{replications_value},"

    # Sample
    print("    -> Running replications:", replications_value)
    (
        average_fitnesses,
        average_descriptors,
        median_fitnesses,
        median_descriptors,
        closer_fitnesses,
        closer_descriptors,
        random_key,
    ) = sampling(
        policies_params=policies_params,
        batch_size=args.num_centroids,
        random_key=random_key,
        scoring_fn=scoring_fn,
        num_samples=replications_value,
        max_replications_batch=256,
        scan_batch=args.scan_batch,
    )
    average_fitnesses = average_fitnesses[repertoire_fill]
    average_descriptors = average_descriptors[repertoire_fill]
    median_fitnesses = median_fitnesses[repertoire_fill]
    median_descriptors = median_descriptors[repertoire_fill]
    closer_fitnesses = closer_fitnesses[repertoire_fill]
    closer_descriptors = closer_descriptors[repertoire_fill]

    # Compute distances
    average_fitnesses_distances = jnp.abs(max_average_fitnesses - average_fitnesses)
    average_descriptors_distances = jnp.linalg.norm(
        max_average_descriptors - average_descriptors, axis=1
    )
    median_fitnesses_distances = jnp.abs(max_median_fitnesses - median_fitnesses)
    median_descriptors_distances = jnp.linalg.norm(
        max_median_descriptors - median_descriptors, axis=1
    )
    closer_fitnesses_distances = jnp.abs(max_closer_fitnesses - closer_fitnesses)
    closer_descriptors_distances = jnp.linalg.norm(
        max_closer_descriptors - closer_descriptors, axis=1
    )
    print("    -> Finished replications:", replications_value)

    # Save values
    average_rows = [
        row_prefixe
        + "average,{},{}".format(
            average_fitnesses_distances[i], average_descriptors_distances[i]
        )
        for i in range(num_indivs)
    ]
    median_rows = [
        row_prefixe
        + "median,{},{}".format(
            median_fitnesses_distances[i], median_descriptors_distances[i]
        )
        for i in range(num_indivs)
    ]
    closer_rows = [
        row_prefixe
        + "closer,{},{}".format(
            closer_fitnesses_distances[i], closer_descriptors_distances[i]
        )
        for i in range(num_indivs)
    ]
    all_rows = average_rows + median_rows + closer_rows
    text = "\n".join(all_rows) + "\n"
    file_reevals = open(file_reevals_name, "a")
    file_reevals.write(text)
    file_reevals.flush()
    file_reevals.close()

print("\nFinished reevals tunning:", time.time() - step_t, "\n\n")
