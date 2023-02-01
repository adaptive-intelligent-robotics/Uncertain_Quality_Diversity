import argparse
import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from numpy.random import randint
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids

from analysis.save_metrics import create_metrics_csv, save_config, write_metrics_csv
from core.containers.mapelites_repertoire import MapElitesRepertoire
from core.incell_stochasticity_utils import (
    incell_reevaluation_function,
    metrics_incell_random_wrapper,
)
from core.stochasticity_utils import reevaluation_function, sampling
from set_up_algo import set_up_algo
from set_up_environment import ENV_LIST, ENV_NEUROEVOLUTION, set_up_environment

# Limit CPU usage for HPC
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false " "intra_op_parallelism_threads=4"
)

# Container list
CONTAINER_LIST = [
    "MAP-Elites",
    "Archive-Sampling",
    "Deep-Grid",
    "Parallel-Adaptive-Sampling",
]

# Emitter list
EMITTER_LIST = ["Random", "Mixing", "PGA"]

# Container that require to input a depth
CONTAINER_REQUIRE_DEPTH = [
    "Deep-Grid",
    "Parallel-Adaptive-Sampling",
    "Archive-Sampling",
]

# Container that uses an in-cell selection to compute reevaluation stat
# WARNING "sample_all_cell" method needs to be implemented.
CONTAINER_REQUIRE_INCELL_SELECTION = ["Deep-Grid"]

# Container that reeval the archive periodically
CONTAINER_REEVAL_ARCHIVE = ["Archive-Sampling", "Parallel-Adaptive-Sampling"]


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
parser.add_argument("--num-reevals", default=512, type=int)
parser.add_argument("--use-average-reeval", action="store_true")

# Stopping criterion
parser.add_argument("--num-iterations", default=0, type=int)
parser.add_argument("--num-evaluations", default=0, type=int)

# Compare size
parser.add_argument("--batch-size", default=0, type=int)
parser.add_argument("--sampling-size", default=0, type=int)
parser.add_argument("--archive-out-sampling", action="store_true")

# Environment
parser.add_argument("--env-name", default="ant_omni", type=str)
parser.add_argument("--episode-length", default=250, type=int)
parser.add_argument("--policy-hidden-layer-sizes", default="8", type=str)

# Archive
parser.add_argument("--num-centroids", default=1024, type=int)
parser.add_argument("--num-init-cvt-samples", default=50000, type=int)

# Algorithm
parser.add_argument("--container", default="MAP-Elites", type=str)
parser.add_argument("--emitter", default="Mixing", type=str)
parser.add_argument("--num-samples", default=0, type=int)
parser.add_argument("--use-median", action="store_true")
parser.add_argument("--depth", default=0, type=int)
parser.add_argument("--eas-max-samples", default=0, type=int)
parser.add_argument("--eas-use-evals", default="median", type=str)

args = parser.parse_args()

# Check that inputs are valid
if args.batch_size != 0 and args.sampling_size != 0:
    print("\n!!!WARNING!!! Considering sampling size over batch size.")
    args.batch_size = 0
elif args.batch_size == 0 and args.sampling_size == 0:
    assert 0, "\n!!!ERROR!!! No --sampling-size nor --batch-size."
if args.archive_out_sampling:
    if args.sampling_size == 0:
        args.archive_out_sampling = False
    else:
        print("\n!!!WARNING!!! Not considering archive evaluation in sampling budget.")
assert args.env_name in ENV_LIST, "\n!!!ERROR!!! Invalid env name:" + args.env_name
assert args.container in CONTAINER_LIST, "\n!!!ERROR!!! Invalid " + args.container
if args.container in CONTAINER_REQUIRE_DEPTH:
    assert args.depth != 0, "\n!!!ERROR!!! " + args.container + " has depth 0."
assert args.emitter in EMITTER_LIST, "\n!!!ERROR!!! Invalid emitter:" + args.emitter
if args.emitter == "PGA":
    assert args.env_name in ENV_NEUROEVOLUTION, "\n!!!ERROR!!! PGA only for NN tasks."
if args.num_iterations != 0 and args.num_evaluations != 0:
    print("\n!!!WARNING!!! Considering iterations over evaluations.")
    args.num_evaluations = 0
elif args.num_iterations == 0 and args.num_evaluations == 0:
    assert 0, "\n!!!ERROR!!! No stopping criterion."
if args.use_median:
    if args.container in CONTAINER_REEVAL_ARCHIVE:
        print("!!!WARNING!!! --use-median has no impact on", CONTAINER_REEVAL_ARCHIVE)
        args.use_median = False
    if args.num_samples == 0:
        print("!!!WARNING!!! --use-median has no impact with --num-samples 0.")
        args.use_median = False
if args.num_samples != 0 and args.container in CONTAINER_REEVAL_ARCHIVE:
    print("!!!WARNING!!! --num-samples applies for archive reevaluation as well.")


####################
# I. Configuration #
####################

# Set random seed
args.seed = randint(1000000) if args.seed == 0 else args.seed

# Process policy structure
args.policy_hidden_layer_sizes = tuple(
    [int(x) for x in args.policy_hidden_layer_sizes.split("_")]
)

# Compute number of evals per offpsring
evals_per_offspring = max(args.num_samples, 1)

# Compute addtional number of evals per iteration
add_evals_per_iter = 0
if args.container in CONTAINER_REEVAL_ARCHIVE and not (args.archive_out_sampling):
    add_evals_per_iter += args.num_centroids * args.depth

# Compute batch_size from sampling_size
if args.sampling_size != 0:
    assert args.sampling_size > add_evals_per_iter + evals_per_offspring, (
        "!!!ERROR!!! Missing sampling credit for evaluation, got "
        + str(args.sampling_size)
        + " left, require at least "
        + str(evals_per_offspring)
        + " per offspring and "
        + str(add_evals_per_iter)
        + " for the rest (archive reevaluation, etc)."
    )
    left_sampling_size = args.sampling_size - add_evals_per_iter
    args.batch_size = left_sampling_size // evals_per_offspring
base_evals_per_iter = args.batch_size * evals_per_offspring

# Compute number of evals per iteration
evals_per_iter = base_evals_per_iter + add_evals_per_iter

# Compute run length from num_evaluations
if args.num_evaluations > 0:
    args.num_iterations = args.num_evaluations // evals_per_iter

# Create algo name for metrics and analysis files
name = args.container + args.suffixe
if args.emitter != "Mixing":
    name += "-" + args.emitter
if args.container == "Parallel-Adaptive-Sampling":
    name += "-" + args.eas_use_evals
    if args.eas_max_samples > 0:
        name += "-max" + str(args.eas_max_samples)
if args.depth != 0:
    name += "-depth-" + str(args.depth)
if args.num_samples != 0:
    if args.use_median:
        name += "-medsampling-" + str(args.num_samples)
    else:
        name += "-sampling-" + str(args.num_samples)
if args.container in CONTAINER_REEVAL_ARCHIVE and args.archive_out_sampling:
    name += "-archive-out-sampling"
if args.deterministic:
    name += "_deterministic"
if name == "MAP-Elites":
    name = "Vanilla-MAP-Elites"

# Print
print("\n\nParameters:")
print("  Name:", name)
print("  Run:")
print("    -> seed:", args.seed)
print("    -> results:", args.results)
print("    -> log_period:", args.log_period)
print("    -> archive_log_period:", args.archive_log_period)
if args.num_reevals > 0:
    print("    -> num_reevals:", args.num_reevals)
print("  Env:")
print("    -> env_name:", args.env_name)
print("    -> episode_length:", args.episode_length)
print("    -> policy_hidden_layer_sizes:", args.policy_hidden_layer_sizes)
print("  Algo:")
print("    -> container:", args.container)
print("    -> emitter:", args.emitter)
print("    -> num_init_cvt_samples:", args.num_init_cvt_samples)
print("    -> num_centroids:", args.num_centroids)
if args.num_samples > 0:
    print("    -> num_samples:", args.num_samples)
if args.depth > 0:
    print("    -> depth:", args.depth)
print("  Evals and epochs:")
if args.num_evaluations != 0:
    print("    -> num_evaluations:", args.num_evaluations)
else:
    print("    -> num_iterations:", args.num_iterations)
if args.sampling_size != 0:
    print("    -> sampling_size:", args.sampling_size)
print("    -> batch_size:", args.batch_size)
print("    -> evals_per_iter:", evals_per_iter)


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
    batch_size=args.batch_size,
    policy_hidden_layer_sizes=args.policy_hidden_layer_sizes,
    random_key=random_key,
)

# Set up sampling
if args.num_samples > 0:
    sampling_scoring_fn = partial(
        sampling,
        scoring_fn=scoring_fn,
        num_samples=args.num_samples,
        use_median=args.use_median,
    )
else:
    sampling_scoring_fn = scoring_fn

# Define algo
metrics_function, map_elites = set_up_algo(
    container_name=args.container,
    emitter_name=args.emitter,
    num_iterations=args.num_iterations,
    batch_size=args.batch_size,
    sampling_size=args.sampling_size,
    env=env,
    scoring_fn=sampling_scoring_fn,
    num_descriptors=num_descriptors,
    min_genotype=min_genotype,
    max_genotype=max_genotype,
    policy_structure=policy_structure,
    init_policies=init_policies,
    depth=args.depth,
    eas_max_samples=args.eas_max_samples,
    eas_use_evals=args.eas_use_evals,
    eas_archive_out_sampling=args.archive_out_sampling,
    qd_offset=qd_offset,
    use_median=args.use_median,
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

# Prepare the reeval function
metric_repertoire = MapElitesRepertoire.init(
    genotypes=init_policies,
    fitnesses=jnp.zeros(args.batch_size),
    descriptors=jnp.zeros((args.batch_size, num_descriptors)),
    extra_scores={},
    centroids=centroids,
)
reevaluation_fn = partial(
    reevaluation_function,
    metric_repertoire=metric_repertoire,
    scoring_fn=scoring_fn,
    num_reevals=args.num_reevals,
    use_median=not args.use_average_reeval,
)
if args.container in CONTAINER_REQUIRE_INCELL_SELECTION:
    in_cell_metrics_function_fn = partial(
        metrics_incell_random_wrapper,
        metrics_function=metrics_function,
        depth=args.depth,
    )
    in_cell_reevaluation_fn = partial(
        incell_reevaluation_function,
        metric_repertoire=metric_repertoire,
        scoring_fn=scoring_fn,
        depth=args.depth,
        num_reevals=args.num_reevals,
        use_median=not args.use_average_reeval,
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

# Compute initial metrics
metrics_t = time.time()
epoch = 0
evals = args.batch_size * max(args.num_samples, 1)
previous_evals = 0
metrics = metrics_function(repertoire)
jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
metrics_t = time.time() - metrics_t
total_metrics_t = metrics_t

# Compute initial reeval metrics
reeval_t = time.time()
(
    reeval_repertoire,
    fit_reeval_repertoire,
    desc_reeval_repertoire,
    fit_var_repertoire,
    desc_var_repertoire,
    random_key,
) = reevaluation_fn(repertoire, random_key)
jax.tree_util.tree_map(lambda x: x.block_until_ready(), reeval_repertoire.genotypes)
reeval_metrics = metrics_function(reeval_repertoire)
fit_reeval_metrics = metrics_function(fit_reeval_repertoire)
desc_reeval_metrics = metrics_function(desc_reeval_repertoire)
fit_var_metrics = metrics_function(fit_var_repertoire)
desc_var_metrics = metrics_function(desc_var_repertoire)
jax.tree_util.tree_map(lambda x: x.block_until_ready(), desc_var_metrics)
if args.container in CONTAINER_REQUIRE_INCELL_SELECTION:
    in_cell_metrics, random_key = in_cell_metrics_function_fn(repertoire, random_key)
    (
        in_cell_reeval_repertoire,
        in_cell_fit_reeval_repertoire,
        in_cell_desc_reeval_repertoire,
        in_cell_fit_var_repertoire,
        in_cell_desc_var_repertoire,
        random_key,
    ) = in_cell_reevaluation_fn(repertoire, random_key)
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready(), in_cell_reeval_repertoire.genotypes
    )
    in_cell_reeval_metrics = metrics_function(in_cell_reeval_repertoire)
    in_cell_fit_reeval_metrics = metrics_function(in_cell_fit_reeval_repertoire)
    in_cell_desc_reeval_metrics = metrics_function(in_cell_desc_reeval_repertoire)
    in_cell_fit_var_metrics = metrics_function(in_cell_fit_var_repertoire)
    in_cell_desc_var_metrics = metrics_function(in_cell_desc_var_repertoire)
else:
    in_cell_reeval_repertoire = reeval_repertoire
    in_cell_fit_reeval_repertoire = fit_reeval_repertoire
    in_cell_desc_reeval_repertoire = desc_reeval_repertoire
    in_cell_fit_var_repertoire = fit_var_repertoire
    in_cell_desc_var_repertoire = desc_var_repertoire
    in_cell_metrics = metrics
    in_cell_reeval_metrics = reeval_metrics
    in_cell_fit_reeval_metrics = fit_reeval_metrics
    in_cell_desc_reeval_metrics = desc_reeval_metrics
    in_cell_fit_var_metrics = fit_var_metrics
    in_cell_desc_var_metrics = desc_var_metrics
jax.tree_util.tree_map(lambda x: x.block_until_ready(), in_cell_desc_var_metrics)
reeval_t = time.time() - reeval_t
total_reeval_t = reeval_t

# Write initial metrics
current_t = time.time() - step_t - total_reeval_t - total_metrics_t
write_t = time.time()

# Create results folders
repertoire_suffixe = "repertoire_" + name + "_" + str(args.seed) + "/"
results_repertoire = args.results + "/" + repertoire_suffixe
results_reeval_repertoire = args.results + "/reeval_" + repertoire_suffixe
results_fit_reeval_repertoire = args.results + "/fit_reeval_" + repertoire_suffixe
results_desc_reeval_repertoire = args.results + "/desc_reeval_" + repertoire_suffixe
results_fit_var_repertoire = args.results + "/fit_var_" + repertoire_suffixe
results_desc_var_repertoire = args.results + "/desc_var_" + repertoire_suffixe
results_in_cell_reeval_repertoire = (
    args.results + "/in_cell_reeval_" + repertoire_suffixe
)
results_in_cell_fit_reeval_repertoire = (
    args.results + "/in_cell_fit_reeval_" + repertoire_suffixe
)
results_in_cell_desc_reeval_repertoire = (
    args.results + "/in_cell_desc_reeval_" + repertoire_suffixe
)
results_in_cell_fit_var_repertoire = (
    args.results + "/in_cell_fit_var_" + repertoire_suffixe
)
results_in_cell_desc_var_repertoire = (
    args.results + "/in_cell_desc_var_" + repertoire_suffixe
)
if not os.path.exists(args.results):
    os.mkdir(args.results)
if not os.path.exists(results_repertoire):
    os.mkdir(results_repertoire)
if not os.path.exists(results_reeval_repertoire):
    os.mkdir(results_reeval_repertoire)
if not os.path.exists(results_fit_reeval_repertoire):
    os.mkdir(results_fit_reeval_repertoire)
if not os.path.exists(results_desc_reeval_repertoire):
    os.mkdir(results_desc_reeval_repertoire)
if not os.path.exists(results_fit_var_repertoire):
    os.mkdir(results_fit_var_repertoire)
if not os.path.exists(results_desc_var_repertoire):
    os.mkdir(results_desc_var_repertoire)
if not os.path.exists(results_in_cell_reeval_repertoire):
    os.mkdir(results_in_cell_reeval_repertoire)
if not os.path.exists(results_in_cell_fit_reeval_repertoire):
    os.mkdir(results_in_cell_fit_reeval_repertoire)
if not os.path.exists(results_in_cell_desc_reeval_repertoire):
    os.mkdir(results_in_cell_desc_reeval_repertoire)
if not os.path.exists(results_in_cell_fit_var_repertoire):
    os.mkdir(results_in_cell_fit_var_repertoire)
if not os.path.exists(results_in_cell_desc_var_repertoire):
    os.mkdir(results_in_cell_desc_var_repertoire)

# Saving initial metrics as csv
metrics_file = create_metrics_csv(
    args.results,
    name,
    args.seed,
    epoch,
    evals,
    current_t,
    metrics["qd_score"],
    metrics["coverage"],
    metrics["max_fitness"],
    reeval_metrics["qd_score"],
    reeval_metrics["coverage"],
    reeval_metrics["max_fitness"],
    fit_reeval_metrics["qd_score"],
    fit_reeval_metrics["coverage"],
    fit_reeval_metrics["max_fitness"],
    desc_reeval_metrics["qd_score"],
    desc_reeval_metrics["coverage"],
    desc_reeval_metrics["max_fitness"],
    fit_var_metrics["qd_score"],
    fit_var_metrics["coverage"],
    fit_var_metrics["max_fitness"],
    desc_var_metrics["qd_score"],
    desc_var_metrics["coverage"],
    desc_var_metrics["max_fitness"],
    evals_per_offspring,
    evals_per_iter,
    args.batch_size,
)
in_cell_metrics_file = create_metrics_csv(
    args.results,
    name,
    args.seed,
    epoch,
    evals,
    current_t,
    in_cell_metrics["qd_score"],
    in_cell_metrics["coverage"],
    in_cell_metrics["max_fitness"],
    in_cell_reeval_metrics["qd_score"],
    in_cell_reeval_metrics["coverage"],
    in_cell_reeval_metrics["max_fitness"],
    in_cell_fit_reeval_metrics["qd_score"],
    in_cell_fit_reeval_metrics["coverage"],
    in_cell_fit_reeval_metrics["max_fitness"],
    in_cell_desc_reeval_metrics["qd_score"],
    in_cell_desc_reeval_metrics["coverage"],
    in_cell_desc_reeval_metrics["max_fitness"],
    in_cell_fit_var_metrics["qd_score"],
    in_cell_fit_var_metrics["coverage"],
    in_cell_fit_var_metrics["max_fitness"],
    in_cell_desc_var_metrics["qd_score"],
    in_cell_desc_var_metrics["coverage"],
    in_cell_desc_var_metrics["max_fitness"],
    evals_per_offspring,
    evals_per_iter,
    args.batch_size,
    prefixe="in_cell_",
)
print("  -> Initial metrics saved in", metrics_file)
print("  -> Initial in_cell_metrics saved in", in_cell_metrics_file)

# Saving initial repertoire as npy
repertoire.save(path=results_repertoire)
reeval_repertoire.save(path=results_reeval_repertoire)
fit_reeval_repertoire.save(path=results_fit_reeval_repertoire)
desc_reeval_repertoire.save(path=results_desc_reeval_repertoire)
fit_var_repertoire.save(path=results_fit_var_repertoire)
desc_var_repertoire.save(path=results_desc_var_repertoire)
in_cell_reeval_repertoire.save(path=results_in_cell_reeval_repertoire)
in_cell_fit_reeval_repertoire.save(path=results_in_cell_fit_reeval_repertoire)
in_cell_desc_reeval_repertoire.save(path=results_in_cell_desc_reeval_repertoire)
in_cell_fit_var_repertoire.save(path=results_in_cell_fit_var_repertoire)
in_cell_desc_var_repertoire.save(path=results_in_cell_desc_var_repertoire)
print("  -> All initial repertoire saved, original in", results_repertoire)

# Create config
config_file = save_config(
    args.results,
    name,
    args.seed,
    args.env_name,
    args.episode_length,
    min_bd,
    max_bd,
    args.batch_size,
    args.sampling_size,
    evals_per_iter,
    args.num_iterations,
    args.policy_hidden_layer_sizes,
    args.num_init_cvt_samples,
    args.num_centroids,
    args.num_samples,
    args.num_reevals,
    args.depth,
    metrics_file,
    in_cell_metrics_file,
    results_repertoire,
    results_reeval_repertoire,
    results_fit_reeval_repertoire,
    results_desc_reeval_repertoire,
    results_fit_var_repertoire,
    results_desc_var_repertoire,
    results_in_cell_reeval_repertoire,
    results_in_cell_fit_reeval_repertoire,
    results_in_cell_desc_reeval_repertoire,
    results_in_cell_fit_var_repertoire,
    results_in_cell_desc_var_repertoire,
)
print("  -> Config saved in", config_file)
write_t = time.time() - write_t
total_write_t = write_t

# Initialise counter for convergence-based stopping criterion
previous_qd_score = metrics["qd_score"]
gen_counter = 0

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
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), repertoire.genotypes)

    # Update metrics
    metrics_t = time.time()
    epoch += 1
    previous_evals = evals
    evals += evals_per_iter
    metrics = metrics_function(repertoire)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
    metrics_t = time.time() - metrics_t
    total_metrics_t += metrics_t

    # Parallel-Adaptive-Sampling is the only algo that adapts live num_iterations
    if args.container == "Parallel-Adaptive-Sampling":
        evals = repertoire.total_evaluations
        if args.num_evaluations > 0 and evals >= args.num_evaluations:
            args.num_iterations = epoch

    ########################
    # Metrics and analysis #

    # Write metrics
    if epoch % args.log_period != 0:
        continue

    reeval_t = time.time()
    print(
        "\n    Epoch:",
        epoch,
        "/",
        args.num_iterations,
        "-- evals:",
        evals,
        "-- time:",
        time.time() - step_t,
        "-- runnning time:",
        time.time() - step_t - total_reeval_t - total_write_t - total_metrics_t,
    )

    # Compute reeval metrics
    (
        reeval_repertoire,
        fit_reeval_repertoire,
        desc_reeval_repertoire,
        fit_var_repertoire,
        desc_var_repertoire,
        random_key,
    ) = reevaluation_fn(repertoire, random_key)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), reeval_repertoire.genotypes)
    reeval_metrics = metrics_function(reeval_repertoire)
    fit_reeval_metrics = metrics_function(fit_reeval_repertoire)
    desc_reeval_metrics = metrics_function(desc_reeval_repertoire)
    fit_var_metrics = metrics_function(fit_var_repertoire)
    desc_var_metrics = metrics_function(desc_var_repertoire)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), desc_var_metrics)
    if args.container in CONTAINER_REQUIRE_INCELL_SELECTION:
        in_cell_metrics, random_key = in_cell_metrics_function_fn(
            repertoire, random_key
        )
        (
            in_cell_reeval_repertoire,
            in_cell_fit_reeval_repertoire,
            in_cell_desc_reeval_repertoire,
            in_cell_fit_var_repertoire,
            in_cell_desc_var_repertoire,
            random_key,
        ) = in_cell_reevaluation_fn(repertoire, random_key)
        jax.tree_util.tree_map(
            lambda x: x.block_until_ready(), in_cell_reeval_repertoire.genotypes
        )
        in_cell_reeval_metrics = metrics_function(in_cell_reeval_repertoire)
        in_cell_fit_reeval_metrics = metrics_function(in_cell_fit_reeval_repertoire)
        in_cell_desc_reeval_metrics = metrics_function(in_cell_desc_reeval_repertoire)
        in_cell_fit_var_metrics = metrics_function(in_cell_fit_var_repertoire)
        in_cell_desc_var_metrics = metrics_function(in_cell_desc_var_repertoire)
    else:
        in_cell_reeval_repertoire = reeval_repertoire
        in_cell_fit_reeval_repertoire = fit_reeval_repertoire
        in_cell_desc_reeval_repertoire = desc_reeval_repertoire
        in_cell_fit_var_repertoire = fit_var_repertoire
        in_cell_desc_var_repertoire = desc_var_repertoire
        in_cell_metrics = metrics
        in_cell_reeval_metrics = reeval_metrics
        in_cell_fit_reeval_metrics = fit_reeval_metrics
        in_cell_desc_reeval_metrics = desc_reeval_metrics
        in_cell_fit_var_metrics = fit_var_metrics
        in_cell_desc_var_metrics = desc_var_metrics
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), in_cell_desc_var_metrics)
    reeval_t = time.time() - reeval_t
    total_reeval_t += reeval_t

    # Timer
    current_t = time.time() - step_t - total_reeval_t - total_write_t - total_metrics_t
    write_t = time.time()

    # Sanity check
    fitnesses = repertoire.fitnesses
    fitnesses = jnp.where(fitnesses == -jnp.inf, jnp.inf, fitnesses)
    if min(fitnesses) < -qd_offset:
        print("!!!WARNING!!! wrong min fitness value: ", -qd_offset)
        print("Got fitness value: ", min(fitnesses))
        print("This may lead to inacurate QD-Score.")
    reeval_fitnesses = reeval_repertoire.fitnesses
    reeval_fitnesses = jnp.where(
        reeval_fitnesses == -jnp.inf, jnp.inf, reeval_fitnesses
    )
    if min(reeval_fitnesses) < -qd_offset:
        print("!!!WARNING!!! wrong min fitness value: ", -qd_offset)
        print("Got fitness value: ", min(reeval_fitnesses))
        print("This may lead to inacurate QD-Score.")

    # If Parallel-Adaptive-Sampling, get num_samples
    if args.container == "Parallel-Adaptive-Sampling":
        num_samples = map_elites.num_samples
        batch_size = map_elites.batch_size
    else:
        num_samples = evals_per_offspring
        batch_size = args.batch_size

    # Add metrics
    write_metrics_csv(
        metrics_file,
        epoch,
        evals,
        current_t,
        metrics["qd_score"],
        metrics["coverage"],
        metrics["max_fitness"],
        reeval_metrics["qd_score"],
        reeval_metrics["coverage"],
        reeval_metrics["max_fitness"],
        fit_reeval_metrics["qd_score"],
        fit_reeval_metrics["coverage"],
        fit_reeval_metrics["max_fitness"],
        desc_reeval_metrics["qd_score"],
        desc_reeval_metrics["coverage"],
        desc_reeval_metrics["max_fitness"],
        fit_var_metrics["qd_score"],
        fit_var_metrics["coverage"],
        fit_var_metrics["max_fitness"],
        desc_var_metrics["qd_score"],
        desc_var_metrics["coverage"],
        desc_var_metrics["max_fitness"],
        num_samples,
        evals - previous_evals,
        batch_size,
    )
    write_metrics_csv(
        in_cell_metrics_file,
        epoch,
        evals,
        current_t,
        in_cell_metrics["qd_score"],
        in_cell_metrics["coverage"],
        in_cell_metrics["max_fitness"],
        in_cell_reeval_metrics["qd_score"],
        in_cell_reeval_metrics["coverage"],
        in_cell_reeval_metrics["max_fitness"],
        in_cell_fit_reeval_metrics["qd_score"],
        in_cell_fit_reeval_metrics["coverage"],
        in_cell_fit_reeval_metrics["max_fitness"],
        in_cell_desc_reeval_metrics["qd_score"],
        in_cell_desc_reeval_metrics["coverage"],
        in_cell_desc_reeval_metrics["max_fitness"],
        in_cell_fit_var_metrics["qd_score"],
        in_cell_fit_var_metrics["coverage"],
        in_cell_fit_var_metrics["max_fitness"],
        in_cell_desc_var_metrics["qd_score"],
        in_cell_desc_var_metrics["coverage"],
        in_cell_desc_var_metrics["max_fitness"],
        num_samples,
        evals - previous_evals,
        batch_size,
    )
    print("    -> Metrics saved in", metrics_file)
    print("    -> In cell metrics saved in", in_cell_metrics_file)

    # Write repertoire
    if epoch % args.archive_log_period == 0:
        repertoire.save(path=results_repertoire)
        reeval_repertoire.save(path=results_reeval_repertoire)
        fit_reeval_repertoire.save(path=results_fit_reeval_repertoire)
        desc_reeval_repertoire.save(path=results_desc_reeval_repertoire)
        fit_var_repertoire.save(path=results_fit_var_repertoire)
        desc_var_repertoire.save(path=results_desc_var_repertoire)
        in_cell_reeval_repertoire.save(path=results_in_cell_reeval_repertoire)
        in_cell_fit_reeval_repertoire.save(path=results_in_cell_fit_reeval_repertoire)
        in_cell_desc_reeval_repertoire.save(path=results_in_cell_desc_reeval_repertoire)
        in_cell_fit_var_repertoire.save(path=results_in_cell_fit_var_repertoire)
        in_cell_desc_var_repertoire.save(path=results_in_cell_desc_var_repertoire)
        print("    -> All repertoire saved, original in", results_repertoire)

    write_t = time.time() - write_t
    total_write_t += write_t

##############################
# Final metrics and analysis #
print(
    "\n    Ended at epoch:",
    epoch,
    "-- evals:",
    evals,
    "-- time:",
    time.time() - step_t,
    "-- runnning time:",
    time.time() - step_t - total_reeval_t - total_write_t - total_metrics_t,
)

# Compute reeval metrics
reeval_t = time.time()
(
    reeval_repertoire,
    fit_reeval_repertoire,
    desc_reeval_repertoire,
    fit_var_repertoire,
    desc_var_repertoire,
    random_key,
) = reevaluation_fn(repertoire, random_key)
jax.tree_util.tree_map(lambda x: x.block_until_ready(), reeval_repertoire.genotypes)
reeval_metrics = metrics_function(reeval_repertoire)
fit_reeval_metrics = metrics_function(fit_reeval_repertoire)
desc_reeval_metrics = metrics_function(desc_reeval_repertoire)
fit_var_metrics = metrics_function(fit_var_repertoire)
desc_var_metrics = metrics_function(desc_var_repertoire)
jax.tree_util.tree_map(lambda x: x.block_until_ready(), desc_var_metrics)
if args.container in CONTAINER_REQUIRE_INCELL_SELECTION:
    in_cell_metrics, random_key = in_cell_metrics_function_fn(repertoire, random_key)
    (
        in_cell_reeval_repertoire,
        in_cell_fit_reeval_repertoire,
        in_cell_desc_reeval_repertoire,
        in_cell_fit_var_repertoire,
        in_cell_desc_var_repertoire,
        random_key,
    ) = in_cell_reevaluation_fn(repertoire, random_key)
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready(), in_cell_reeval_repertoire.genotypes
    )
    in_cell_reeval_metrics = metrics_function(in_cell_reeval_repertoire)
    in_cell_fit_reeval_metrics = metrics_function(in_cell_fit_reeval_repertoire)
    in_cell_desc_reeval_metrics = metrics_function(in_cell_desc_reeval_repertoire)
    in_cell_fit_var_metrics = metrics_function(in_cell_fit_var_repertoire)
    in_cell_desc_var_metrics = metrics_function(in_cell_desc_var_repertoire)
else:
    in_cell_reeval_repertoire = reeval_repertoire
    in_cell_fit_reeval_repertoire = fit_reeval_repertoire
    in_cell_desc_reeval_repertoire = desc_reeval_repertoire
    in_cell_fit_var_repertoire = fit_var_repertoire
    in_cell_desc_var_repertoire = desc_var_repertoire
    in_cell_metrics = metrics
    in_cell_reeval_metrics = reeval_metrics
    in_cell_fit_reeval_metrics = fit_reeval_metrics
    in_cell_desc_reeval_metrics = desc_reeval_metrics
    in_cell_fit_var_metrics = fit_var_metrics
    in_cell_desc_var_metrics = desc_var_metrics
jax.tree_util.tree_map(lambda x: x.block_until_ready(), in_cell_desc_var_metrics)
reeval_t = time.time() - reeval_t
total_reeval_t += reeval_t

# Timer
current_t = time.time() - step_t - total_reeval_t - total_write_t - total_metrics_t

# If Parallel-Adaptive-Sampling, get num_samples
if args.container == "Parallel-Adaptive-Sampling":
    num_samples = map_elites.num_samples
    batch_size = map_elites.batch_size
else:
    num_samples = evals_per_offspring
    batch_size = args.batch_size

# Add metrics
write_metrics_csv(
    metrics_file,
    epoch,
    evals,
    current_t,
    metrics["qd_score"],
    metrics["coverage"],
    metrics["max_fitness"],
    reeval_metrics["qd_score"],
    reeval_metrics["coverage"],
    reeval_metrics["max_fitness"],
    fit_reeval_metrics["qd_score"],
    fit_reeval_metrics["coverage"],
    fit_reeval_metrics["max_fitness"],
    desc_reeval_metrics["qd_score"],
    desc_reeval_metrics["coverage"],
    desc_reeval_metrics["max_fitness"],
    fit_var_metrics["qd_score"],
    fit_var_metrics["coverage"],
    fit_var_metrics["max_fitness"],
    desc_var_metrics["qd_score"],
    desc_var_metrics["coverage"],
    desc_var_metrics["max_fitness"],
    num_samples,
    evals - previous_evals,
    batch_size,
)
write_metrics_csv(
    in_cell_metrics_file,
    epoch,
    evals,
    current_t,
    in_cell_metrics["qd_score"],
    in_cell_metrics["coverage"],
    in_cell_metrics["max_fitness"],
    in_cell_reeval_metrics["qd_score"],
    in_cell_reeval_metrics["coverage"],
    in_cell_reeval_metrics["max_fitness"],
    in_cell_fit_reeval_metrics["qd_score"],
    in_cell_fit_reeval_metrics["coverage"],
    in_cell_fit_reeval_metrics["max_fitness"],
    in_cell_desc_reeval_metrics["qd_score"],
    in_cell_desc_reeval_metrics["coverage"],
    in_cell_desc_reeval_metrics["max_fitness"],
    in_cell_fit_var_metrics["qd_score"],
    in_cell_fit_var_metrics["coverage"],
    in_cell_fit_var_metrics["max_fitness"],
    in_cell_desc_var_metrics["qd_score"],
    in_cell_desc_var_metrics["coverage"],
    in_cell_desc_var_metrics["max_fitness"],
    num_samples,
    evals - previous_evals,
    batch_size,
)
print("    -> Final metrics saved in", metrics_file)
print("    -> Final in_cell_metrics saved in", in_cell_metrics_file)

# Write repertoire
repertoire.save(path=results_repertoire)
reeval_repertoire.save(path=results_reeval_repertoire)
fit_reeval_repertoire.save(path=results_fit_reeval_repertoire)
desc_reeval_repertoire.save(path=results_desc_reeval_repertoire)
fit_var_repertoire.save(path=results_fit_var_repertoire)
desc_var_repertoire.save(path=results_desc_var_repertoire)
in_cell_reeval_repertoire.save(path=results_in_cell_reeval_repertoire)
in_cell_fit_reeval_repertoire.save(path=results_in_cell_fit_reeval_repertoire)
in_cell_desc_reeval_repertoire.save(path=results_in_cell_desc_reeval_repertoire)
in_cell_fit_var_repertoire.save(path=results_in_cell_fit_var_repertoire)
in_cell_desc_var_repertoire.save(path=results_in_cell_desc_var_repertoire)
print("    -> All final repertoire saved, original in", results_repertoire)


print("\nFinished run:", time.time() - step_t)
