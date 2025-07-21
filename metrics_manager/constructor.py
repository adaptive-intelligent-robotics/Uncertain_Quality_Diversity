from typing import Any, Callable, List

from qdax.core.containers.repertoire import Repertoire

from metrics_manager.deep_grid_metrics_manager import DeepGridMetricsManager
from metrics_manager.default_metrics_manager import MetricsManager
from metrics_manager.mome_metrics_manager import MOMEMetricsManager
from set_up_container import (
    CONTAINER_REEVAL_ARCHIVE,
    CONTAINER_REQUIRE_INCELL_SELECTION,
)
from set_up_environment import ENV_OPTIMISATION


def get_metrics_manager(
    args: Any,
    name: str,
    scoring_fn: Callable,
    metric_repertoire: Repertoire,
    evals_per_iter: int,
    min_bd: List,
    max_bd: List,
    qd_offset: float,
) -> MetricsManager:
    """
    Create the correct metrics manager for current args.
    """

    if args.container == "MOME-Reprod":
        metrics_manager = MOMEMetricsManager(
            args=args,
            name=name,
            scoring_fn=scoring_fn,
            metric_repertoire=metric_repertoire,
            evals_per_iter=evals_per_iter,
            min_bd=min_bd,
            max_bd=max_bd,
            qd_offset=qd_offset,
            DefaultMetricsManager=MetricsManager,
        )
    elif args.container in CONTAINER_REQUIRE_INCELL_SELECTION:
        metrics_manager = DeepGridMetricsManager(
            args=args,
            name=name,
            scoring_fn=scoring_fn,
            metric_repertoire=metric_repertoire,
            evals_per_iter=evals_per_iter,
            min_bd=min_bd,
            max_bd=max_bd,
            qd_offset=qd_offset,
        )
    else:
        metrics_manager = MetricsManager(
            args=args,
            name=name,
            scoring_fn=scoring_fn,
            metric_repertoire=metric_repertoire,
            evals_per_iter=evals_per_iter,
            min_bd=min_bd,
            max_bd=max_bd,
            qd_offset=qd_offset,
        )
    return metrics_manager


def create_algo_name(args: Any) -> str:
    """
    Name the algo from the content of args.
    """

    name = args.container + args.suffixe

    if args.emitter == "Mixing":
        if args.mutation == "isoline" and (
            args.iso_sigma != 0.005 or args.line_sigma != 0.05
        ):
            name += "-line-" + str(args.iso_sigma) + "-" + str(args.line_sigma)
        if args.mutation == "polynomial":
            name += "-poly-" + str(args.proportion_mutation) + "-eta" + str(args.eta)
    else:
        name += "-" + args.emitter

    if args.container == "Parallel-Adaptive-Sampling":
        name += "-" + args.eas_use_evals
        if args.eas_max_samples > 0:
            name += "-max" + str(args.eas_max_samples)
    if args.container == "Estimator-Parallel-Adaptive-Sampling":
        name += "-" + args.eas_use_evals
        name += "-max" + str(args.eas_max_samples)
    if args.container == "MAP-Elites-WeightedReprod":
        name += (
            "-f"
            + str(args.mer_delta_fitness)
            + "-r"
            + str(args.mer_delta_reproducibility)
        )
        name += "-" + str(args.mer_rho)
    if args.container == "MAP-Elites-DeltaReprod":
        name += (
            "-f"
            + str(args.mer_delta_fitness)
            + "-r"
            + str(args.mer_delta_reproducibility)
        )
    if args.container == "Archive-Sampling-WeightedReprod":
        name += (
            "-f"
            + str(args.mer_delta_fitness)
            + "-r"
            + str(args.mer_delta_reproducibility)
        )
        name += "-" + str(args.mer_rho)
    if args.container == "Archive-Sampling-DeltaReprod":
        name += (
            "-f"
            + str(args.mer_delta_fitness)
            + "-r"
            + str(args.mer_delta_reproducibility)
        )
    if args.container == "Parallel-Adaptive-Sampling-WeightedReprod":
        name += "-" + args.eas_use_evals
        name += "-max" + str(args.eas_max_samples)
        name += (
            "-f"
            + str(args.mer_delta_fitness)
            + "-r"
            + str(args.mer_delta_reproducibility)
        )
        name += "-" + str(args.mer_rho)
    if args.container == "Parallel-Adaptive-Sampling-DeltaReprod":
        name += "-" + args.eas_use_evals
        name += "-max" + str(args.eas_max_samples)
        name += (
            "-f"
            + str(args.mer_delta_fitness)
            + "-r"
            + str(args.mer_delta_reproducibility)
        )
    if args.container == "MOME-Reprod":
        if args.mome_biased_sampling:
            name += "-biased"
    if args.depth > 1:
        name += "-depth-" + str(args.depth)
    if args.reproducibility_objective:
        name += "-reproducibility"
    if args.num_samples != 0:
        name += "-sampling-" + str(args.num_samples)
        if args.fitness_extractor != "Average":
            name += "-" + args.fitness_extractor + "-fitextract"
        if args.fitness_reproducibility_extractor != "STD":
            name += "-" + args.fitness_reproducibility_extractor + "-fitreprodextract"
        if args.descriptor_extractor != "Average":
            name += "-" + args.descriptor_extractor + "-descextract"
        if args.descriptor_reproducibility_extractor != "STD":
            name += (
                "-" + args.descriptor_reproducibility_extractor + "-fitreprodextract"
            )
    if args.container in CONTAINER_REEVAL_ARCHIVE and args.archive_out_sampling:
        name += "-archive-out-sampling"
    if args.params_std > 0:
        name + "_params-var_" + str(args.params_std)
    elif args.env_name not in ENV_OPTIMISATION:
        if args.gaussian_vel:
            name += "_velnormal"
        if args.gaussian_pos:
            name += "_posnormal"
    if args.deterministic:
        name += "_deterministic"
    if name == "MAP-Elites":
        name = "Vanilla-MAP-Elites"
    if args.num_reevals != 0:
        if args.reeval_fitness_extractor != "Average":
            name += "-" + args.reeval_fitness_extractor + "-fitreeval"
        if args.reeval_fitness_reproducibility_extractor != "STD":
            name += (
                "-" + args.reeval_fitness_reproducibility_extractor + "-fitreprodreeval"
            )
        if args.reeval_descriptor_extractor != "Average":
            name += "-" + args.reeval_descriptor_extractor + "-descreeval"
        if args.reeval_descriptor_reproducibility_extractor != "STD":
            name += (
                "-"
                + args.reeval_descriptor_reproducibility_extractor
                + "-fitreprodreeval"
            )
    return name  # type: ignore
