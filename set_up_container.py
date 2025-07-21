from typing import Any, Callable, Tuple

from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import Emitter
from qdax.core.map_elites import MAPElites
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, Metrics, RNGKey

from core.archive_sampling import ArchiveSampling
from core.archive_sampling_delta_reprod import ArchiveSamplingDeltaReprod
from core.deep_grid import DeepGrid
from core.map_elites_delta_reprod import MAPElitesDeltaReprod
from core.map_elites_depth import MAPElitesDepth
from core.mels import MELS
from core.mome_reprod import MOMEReprod
from core.parallel_adaptive_sampling import ParallelAdaptiveSampling
from core.sampling import average, closest, iqr, mad, median, mode, std

# Container list
CONTAINER_LIST = [
    "MAP-Elites",
    "MAP-Elites-Low-Spread",
    "Archive-Sampling",
    "Deep-Grid",
    "Parallel-Adaptive-Sampling",
    "MAP-Elites-WeightedReprod",
    "MAP-Elites-DeltaReprod",
    "MOME-Reprod",
    "Archive-Sampling-WeightedReprod",
    "Archive-Sampling-DeltaReprod",
]
# Container that require to get all the samples and not only expectation
CONTAINER_REQUIRE_ALL_SAMPLES = [
    "MAP-Elites-WeightedReprod",
    "MAP-Elites-DeltaReprod",
    "MOME-Reprod",
]
# Container which code handle the sampling internally
CONTAINER_NO_SAMPLES = [
    "MAP-Elites-Low-Spread",
]
# Container that uses an in-cell selection to compute reevaluation stat
# WARNING "sample_all_cell" method needs to be implemented.
CONTAINER_REQUIRE_INCELL_SELECTION = [
    "Deep-Grid",
]
# Container that reeval the archive periodically
CONTAINER_REEVAL_ARCHIVE = [
    "Archive-Sampling",
    "Parallel-Adaptive-Sampling",
    "Archive-Sampling-WeightedReprod",
    "Archive-Sampling-DeltaReprod",
]

# Extractor list
EXTRACTOR_LIST = {
    "Average": average,
    "Median": median,
    "Mode": mode,
    "Closest": closest,
    "STD": std,
    "MAD": mad,
    "IQR": iqr,
}


def get_add_evals_per_iter(args: Any) -> int:
    add_evals_per_iter = 0
    if args.container in CONTAINER_REEVAL_ARCHIVE and not (args.archive_out_sampling):
        add_evals_per_iter += args.num_centroids * args.depth
    return add_evals_per_iter


def set_up_container(
    container_name: str,
    emitter_name: str,
    emitter: Emitter,
    num_iterations: int,
    batch_size: int,
    init_policies_fn: Callable[[int, RNGKey], Tuple[Genotype, RNGKey]],
    sampling_size: int,
    scoring_fn: Callable[
        [Genotype, RNGKey],
        Tuple[Fitness, Descriptor, ExtraScores, RNGKey],
    ],
    num_samples: int,
    depth: int,
    mutation: str,
    line_sigma: float,
    eta: float,
    eas_max_samples: int,
    eas_use_evals: str,
    eas_archive_out_sampling: bool,
    fitness_extractor: str,
    fitness_reproducibility_extractor: str,
    descriptor_extractor: str,
    descriptor_reproducibility_extractor: str,
    mer_delta_fitness: float,
    mer_delta_reproducibility: float,
    mer_rho: float,
    mome_pareto_front_max_length: int,
    mome_biased_sampling: bool,
) -> MAPElites:

    # Input check
    assert container_name in CONTAINER_LIST, "\n!!!ERROR!!! Invalid " + container_name
    if container_name in CONTAINER_REEVAL_ARCHIVE:
        assert (
            num_samples > 0 or eas_max_samples > 0
        ), "!!!ERROR!!! Require sampling (can be 1)."
        print("!!!WARNING!!! num_samples does NOT apply for archive reevaluation.")
    if container_name in CONTAINER_REQUIRE_ALL_SAMPLES:
        assert num_samples > 0, "\n!!!ERROR!!!" + container_name + " require sampling."
    assert (
        fitness_extractor in EXTRACTOR_LIST.keys()
    ), "\n !!!ERROR!!! invalid fitness_extractor."
    assert (
        fitness_reproducibility_extractor in EXTRACTOR_LIST.keys()
    ), "\n !!!ERROR!!! invalid fitness_reproducibility_extractor."
    assert (
        descriptor_extractor in EXTRACTOR_LIST.keys()
    ), "\n !!!ERROR!!! invalid descriptor_extractor."
    assert (
        descriptor_reproducibility_extractor in EXTRACTOR_LIST.keys()
    ), "\n !!!ERROR!!! invalid descriptor_reproducibility_extractor."

    # Initialise all the algos with an empty metrics function
    # So the metrics computation is not timed as part of the algorithms
    def empty_metrics_function(repertoire: Repertoire) -> Metrics:
        return {}

    # Instantiate MAP-Elites
    if container_name == "MAP-Elites":
        if depth > 1:
            map_elites = MAPElitesDepth(
                scoring_function=scoring_fn,
                emitter=emitter,
                metrics_function=empty_metrics_function,
                depth=depth,
            )
        else:
            map_elites = MAPElites(
                scoring_function=scoring_fn,
                emitter=emitter,
                metrics_function=empty_metrics_function,
            )
    elif container_name == "MAP-Elites-Low-Spread":
        assert num_samples > 0, "\n!!!ERROR!!! MELS require num-samples."
        map_elites = MELS(
            scoring_function=scoring_fn,
            emitter=emitter,
            metrics_function=empty_metrics_function,
            num_samples=num_samples,
        )
    elif container_name == "Deep-Grid":
        map_elites = DeepGrid(
            scoring_function=scoring_fn,
            emitter=emitter,
            metrics_function=empty_metrics_function,
            depth=depth,
        )
    elif container_name == "Parallel-Adaptive-Sampling":
        map_elites = ParallelAdaptiveSampling(
            scoring_function=scoring_fn,
            emitter=emitter,
            metrics_function=empty_metrics_function,
            depth=depth,
            num_iterations=num_iterations,
            fitness_extractor=EXTRACTOR_LIST[fitness_extractor],
            fitness_reproducibility_extractor=EXTRACTOR_LIST[
                fitness_reproducibility_extractor
            ],
            descriptor_extractor=EXTRACTOR_LIST[descriptor_extractor],
            descriptor_reproducibility_extractor=EXTRACTOR_LIST[
                descriptor_reproducibility_extractor
            ],
            max_num_samples=eas_max_samples,
            sampling_size=sampling_size,
            batch_size=batch_size,
            use_evals=eas_use_evals,
            archive_out_sampling=eas_archive_out_sampling,
        )
    elif container_name == "Archive-Sampling":
        map_elites = ArchiveSampling(
            scoring_function=scoring_fn,
            emitter=emitter,
            metrics_function=empty_metrics_function,
            depth=depth,
            num_iterations=num_iterations,
            num_samples=num_samples,
            fitness_extractor=EXTRACTOR_LIST[fitness_extractor],
            fitness_reproducibility_extractor=EXTRACTOR_LIST[
                fitness_reproducibility_extractor
            ],
            descriptor_extractor=EXTRACTOR_LIST[descriptor_extractor],
            descriptor_reproducibility_extractor=EXTRACTOR_LIST[
                descriptor_reproducibility_extractor
            ],
        )
    elif container_name == "MAP-Elites-WeightedReprod":
        map_elites = MAPElitesDeltaReprod(
            scoring_function=scoring_fn,
            emitter=emitter,
            metrics_function=empty_metrics_function,
            fitness_extractor=EXTRACTOR_LIST[fitness_extractor],
            fitness_reproducibility_extractor=EXTRACTOR_LIST[
                fitness_reproducibility_extractor
            ],
            descriptor_extractor=EXTRACTOR_LIST[descriptor_extractor],
            descriptor_reproducibility_extractor=EXTRACTOR_LIST[
                descriptor_reproducibility_extractor
            ],
            use_weighting=True,
            delta_fitness=mer_delta_fitness,
            delta_reproducibility=mer_delta_reproducibility,
            rho=mer_rho,
        )
    elif container_name == "MAP-Elites-DeltaReprod":
        map_elites = MAPElitesDeltaReprod(
            scoring_function=scoring_fn,
            emitter=emitter,
            metrics_function=empty_metrics_function,
            fitness_extractor=EXTRACTOR_LIST[fitness_extractor],
            fitness_reproducibility_extractor=EXTRACTOR_LIST[
                fitness_reproducibility_extractor
            ],
            descriptor_extractor=EXTRACTOR_LIST[descriptor_extractor],
            descriptor_reproducibility_extractor=EXTRACTOR_LIST[
                descriptor_reproducibility_extractor
            ],
            use_weighting=False,
            delta_fitness=mer_delta_fitness,
            delta_reproducibility=mer_delta_reproducibility,
            rho=mer_rho,
        )
    elif container_name == "MOME-Reprod":
        map_elites = MOMEReprod(
            scoring_function=scoring_fn,
            emitter=emitter,
            metrics_function=empty_metrics_function,
            fitness_extractor=EXTRACTOR_LIST[fitness_extractor],
            fitness_reproducibility_extractor=EXTRACTOR_LIST[
                fitness_reproducibility_extractor
            ],
            descriptor_extractor=EXTRACTOR_LIST[descriptor_extractor],
            descriptor_reproducibility_extractor=EXTRACTOR_LIST[
                descriptor_reproducibility_extractor
            ],
            biased_sampling=mome_biased_sampling,
            pareto_front_max_length=mome_pareto_front_max_length,
        )
    elif container_name == "Archive-Sampling-WeightedReprod":
        map_elites = ArchiveSamplingDeltaReprod(
            scoring_function=scoring_fn,
            emitter=emitter,
            metrics_function=empty_metrics_function,
            depth=depth,
            num_iterations=num_iterations,
            num_samples=num_samples,
            fitness_extractor=EXTRACTOR_LIST[fitness_extractor],
            fitness_reproducibility_extractor=EXTRACTOR_LIST[
                fitness_reproducibility_extractor
            ],
            descriptor_extractor=EXTRACTOR_LIST[descriptor_extractor],
            descriptor_reproducibility_extractor=EXTRACTOR_LIST[
                descriptor_reproducibility_extractor
            ],
            use_weighting=True,
            delta_fitness=mer_delta_fitness,
            delta_reproducibility=mer_delta_reproducibility,
            rho=mer_rho,
        )
    elif container_name == "Archive-Sampling-DeltaReprod":
        map_elites = ArchiveSamplingDeltaReprod(
            scoring_function=scoring_fn,
            emitter=emitter,
            metrics_function=empty_metrics_function,
            depth=depth,
            num_iterations=num_iterations,
            num_samples=num_samples,
            fitness_extractor=EXTRACTOR_LIST[fitness_extractor],
            fitness_reproducibility_extractor=EXTRACTOR_LIST[
                fitness_reproducibility_extractor
            ],
            descriptor_extractor=EXTRACTOR_LIST[descriptor_extractor],
            descriptor_reproducibility_extractor=EXTRACTOR_LIST[
                descriptor_reproducibility_extractor
            ],
            use_weighting=False,
            delta_fitness=mer_delta_fitness,
            delta_reproducibility=mer_delta_reproducibility,
            rho=mer_rho,
        )

    else:
        assert 0, f"\n!!!ERROR!!! Undefined container {container_name}."

    return map_elites
