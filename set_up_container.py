from typing import Any, Callable, Tuple

from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import Emitter
from qdax.core.map_elites import MAPElites
from qdax.core.mels import MELS
from qdax.custom_types import Metrics, RNGKey

from core.adaptive_sampling import AdaptiveSampling
from core.archive_sampling import ArchiveSampling
from core.archive_sampling_delta import ArchiveSamplingDelta
from core.archive_sampling_reprod import ArchiveSamplingReprod
from core.archive_sampling_weighted import ArchiveSamplingWeighted
from core.deep_grid import DeepGrid
from core.extract_map_elites import ExtractMAPElites
from core.map_elites_delta import MAPElitesDelta
from core.map_elites_depth import MAPElitesDepth
from core.map_elites_sampling import MAPElitesSampling
from core.map_elites_sampling_reprod import MAPElitesSamplingReprod
from core.map_elites_weighted import MAPElitesWeighted
from core.mome_reprod import MOMEReprod
from core.parallel_adaptive_sampling import ParallelAdaptiveSampling
from core.sampling import average, closest, iqr, mad, median, mode, std

# Container list
CONTAINER_LIST = [
    "MAP-Elites",
    "Extract-MAP-Elites",
    "Adaptive-Sampling",
    "Deep-Grid",
    "Archive-Sampling",
    "Parallel-Adaptive-Sampling",
    "MAP-Elites-Low-Spread",
    "MAP-Elites-Sampling-Reprod",
    "Archive-Sampling-Reprod",
    "MAP-Elites-Delta",
    "MAP-Elites-Weighted",
    "Archive-Sampling-Delta",
    "Archive-Sampling-Weighted",
    "MOME-Reprod",
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
    "Archive-Sampling-Reprod",
    "Archive-Sampling-Weighted",
    "Archive-Sampling-Delta",
]

# For batch-size setting
CONTAINER_EXTRACT_PROPORTION_RESAMPLE = [
    "Extract-MAP-Elites",
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


def _get_add_evals_per_iter(args: Any) -> int:
    """
    Return the number of additional evaluations independent
    from emitter or extraction caused by the container at hand.
    """
    add_evals_per_iter = 0
    if args.container in CONTAINER_REEVAL_ARCHIVE:
        add_evals_per_iter += (
            args.num_centroids * args.depth * args.as_repertoire_num_samples
        )
    return add_evals_per_iter


def get_batch_size(
    args: Any, sampling_size: int, evals_per_offspring: int
) -> Tuple[int, int, int, int]:
    """
    From the sampling_size and evals_per_offspring, return all the
    other information about sampling.

    Args:
        sampling_size
        evals_per_offspring
    Returns:
        batch_size: the overall batch-size of the algorithm
        init_batch_size: the batch size of the first iteration
        effective_batch_size: the effective batch size of the
            emitter for the following iterations
        real_evals_per_iter: the number of evaluations per iteration
    """

    # Compute evals per extract
    evals_per_extract = args.num_samples
    if args.container in CONTAINER_EXTRACT_PROPORTION_RESAMPLE:
        evals_per_extract = args.as_repertoire_num_samples

    # Compute additional evals per iteration
    add_evals_per_iter = _get_add_evals_per_iter(args=args)
    assert sampling_size > add_evals_per_iter, (
        "!!!ERROR!!! Missing sampling credit for evaluation, not enough for"
        + str(add_evals_per_iter)
        + "additional evaluations."
    )
    left_sampling_size = sampling_size - add_evals_per_iter

    # Infer init_batch_size
    init_batch_size = sampling_size // evals_per_offspring

    # Infer batch_size, real_evals_per_iter and effective_batch_size
    if args.container in CONTAINER_EXTRACT_PROPORTION_RESAMPLE:
        extract_sampling_size = int(
            left_sampling_size * args.extract_proportion_resample
        )
        extract_sampling_size = min(extract_sampling_size, args.extract_cap_resample)
        assert extract_sampling_size >= evals_per_extract, (
            "!!!ERROR!!! Missing sampling credit for evaluation, not enough left for"
            + evals_per_extract
            + "eval per extract."
        )

        effective_sampling_size = left_sampling_size - extract_sampling_size
        assert effective_sampling_size >= evals_per_offspring, (
            "!!!ERROR!!! Missing sampling credit for evaluation, not enough left for"
            + str(evals_per_offspring)
            + "eval per offspring."
        )

        effective_batch_size = effective_sampling_size // evals_per_offspring
        extract_batch_size = extract_sampling_size // evals_per_extract

        batch_size = effective_batch_size + extract_batch_size
        real_evals_per_iter = (
            effective_batch_size * evals_per_offspring
            + extract_batch_size * evals_per_extract
            + add_evals_per_iter
        )
        print("\nFor Extract:")
        print(
            f"Emitting {effective_batch_size} offspring sampled {evals_per_offspring} times."
        )
        print(f"Extracting {extract_batch_size} sampled {evals_per_extract} times.")

    else:
        assert left_sampling_size >= evals_per_offspring, (
            "!!!ERROR!!! Missing sampling credit for evaluation, not enough left for"
            + str(evals_per_offspring)
            + "eval per offspring."
        )
        batch_size = int(left_sampling_size // evals_per_offspring)
        effective_batch_size = batch_size
        real_evals_per_iter = batch_size * evals_per_offspring + add_evals_per_iter

    return batch_size, init_batch_size, effective_batch_size, real_evals_per_iter


def get_sampling_size(
    args: Any, batch_size: int, evals_per_offspring: int
) -> Tuple[int, int, int, int]:
    """
    From the batch_size and evals_per_offspring, return all the
    other information about sampling.

    Args:
        sampling_size
        evals_per_offspring
    Returns:
        sampling_size: the overall sampling-size of the algorithm
        init_batch_size: the batch size of the first iteration
        effective_batch_size: the effective batch size of the
            emitter for the following iterations
        real_evals_per_iter: the number of evaluations per iteration
    """

    # Compute evals per extract
    evals_per_extract = args.num_samples
    if args.container in CONTAINER_EXTRACT_PROPORTION_RESAMPLE:
        evals_per_extract = args.as_repertoire_num_samples

    add_evals_per_iter = _get_add_evals_per_iter(args=args)

    # Infer the effective batch-size
    effective_batch_size = batch_size
    extract_batch_size = int(batch_size * args.extract_proportion_resample)
    extract_batch_size = min(extract_batch_size, args.extract_cap_resample)
    effective_batch_size = batch_size - extract_batch_size

    # Infer the sampling-size and real_evals_per_iter
    if args.container in CONTAINER_EXTRACT_PROPORTION_RESAMPLE:
        sampling_size = (
            effective_batch_size * evals_per_offspring
            + (batch_size - effective_batch_size) * evals_per_extract
            + add_evals_per_iter
        )
    else:
        sampling_size = batch_size * evals_per_offspring + add_evals_per_iter
    real_evals_per_iter = sampling_size

    # Infer init_batch_size
    init_batch_size = batch_size

    return sampling_size, init_batch_size, effective_batch_size, real_evals_per_iter


def set_up_container(
    container_name: str,
    emitter: Emitter,
    scoring_fn: Callable,
    batch_size: int,
    effective_batch_size: int,
    sampling_size: int,
    num_samples: int,
    num_descriptors: int,
    depth: int,
    max_number_evals: int,
    extract_type: str,
    as_repertoire_num_samples: int,
    pas_max_samples: int,
    pas_use_evals: str,
    mer_delta_fitness: float,
    mer_delta_reproducibility: float,
    mer_rho: float,
    mome_pareto_front_max_length: int,
    mome_biased_sampling: bool,
    fitness_extractor: str,
    fitness_reproducibility_extractor: str,
    descriptor_extractor: str,
    descriptor_reproducibility_extractor: str,
    random_key: RNGKey,
) -> Tuple[MAPElites, RNGKey]:

    # Input check
    assert container_name in CONTAINER_LIST, "\n!!!ERROR!!! Invalid " + container_name
    assert num_samples > 0, "!!!ERROR!!! Invalid num_samples."
    assert as_repertoire_num_samples > 0, "!!!ERROR!!! Invalid repertoire_num_samples."
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
                num_samples=num_samples,
                fitness_extractor=EXTRACTOR_LIST[fitness_extractor],
                descriptor_extractor=EXTRACTOR_LIST[descriptor_extractor],
            )
        elif num_samples > 1:
            map_elites = MAPElitesSampling(
                scoring_function=scoring_fn,
                emitter=emitter,
                metrics_function=empty_metrics_function,
                num_samples=num_samples,
                fitness_extractor=EXTRACTOR_LIST[fitness_extractor],
                descriptor_extractor=EXTRACTOR_LIST[descriptor_extractor],
            )
        else:
            map_elites = MAPElites(
                scoring_function=scoring_fn,
                emitter=emitter,
                metrics_function=empty_metrics_function,
            )

    elif container_name == "MAP-Elites-Sampling-Reprod":
        assert (
            num_samples > 1
        ), "\n!!!ERROR!!! num_samples should be greater than 1 for MAP-Elites-Sampling-Reprod."
        map_elites = MAPElitesSamplingReprod(
            scoring_function=scoring_fn,
            emitter=emitter,
            metrics_function=empty_metrics_function,
            num_samples=num_samples,
            descriptor_extractor=EXTRACTOR_LIST[descriptor_extractor],
            descriptor_reproducibility_extractor=EXTRACTOR_LIST[
                descriptor_reproducibility_extractor
            ],
        )

    elif container_name == "MAP-Elites-Low-Spread":
        map_elites = MELS(
            scoring_function=scoring_fn,
            emitter=emitter,
            metrics_function=empty_metrics_function,
            num_samples=num_samples,
        )

    elif container_name == "Adaptive-Sampling":
        map_elites = AdaptiveSampling(
            scoring_function=scoring_fn,
            emitter=emitter,
            metrics_function=empty_metrics_function,
            depth=depth,
            max_number_evals=max_number_evals,
            sampling_size=sampling_size,
            num_descriptors=num_descriptors,
            fitness_extractor=EXTRACTOR_LIST[fitness_extractor],
            fitness_reproducibility_extractor=EXTRACTOR_LIST[
                fitness_reproducibility_extractor
            ],
            descriptor_extractor=EXTRACTOR_LIST[descriptor_extractor],
            descriptor_reproducibility_extractor=EXTRACTOR_LIST[
                descriptor_reproducibility_extractor
            ],
        )

    elif container_name == "Deep-Grid":
        map_elites = DeepGrid(
            scoring_function=scoring_fn,
            emitter=emitter,
            metrics_function=empty_metrics_function,
            depth=depth,
            num_samples=num_samples,
            fitness_extractor=EXTRACTOR_LIST[fitness_extractor],
            descriptor_extractor=EXTRACTOR_LIST[descriptor_extractor],
        )

    elif container_name == "Parallel-Adaptive-Sampling":
        assert (
            num_samples == 1
        ), "\n!!!ERROR!!! num_samples should be 1 in Parallel-Adaptive-Sampling."
        assert (
            as_repertoire_num_samples == 1
        ), "\n!!!ERROR!!! as_repertoire_num_samples does not impact this algo."
        map_elites = ParallelAdaptiveSampling(
            scoring_function=scoring_fn,
            emitter=emitter,
            metrics_function=empty_metrics_function,
            depth=depth,
            batch_size=batch_size,
            max_number_evals=max_number_evals,
            fitness_extractor=EXTRACTOR_LIST[fitness_extractor],
            fitness_reproducibility_extractor=EXTRACTOR_LIST[
                fitness_reproducibility_extractor
            ],
            descriptor_extractor=EXTRACTOR_LIST[descriptor_extractor],
            descriptor_reproducibility_extractor=EXTRACTOR_LIST[
                descriptor_reproducibility_extractor
            ],
            max_num_samples=pas_max_samples,
            sampling_size=sampling_size,
            use_evals=pas_use_evals,
        )

    elif container_name == "Archive-Sampling":
        map_elites = ArchiveSampling(
            scoring_function=scoring_fn,
            emitter=emitter,
            metrics_function=empty_metrics_function,
            depth=depth,
            max_number_evals=max_number_evals,
            num_samples=num_samples,
            repertoire_num_samples=as_repertoire_num_samples,
            fitness_extractor=EXTRACTOR_LIST[fitness_extractor],
            fitness_reproducibility_extractor=EXTRACTOR_LIST[
                fitness_reproducibility_extractor
            ],
            descriptor_extractor=EXTRACTOR_LIST[descriptor_extractor],
            descriptor_reproducibility_extractor=EXTRACTOR_LIST[
                descriptor_reproducibility_extractor
            ],
        )

    elif container_name == "Archive-Sampling-Reprod":
        map_elites = ArchiveSamplingReprod(
            scoring_function=scoring_fn,
            emitter=emitter,
            metrics_function=empty_metrics_function,
            depth=depth,
            max_number_evals=max_number_evals,
            num_samples=num_samples,
            repertoire_num_samples=as_repertoire_num_samples,
            fitness_extractor=EXTRACTOR_LIST[fitness_extractor],
            fitness_reproducibility_extractor=EXTRACTOR_LIST[
                fitness_reproducibility_extractor
            ],
            descriptor_extractor=EXTRACTOR_LIST[descriptor_extractor],
            descriptor_reproducibility_extractor=EXTRACTOR_LIST[
                descriptor_reproducibility_extractor
            ],
        )

    elif container_name == "Extract-MAP-Elites":
        map_elites = ExtractMAPElites(
            scoring_function=scoring_fn,
            emitter=emitter,
            metrics_function=empty_metrics_function,
            depth=depth,
            batch_size=batch_size,
            emit_batch_size=effective_batch_size,
            max_number_evals=max_number_evals,
            num_samples=num_samples,
            repertoire_num_samples=as_repertoire_num_samples,
            fitness_extractor=EXTRACTOR_LIST[fitness_extractor],
            fitness_reproducibility_extractor=EXTRACTOR_LIST[
                fitness_reproducibility_extractor
            ],
            descriptor_extractor=EXTRACTOR_LIST[descriptor_extractor],
            descriptor_reproducibility_extractor=EXTRACTOR_LIST[
                descriptor_reproducibility_extractor
            ],
            extract_type=extract_type,
        )

    elif container_name == "MAP-Elites-Weighted":
        assert (
            num_samples > 1
        ), "\n!!!ERROR!!! Requires num_samples > 1 to estimate reproducibility."
        assert (
            mer_delta_fitness > 0 or mer_delta_reproducibility > 0
        ), "\n!!!ERROR!!! No delta set."
        map_elites = MAPElitesWeighted(
            scoring_function=scoring_fn,
            emitter=emitter,
            metrics_function=empty_metrics_function,
            num_samples=num_samples,
            fitness_extractor=EXTRACTOR_LIST[fitness_extractor],
            fitness_reproducibility_extractor=EXTRACTOR_LIST[
                fitness_reproducibility_extractor
            ],
            descriptor_extractor=EXTRACTOR_LIST[descriptor_extractor],
            descriptor_reproducibility_extractor=EXTRACTOR_LIST[
                descriptor_reproducibility_extractor
            ],
            delta_fitness=mer_delta_fitness,
            delta_reproducibility=mer_delta_reproducibility,
            rho=mer_rho,
        )

    elif container_name == "MAP-Elites-Delta":
        assert (
            num_samples > 1
        ), "\n!!!ERROR!!! Requires num_samples > 1 to estimate reproducibility."
        assert (
            mer_delta_fitness > 0 or mer_delta_reproducibility > 0
        ), "\n!!!ERROR!!! No delta set."
        map_elites = MAPElitesDelta(
            scoring_function=scoring_fn,
            emitter=emitter,
            metrics_function=empty_metrics_function,
            num_samples=num_samples,
            fitness_extractor=EXTRACTOR_LIST[fitness_extractor],
            fitness_reproducibility_extractor=EXTRACTOR_LIST[
                fitness_reproducibility_extractor
            ],
            descriptor_extractor=EXTRACTOR_LIST[descriptor_extractor],
            descriptor_reproducibility_extractor=EXTRACTOR_LIST[
                descriptor_reproducibility_extractor
            ],
            delta_fitness=mer_delta_fitness,
            delta_reproducibility=mer_delta_reproducibility,
        )

    elif container_name == "MOME-Reprod":
        map_elites = MOMEReprod(
            scoring_function=scoring_fn,
            emitter=emitter,
            metrics_function=empty_metrics_function,
            num_samples=num_samples,
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

    elif container_name == "Archive-Sampling-Weighted":
        assert (
            num_samples > 1
        ), "\n!!!ERROR!!! Requires num_samples > 1 to estimate reproducibility."
        assert (
            mer_delta_fitness > 0 or mer_delta_reproducibility > 0
        ), "\n!!!ERROR!!! No delta set."
        map_elites = ArchiveSamplingWeighted(
            scoring_function=scoring_fn,
            emitter=emitter,
            metrics_function=empty_metrics_function,
            depth=depth,
            max_number_evals=max_number_evals,
            num_samples=num_samples,
            repertoire_num_samples=as_repertoire_num_samples,
            fitness_extractor=EXTRACTOR_LIST[fitness_extractor],
            fitness_reproducibility_extractor=EXTRACTOR_LIST[
                fitness_reproducibility_extractor
            ],
            descriptor_extractor=EXTRACTOR_LIST[descriptor_extractor],
            descriptor_reproducibility_extractor=EXTRACTOR_LIST[
                descriptor_reproducibility_extractor
            ],
            delta_fitness=mer_delta_fitness,
            delta_reproducibility=mer_delta_reproducibility,
            rho=mer_rho,
        )

    elif container_name == "Archive-Sampling-Delta":
        assert (
            num_samples > 1
        ), "\n!!!ERROR!!! Requires num_samples > 1 to estimate reproducibility."
        assert (
            mer_delta_fitness > 0 or mer_delta_reproducibility > 0
        ), "\n!!!ERROR!!! No delta set."
        map_elites = ArchiveSamplingDelta(
            scoring_function=scoring_fn,
            emitter=emitter,
            metrics_function=empty_metrics_function,
            depth=depth,
            max_number_evals=max_number_evals,
            num_samples=num_samples,
            repertoire_num_samples=as_repertoire_num_samples,
            fitness_extractor=EXTRACTOR_LIST[fitness_extractor],
            fitness_reproducibility_extractor=EXTRACTOR_LIST[
                fitness_reproducibility_extractor
            ],
            descriptor_extractor=EXTRACTOR_LIST[descriptor_extractor],
            descriptor_reproducibility_extractor=EXTRACTOR_LIST[
                descriptor_reproducibility_extractor
            ],
            delta_fitness=mer_delta_fitness,
            delta_reproducibility=mer_delta_reproducibility,
        )

    else:
        assert 0, f"\n!!!ERROR!!! Undefined container {container_name}."

    return map_elites, random_key
