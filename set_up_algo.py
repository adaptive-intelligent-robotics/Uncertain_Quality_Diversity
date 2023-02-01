from functools import partial
from typing import Any, Callable, Tuple

import jax
from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.pga_me_emitter import PGAMEConfig, PGAMEEmitter
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, Metrics, RNGKey
from qdax.utils.metrics import default_qd_metrics

from core.archive_sampling import ArchiveSampling
from core.deep_grid import DeepGrid
from core.emitters.random_emitter import RandomEmitter
from core.map_elites import MAPElites
from core.map_elites_depth import MAPElitesDepth
from core.parallel_adaptive_sampling import ParallelAdaptiveSampling


def set_up_algo(
    container_name: str,
    emitter_name: str,
    num_iterations: int,
    batch_size: int,
    sampling_size: int,
    env: int,
    scoring_fn: Callable[
        [Genotype, RNGKey],
        Tuple[Fitness, Descriptor, ExtraScores, RNGKey],
    ],
    num_descriptors: int,
    min_genotype: float,
    max_genotype: float,
    policy_structure: Any,
    init_policies: Genotype,
    depth: int,
    eas_max_samples: int,
    eas_use_evals: str,
    eas_archive_out_sampling: bool,
    qd_offset: float,
    use_median: bool,
) -> Tuple[Callable[[Repertoire], Metrics], MAPElites]:
    # Define emitter
    variation_fn = partial(isoline_variation, iso_sigma=0.005, line_sigma=0.05)
    if emitter_name == "Random":
        emitter = RandomEmitter(
            batch_size=batch_size,
            init_genotype=jax.tree_map(lambda x: x[0], init_policies),
            min_genotype=min_genotype,
            max_genotype=max_genotype,
        )
    elif emitter_name == "Mixing":
        emitter = MixingEmitter(
            mutation_fn=None,
            variation_fn=variation_fn,
            variation_percentage=1.0,
            batch_size=batch_size,
        )
    elif emitter_name == "PGA":
        pga_config = PGAMEConfig(
            env_batch_size=batch_size,
        )
        emitter = PGAMEEmitter(
            config=pga_config,
            policy_network=policy_structure,
            env=env,
            variation_fn=variation_fn,
        )

    metrics_function = partial(default_qd_metrics, qd_offset=qd_offset)

    def empty_metrics_function(repertoire: Any) -> Metrics:
        return {}

    # Instantiate MAP-Elites
    if container_name == "MAP-Elites":
        if depth != 0:
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
            max_num_samples=eas_max_samples,
            depth=depth,
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
        )

    return metrics_function, map_elites
