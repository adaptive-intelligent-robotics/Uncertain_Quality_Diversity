from typing import Any, Callable

from qdax.core.containers.repertoire import Repertoire

from metrics_manager.archive_sampling_metrics_manager import (
    ArchiveSamplingMetricsManager,
)
from metrics_manager.deep_grid_metrics_manager import DeepGridMetricsManager
from metrics_manager.default_metrics_manager import MetricsManager
from metrics_manager.mome_metrics_manager import MOMEMetricsManager
from set_up_container import CONTAINER_REQUIRE_INCELL_SELECTION


def get_metrics_manager(
    args: Any,
    name: str,
    scoring_fn: Callable,
    reevaluation_fn: Callable,
    reevaluation_in_cell_fn: Callable,
    metric_repertoire: Repertoire,
) -> MetricsManager:
    """
    Create the correct metrics manager for current args.
    """

    if "MOME" in args.container:
        if (
            "Archive-Sampling" in args.container
            or "Adaptive-Sampling" in args.container
        ) and args.container != "Average-Archive-Sampling":
            metrics_manager = MOMEMetricsManager(
                args=args,
                name=name,
                scoring_fn=scoring_fn,
                reevaluation_fn=reevaluation_fn,
                reevaluation_in_cell_fn=reevaluation_in_cell_fn,
                metric_repertoire=metric_repertoire,
                DefaultMetricsManager=ArchiveSamplingMetricsManager,
            )
            print("    Metrics Manager: ArchiveSampling + MOME.")
        else:
            metrics_manager = MOMEMetricsManager(
                args=args,
                name=name,
                scoring_fn=scoring_fn,
                reevaluation_fn=reevaluation_fn,
                reevaluation_in_cell_fn=reevaluation_in_cell_fn,
                metric_repertoire=metric_repertoire,
                DefaultMetricsManager=MetricsManager,
            )
            print("    Metrics Manager: MOME.")
    elif args.container in CONTAINER_REQUIRE_INCELL_SELECTION:
        metrics_manager = DeepGridMetricsManager(
            args=args,
            name=name,
            scoring_fn=scoring_fn,
            reevaluation_fn=reevaluation_fn,
            reevaluation_in_cell_fn=reevaluation_in_cell_fn,
            metric_repertoire=metric_repertoire,
        )
        print("    Metrics Manager: In-Cell.")
    elif (
        "Archive-Sampling" in args.container or "Adaptive-Sampling" in args.container
    ) and args.container != "Average-Archive-Sampling":
        metrics_manager = ArchiveSamplingMetricsManager(
            args=args,
            name=name,
            scoring_fn=scoring_fn,
            reevaluation_fn=reevaluation_fn,
            reevaluation_in_cell_fn=reevaluation_in_cell_fn,
            metric_repertoire=metric_repertoire,
        )
        print("    Metrics Manager: ArchiveSampling.")
    else:
        metrics_manager = MetricsManager(
            args=args,
            name=name,
            scoring_fn=scoring_fn,
            reevaluation_fn=reevaluation_fn,
            reevaluation_in_cell_fn=reevaluation_in_cell_fn,
            metric_repertoire=metric_repertoire,
        )
        print("    Metrics Manager: Default.")
    return metrics_manager  # type: ignore
