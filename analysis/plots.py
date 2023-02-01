from typing import Any, Tuple

import numpy as np
import pandas as pd
import seaborn as sns

#################
# Diverse utils #


def first_second_third_quartile(
    self: Any, vals: Any, grouper: Any, units: Any = None
) -> Tuple[Any, Any, Any]:
    """
    Utils used to plot first second third quantile as
    shaded area around the main graph.
    """

    # Group and get the aggregation estimate
    grouped = vals.groupby(grouper, sort=self.sort)
    est = grouped.agg("median")
    min_val = grouped.quantile(0.25)
    max_val = grouped.quantile(0.75)
    cis = pd.DataFrame(
        np.c_[min_val, max_val], index=est.index, columns=["low", "high"]
    ).stack()

    # Unpack the CIs into "wide" format for plotting
    if cis.notnull().any():
        cis = cis.unstack().reindex(est.index)
    else:
        cis = None
    return est.index, est, cis


def customize_axis(ax: Any) -> Any:
    """
    Customise axis for plots.
    """

    # Remove unused axis
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.tick_params(axis="y", length=0)

    # Offset the spines
    for spine in ax.spines.values():
        spine.set_position(("outward", 5))

    # Put the grid behind
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.9", linestyle="--", linewidth=1.5)

    return ax


#################################
# Main low level plot functions #


def sub_plot(
    x: str,
    y: str,
    data: pd.DataFrame,
    ax: Any,
    xlabel: str,
    ylabel: str,
    markers: bool = False,
    hue: str = "algo",
) -> None:
    """Subplot function for lineplot."""
    sns.lineplot(
        x=x,
        y=y,
        data=data,
        hue=hue,
        estimator=np.median,
        ci=None,
        style=hue,
        ax=ax,
        markers=markers,
        dashes=not (markers),
    )
    ax.set_xlabel(xlabel, labelpad=7)
    ax.set_ylabel(ylabel, labelpad=7)
    customize_axis(ax)
