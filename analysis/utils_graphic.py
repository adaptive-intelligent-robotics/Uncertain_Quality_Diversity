from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import PathPatch


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


def adjust_box_widths(g: Any, fac: Any) -> None:
    """
    Adjust the widths of a seaborn-generated boxplot.
    """
    # Iterating through Axes instances
    for ax in g.axes:

        # Iterating through axes artists:
        for c in ax.get_children():

            # Searching for PathPatches
            if isinstance(c, PathPatch):
                # Getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5 * (xmin + xmax)
                xhalf = 0.5 * (xmax - xmin)

                # Setting new width of box
                xmin_new = xmid - fac * xhalf
                xmax_new = xmid + fac * xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # Setting new width of median line
                for line in ax.lines:
                    xdata = line.get_xdata()
                    if len(xdata) == 2 and xdata[0] == xmin and xdata[1] == xmax:
                        line.set_xdata([xmin_new, xmax_new])
