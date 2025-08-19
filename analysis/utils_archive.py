from typing import Any, List, Optional, Tuple

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from qdax.utils.plotting import _get_projection_in_2d, get_voronoi_finite_polygons_2d


def plot_one_paper_archive(
    centroids: jnp.ndarray,
    descriptors: jnp.ndarray,
    fitnesses: jnp.ndarray,
    ax: Any,
    minval: jnp.ndarray,
    maxval: jnp.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar: Optional[bool] = True,
    colorbar_below: Optional[bool] = False,
) -> None:
    """Main function."""

    my_cmap = cm.viridis

    num_descriptors = centroids.shape[1]
    if num_descriptors != 2:
        _sub_plot_multidimensional(
            descriptors=descriptors,
            fitnesses=fitnesses,
            minval=jnp.asarray(minval),
            maxval=jnp.asarray(maxval),
            grid_shape=(6, 6, 6, 6),
            ax=ax,
            my_cmap=my_cmap,
            vmin=vmin,
            vmax=vmax,
            colorbar=colorbar,
            colorbar_below=colorbar_below,
        )
    else:
        _sub_plot(
            centroids=centroids,
            descriptors=descriptors,
            fitnesses=fitnesses,
            ax=ax,
            my_cmap=my_cmap,
            minval=minval,
            maxval=maxval,
            vmin=vmin,
            vmax=vmax,
            colorbar=colorbar,
            colorbar_below=colorbar_below,
        )


def _sub_plot(
    centroids: jnp.ndarray,
    descriptors: jnp.ndarray,
    fitnesses: jnp.ndarray,
    ax: Any,
    my_cmap: Any,
    minval: jnp.ndarray,
    maxval: jnp.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar: Optional[bool] = True,
    colorbar_below: Optional[bool] = False,
) -> None:
    """2-dimensional archives."""

    grid_empty = fitnesses == -jnp.inf

    if vmin is None:
        vmin = float(jnp.min(fitnesses[~grid_empty]))
    if vmax is None:
        vmax = float(jnp.max(fitnesses[~grid_empty]))

    assert (
        len(np.array(minval).shape) < 2
    ), f"minval : {minval} should be float or couple of floats"
    assert (
        len(np.array(maxval).shape) < 2
    ), f"maxval : {maxval} should be float or couple of floats"

    if len(np.array(minval).shape) == 0 and len(np.array(maxval).shape) == 0:
        ax.set_xlim(minval, maxval)
        ax.set_ylim(minval, maxval)
    else:
        ax.set_xlim(minval[0], maxval[0])
        ax.set_ylim(minval[1], maxval[1])

    ax.set(adjustable="box", aspect="equal")

    # create the regions and vertices from centroids
    regions, vertices = get_voronoi_finite_polygons_2d(centroids)

    norm = Normalize(vmin=vmin, vmax=vmax)
    non_inf_fitnesses = np.where(fitnesses > -jnp.inf, fitnesses, jnp.inf)
    if any(non_inf_fitnesses < vmin):
        print(f"!!!WARNING!!! Got out of range fitnesses: smaller than {vmin}.")
        fitnesses = fitnesses.at[non_inf_fitnesses < vmin].set(vmin)
    if any(fitnesses > vmax):
        print(f"!!!WARNING!!! Got out of range fitnesses: greater than {vmax}.")
        fitnesses = fitnesses.at[fitnesses > vmax].set(vmax)

    # fill the plot with contours
    for region in regions:
        polygon = vertices[region]
        ax.fill(*zip(*polygon), alpha=0.05, edgecolor="black", facecolor="white", lw=1)

    # fill the plot with the colors
    for idx, fitness in enumerate(fitnesses):
        if fitness > -jnp.inf:
            region = regions[idx]
            polygon = vertices[region]

            ax.fill(*zip(*polygon), alpha=0.8, color=my_cmap(norm(fitness)))

    # Add points location
    descriptors = descriptors[~grid_empty]
    ax.scatter(
        descriptors[:, 0],
        descriptors[:, 1],
        c=fitnesses[~grid_empty],
        cmap=my_cmap,
        s=10,
        zorder=0,
    )

    # aesthetic
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=my_cmap), cax=cax)
        cbar.ax.tick_params(labelsize=48)
    elif colorbar_below:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        cbar = plt.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=my_cmap),
            cax=cax,
            orientation="horizontal",
        )
        cbar.ax.tick_params(labelsize=48)

    ax.tick_params(
        left=False, right=False, labelleft=False, labelbottom=False, bottom=False
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.set_aspect("equal")


def _sub_plot_multidimensional(
    descriptors: jnp.ndarray,
    fitnesses: jnp.ndarray,
    ax: Any,
    my_cmap: Any,
    minval: jnp.ndarray,
    maxval: jnp.ndarray,
    grid_shape: Tuple[int, ...],
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar: Optional[bool] = True,
    colorbar_below: Optional[bool] = False,
) -> None:
    """More-dimensional archives."""

    is_grid_empty = fitnesses.ravel() == -jnp.inf
    num_descriptors = descriptors.shape[1]

    if isinstance(grid_shape, tuple):
        assert (
            len(grid_shape) == num_descriptors
        ), "grid_shape should have the same length as num_descriptors"
    else:
        raise ValueError("resolution should be a tuple")

    assert np.size(minval) == num_descriptors or np.size(minval) == 1, (
        f"minval : {minval} should either be of size 1 "
        f"or have the same size as the number of descriptors: {num_descriptors}"
    )
    assert np.size(maxval) == num_descriptors or np.size(maxval) == 1, (
        f"maxval : {maxval} should either be of size 1 "
        f"or have the same size as the number of descriptors: {num_descriptors}"
    )

    non_empty_descriptors = descriptors[~is_grid_empty]
    non_empty_fitnesses = fitnesses[~is_grid_empty]

    # hack when the descriptor hits the limit
    if jnp.any(non_empty_descriptors >= 1):
        indexes_exception = jnp.argwhere(non_empty_descriptors >= 1)
        for i in range(indexes_exception.shape[0]):
            non_empty_descriptors = non_empty_descriptors.at[indexes_exception[i]].set(
                0.999
            )

    # convert the descriptors to integer coordinates, depending on the resolution.
    resolutions_array = jnp.array(grid_shape)
    descriptors_integers = jnp.asarray(
        jnp.floor(
            resolutions_array * (non_empty_descriptors - minval) / (maxval - minval)
        ),
        dtype=jnp.int32,
    )

    # total number of grid cells along each dimension of the grid
    size_grid_x = np.prod(np.array(grid_shape[0::2]))
    size_grid_y = np.prod(np.array(grid_shape[1::2]), dtype=int)

    # initialise the grid
    grid_2d = np.full(
        (size_grid_x.item(), size_grid_y.item()),
        fill_value=jnp.nan,
    )

    # put solutions in the grid according to their projected 2-dimensional coordinates
    for desc, fit in zip(descriptors_integers, non_empty_fitnesses):
        # if jnp.any(desc > 5):
        # print("Desc: ", desc)
        projection_2d = _get_projection_in_2d(desc, grid_shape)
        if jnp.isnan(grid_2d[projection_2d]) or fit.item() > grid_2d[projection_2d]:
            grid_2d[projection_2d] = fit.item()

    # create the plot object
    ax.set(adjustable="box", aspect="equal")

    if vmin is None:
        vmin = float(jnp.min(non_empty_fitnesses))
    if vmax is None:
        vmax = float(jnp.max(non_empty_fitnesses))

    ax.imshow(
        grid_2d.T,
        origin="lower",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
        cmap=my_cmap,
    )

    norm = Normalize(vmin=vmin, vmax=vmax)

    # aesthetic
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=my_cmap), cax=cax)
        cbar.ax.tick_params(labelsize=48)
    elif colorbar_below:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        cbar = plt.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=my_cmap),
            cax=cax,
            orientation="horizontal",
        )
        cbar.ax.tick_params(labelsize=48)

    ax.set_aspect("equal")

    def _get_ticks_positions(
        total_size_grid_axis: int, step_ticks_on_axis: int
    ) -> jnp.ndarray:
        """
        Get the positions of the ticks on the grid axis.
        Args:
            total_size_grid_axis: total size of the grid axis
            step_ticks_on_axis: step of the ticks
        Returns:
            The positions of the ticks on the plot.
        """
        return np.arange(0, total_size_grid_axis + 1, step_ticks_on_axis) - 0.5

    # Ticks position
    major_ticks_x = _get_ticks_positions(
        size_grid_x.item(), step_ticks_on_axis=np.prod(grid_shape[2::2]).item()
    )
    minor_ticks_x = _get_ticks_positions(
        size_grid_x.item(), step_ticks_on_axis=np.prod(grid_shape[4::2]).item()
    )
    major_ticks_y = _get_ticks_positions(
        size_grid_y.item(), step_ticks_on_axis=np.prod(grid_shape[3::2]).item()
    )
    minor_ticks_y = _get_ticks_positions(
        size_grid_y.item(), step_ticks_on_axis=np.prod(grid_shape[5::2]).item()
    )

    ax.set_xticks(
        major_ticks_x,
    )
    ax.set_xticks(
        minor_ticks_x,
        minor=True,
    )
    ax.set_yticks(
        major_ticks_y,
    )
    ax.set_yticks(
        minor_ticks_y,
        minor=True,
    )

    ax.tick_params(
        left=False, right=False, labelleft=False, labelbottom=False, bottom=False
    )

    # Ticks aesthetics
    # ax.tick_params(
    #    which="minor",
    #    color="gray",
    #    labelcolor="gray",
    #    size=5,
    # )
    # ax.tick_params(
    #    which="major",
    #    labelsize=32,
    #    size=7,
    # )

    ax.grid(which="minor", alpha=1.0, color="#000000", linewidth=0.5)
    if len(grid_shape) > 2:
        ax.grid(which="major", alpha=1.0, color="#000000", linewidth=2.5)

    # def _get_positions_labels(
    #    _minval: float, _maxval: float, _number_ticks: int, _step_labels_ticks: int
    # ) -> List[str]:
    #    positions = jnp.linspace(_minval, _maxval, num=_number_ticks)

    #    list_str_positions = []
    #    for index_tick, position in enumerate(positions):
    #        if index_tick % _step_labels_ticks != 0:
    #            character = ""
    #        else:
    #            character = f"{position:.2E}"
    #        list_str_positions.append(character)
    #    # forcing the last tick label
    #    list_str_positions[-1] = f"{positions[-1]:.2E}"
    #    return list_str_positions

    # number_label_ticks = 4

    # if len(major_ticks_x) // number_label_ticks > 0:
    #    ax.set_xticklabels(
    #        _get_positions_labels(
    #            _minval=minval[0],
    #            _maxval=maxval[0],
    #            _number_ticks=len(major_ticks_x),
    #            _step_labels_ticks=len(major_ticks_x) // number_label_ticks,
    #        )
    #    )
    # if len(major_ticks_y) // number_label_ticks > 0:
    #    ax.set_yticklabels(
    #        _get_positions_labels(
    #            _minval=minval[1],
    #            _maxval=maxval[1],
    #            _number_ticks=len(major_ticks_y),
    #            _step_labels_ticks=len(major_ticks_y) // number_label_ticks,
    #        )
    #    )
