import argparse
import os
import traceback

import pandas as pd
import seaborn as sns

from analysis.config import (
    always_name,
    env_max_xaxis,
    env_order,
    not_name,
    replace_name,
)
from analysis.load_archives import find_min_max
from analysis.load_p_values import p_values
from analysis.load_reproducibilities import load_reproducibilities
from analysis.load_results import load_results, sort_data

#########
# Input #

parser = argparse.ArgumentParser()

# Folder
parser.add_argument("--results", default="results", type=str)
parser.add_argument("--plots", default="plots", type=str)

# For debugging
parser.add_argument("--errors", action="store_true")

# Main plots
parser.add_argument("--plot-metrics", action="store_true")
parser.add_argument("--plot-convergence-metrics", action="store_true")
parser.add_argument("--plot-time", action="store_true")

# Replay from archive plots
parser.add_argument("--plot-replay-archive", action="store_true")
parser.add_argument("--plot-replay-reward", action="store_true")
parser.add_argument("--plot-replay-trajectory", action="store_true")
parser.add_argument("--plot-replay-gr", action="store_true")

# Archive plots
parser.add_argument("--plot-archives", action="store_true")
parser.add_argument("--plot-all-archives", action="store_true")
parser.add_argument("--plot-archive-profiles", action="store_true")

# UQD-paper plots
parser.add_argument("--plot-summary", action="store_true")
parser.add_argument("--plot-pareto", action="store_true")
parser.add_argument("--plot-reproducibility-metrics", action="store_true")
parser.add_argument("--plot-reproducibility-archives", action="store_true")

# Fit-reprod paper plots
parser.add_argument("--plot-benchmark", action="store_true")
parser.add_argument("--plot-benchmark-archives", action="store_true")
parser.add_argument("--plot-robotics", action="store_true")
parser.add_argument("--plot-robotics-archives", action="store_true")
parser.add_argument("--plot-robotics-per-approach-archives", action="store_true")

# Extract paper plots
parser.add_argument("--plot-archive-sampling", action="store_true")
parser.add_argument("--plot-estimation-problem", action="store_true")
parser.add_argument("--plot-extract", action="store_true")
parser.add_argument("--plot-extract-archives", action="store_true")
parser.add_argument("--plot-extract-archive-sampling", action="store_true")
parser.add_argument("--plot-extract-convergence", action="store_true")
parser.add_argument("--plot-extract-time", action="store_true")

# Partial eval paper plots
parser.add_argument("--plot-partial-evaluation", action="store_true")
parser.add_argument("--plot-qdrl-families", action="store_true")
parser.add_argument("--plot-qdrl-families-reprod", action="store_true")
parser.add_argument("--plot-qdrl-families-pe", action="store_true")
parser.add_argument("--plot-qdrl-families-chaining", action="store_true")

# Configuration
parser.add_argument("--xaxis", default="gens", type=str)
parser.add_argument("--use-max-xaxis", action="store_true")
parser.add_argument("--p-values", action="store_true")
parser.add_argument("--plot-in-cell", action="store_true")
parser.add_argument("--additional", action="store_true")
parser.add_argument("--algos", default="", type=str)
parser.add_argument("--exclude-algos", default="", type=str)
parser.add_argument("--exclude-sizes", default="", type=str)
parser.add_argument("--compare-batch-size", action="store_true")
parser.add_argument("--time-pourcent", default=0.95, type=float)
parser.add_argument("--legend-columns", default=1, type=int)
parser.add_argument("--legend-bottom", default=0.1, type=float)
parser.add_argument("--paper-plot", action="store_true")
parser.add_argument("--archives-compare-size", default=16384, type=int)

# Fix for reproducibility computation following thesis writting
parser.add_argument("--reproducibility-fix", action="store_true")

# Visualisation configuration
parser.add_argument("--deterministic", action="store_true")
parser.add_argument("--html", action="store_true")
parser.add_argument("--bd", default="1.0|1.0", type=str)
parser.add_argument("--replications", default=8, type=int)

# Process inputs
args = parser.parse_args()
save_folder = args.results
plot_folder = args.plots
plot_algos = args.algos.rstrip().split("|")
exclude_algos = args.exclude_algos.rstrip().split("|")
exclude_sizes = args.exclude_sizes.rstrip().split("|")
exclude_sizes = [] if exclude_sizes == [""] else [int(size) for size in exclude_sizes]
compare_size = "batch_size" if args.compare_batch_size else "sampling_size"
compare_title = "Batch-size" if args.compare_batch_size else "Sampling-size"
bd = args.bd.rstrip().split("|")
bd = [float(value) for value in bd]
assert os.path.exists(save_folder), "\n!!!ERROR!!! Empty result folder.\n"

if args.xaxis == "gens":
    x_column = "epoch"
    x_name = "Generations"
elif args.xaxis == "eval":
    x_column = "eval"
    x_name = "Evaluations"
elif args.xaxis == "time":
    x_column = "time"
    x_name = "Time (s)"
elif args.xaxis == "timestep":
    x_column = "timestep"
    x_name = "Timesteps"
else:
    assert 0, "!!!ERROR!!! Invalid args.xaxis."

####################
# Other parameters #

if args.paper_plot:
    replace_name = {
        **replace_name,
        " smpl1": "",
        " smpl2": "",
        " smpl32": "",
        " 2": "",
        " 8": "",
    }

# Algo characterisation and order definition
baselines_list = [
    "Vanilla-ME",
    "Vanilla-PGA",
    "Random",
    "ME-Sampling",
    "ME-Sampling-Reprod",
    "ME-LS",
]
categories_list = [
    "ME-Weighted",
    "ME-Delta",
    "MOME-R",
]
baselines_as_list = [
    "Adapt-ME",
    "Deep-Grid",
    "Vanilla-AS",
    "AS-Reprod",
]
categories_as_list = ["AS-Weighted", "AS-Delta", "EME", "EPGA", "AS-MOME", "EMOME"]

order = baselines_list + categories_list + baselines_as_list + categories_as_list

# QD-RL algorithms families
qdrl_families = {
    "PGA": ["Vanilla-ME", "EAS-Weighted-ME", "PGA"],
    "QDPG": ["Vanilla-ME", "EAS-Weighted-ME", "QDPG"],
    "DCRL": ["Vanilla-ME", "EAS-Weighted-ME", "DCRL"],
    # "MEMES": ["Vanilla-ME", "MEMES"],
}

################
# Get results #

# Create results folder if needed
try:
    if not os.path.exists(plot_folder):
        os.mkdir(plot_folder)
    if not os.path.exists(f"{plot_folder}_csv"):
        os.mkdir(f"{plot_folder}_csv")
except Exception:
    print("\n!!!WARNING!!! Cannot create folders for plots.")
    traceback.print_exc()

# Load everything
config_frame, all_convergence, all_finals, all_times, all_var = load_results(
    save_folder=save_folder,
    plot_folder=plot_folder,
    plot_algos=plot_algos,
    exclude_algos=exclude_algos,
    exclude_sizes=exclude_sizes,
    not_name=not_name,
    always_name=always_name,
    replace_name=replace_name,
    time_pourcent=args.time_pourcent,
    compare_size=compare_size,
    order=order,
    use_max_xaxis=args.use_max_xaxis,
    env_max_xaxis=env_max_xaxis,
    x_column=x_column,
)

# Create a color frame
config_frame = sort_data(config_frame, ["algo", compare_size], order)
labels = config_frame["algo"].drop_duplicates().values
colors = sns.color_palette("colorblind", len(labels))  # rocket
color_frame = pd.DataFrame(data={"Label": labels, "Color": colors})


############################
# Plot convergence metrics #

if args.plot_convergence_metrics:

    from analysis.plot_metrics import plot_convergence_metrics

    print("\nPlotting convergence metrics graphs")
    try:
        plot_convergence_metrics(
            plot_folder=plot_folder,
            all_convergence=all_convergence,
            all_times=all_times,
            color_frame=color_frame,
            x_column=x_column,
            x_name=x_name,
            compare_size=compare_size,
            compare_title=compare_title,
            order=order,
            legend_columns=args.legend_columns,
            legend_bottom=args.legend_bottom,
            errors=args.errors,
            additional=args.additional,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot convergence metrics.")
        traceback.print_exc()

    if args.plot_in_cell:
        print("\nPlotting in-cell convergence metrics graphs")
        try:
            plot_convergence_metrics(
                plot_folder=plot_folder,
                all_convergence=all_convergence,
                all_times=all_times,
                color_frame=color_frame,
                x_column=x_column,
                x_name=x_name,
                compare_size=compare_size,
                compare_title=compare_title,
                order=order,
                legend_columns=args.legend_columns,
                legend_bottom=args.legend_bottom,
                prefixe="in_cell_",
                prefixe_title="In-Cell ",
                errors=args.errors,
                additional=args.additional,
            )
        except Exception:
            print("\n!!!WARNING!!! Cannot plot in-cell convergence metrics.")
            traceback.print_exc()

################
# Plot metrics #

if args.plot_metrics:

    from analysis.plot_metrics import plot_metrics

    print("\nPlotting metrics graphs")
    try:
        plot_metrics(
            plot_folder=plot_folder,
            all_finals=all_finals,
            color_frame=color_frame,
            compare_size=compare_size,
            compare_title=compare_title,
            order=order,
            legend_columns=args.legend_columns,
            legend_bottom=args.legend_bottom,
            errors=args.errors,
            additional=args.additional,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot metrics.")
        traceback.print_exc()

    if args.plot_in_cell:
        print("\nPlotting in-cell metrics graphs")
        try:
            plot_metrics(
                plot_folder=plot_folder,
                all_finals=all_finals,
                color_frame=color_frame,
                compare_size=compare_size,
                compare_title=compare_title,
                order=order,
                legend_columns=args.legend_columns,
                legend_bottom=args.legend_bottom,
                prefixe="in_cell_",
                prefixe_title="In-Cell ",
                errors=args.errors,
                additional=args.additional,
            )
        except Exception:
            print("\n!!!WARNING!!! Cannot plot in-cell metrics.")
            traceback.print_exc()

#############
# Plot time #

if args.plot_time:

    from analysis.plot_time import plot_time

    print("\nPlotting time graphs")
    try:
        plot_time(
            plot_folder=plot_folder,
            all_times=all_times,
            color_frame=color_frame,
            compare_size=compare_size,
            compare_title=compare_title,
            order=order,
            legend_columns=args.legend_columns,
            legend_bottom=args.legend_bottom,
            errors=args.errors,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot time.")
        traceback.print_exc()

################
# Plot summary #

if args.plot_summary:

    from analysis.plot_summary import plot_summary

    print("\nPlotting summary graphs")
    try:
        plot_summary(
            plot_folder=plot_folder,
            all_finals=all_finals,
            all_times=all_times,
            all_var=all_var,
            color_frame=color_frame,
            compare_size=compare_size,
            compare_title=compare_title,
            order=order,
            env_order=env_order,
            legend_columns=args.legend_columns,
            legend_bottom=args.legend_bottom,
            errors=args.errors,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot summary.")
        traceback.print_exc()

    if args.p_values:

        print("\nPlotting summary p-values")
        try:
            p_values(
                plot_folder=args.plot_folder,
                compare_size=compare_size,
                stat=f"reeval_qd_score",
                dataframe=all_times,
            )
            p_values(
                plot_folder=args.plot_folder,
                compare_size=compare_size,
                stat=f"loss_reeval_qd_score",
                dataframe=all_finals,
            )
            p_values(
                plot_folder=args.plot_folder,
                compare_size=compare_size,
                stat=f"fit_var_qd_score",
                dataframe=all_var,
            )
            p_values(
                plot_folder=args.plot_folder,
                compare_size=compare_size,
                stat=f"time",
                dataframe=all_times,
            )
        except Exception:
            print("\n!!!WARNING!!! Cannot plot p-values.")
            traceback.print_exc()

    if args.plot_in_cell:
        print("\nPlotting in-cell summary graphs")
        try:
            plot_summary(
                plot_folder=plot_folder,
                all_finals=all_finals,
                all_times=all_times,
                all_var=all_var,
                color_frame=color_frame,
                compare_size=compare_size,
                compare_title=compare_title,
                order=order,
                env_order=env_order,
                legend_columns=args.legend_columns,
                legend_bottom=args.legend_bottom,
                prefixe="in_cell_",
                prefixe_title="In-Cell ",
                errors=args.errors,
            )
        except Exception:
            print("\n!!!WARNING!!! Cannot plot in-cell summary.")
            traceback.print_exc()

        if args.p_values:

            print("\nPlotting in-cell summary p-values")
            try:
                p_values(
                    plot_folder=args.plot_folder,
                    compare_size=compare_size,
                    stat=f"in_cell_reeval_qd_score",
                    dataframe=all_times,
                )
                p_values(
                    plot_folder=args.plot_folder,
                    compare_size=compare_size,
                    stat=f"loss_in_cell_reeval_qd_score",
                    dataframe=all_finals,
                )
                p_values(
                    plot_folder=args.plot_folder,
                    compare_size=compare_size,
                    stat=f"in_cell_fit_var_qd_score",
                    dataframe=all_var,
                )
                p_values(
                    plot_folder=args.plot_folder,
                    compare_size=compare_size,
                    stat=f"in_cell_time",
                    dataframe=all_times,
                )
            except Exception:
                print("\n!!!WARNING!!! Cannot plot in-cell p-values.")
                traceback.print_exc()

###############
# Plot pareto #

if args.plot_pareto:

    from analysis.plot_pareto import plot_pareto

    print("\nPlotting pareto graphs")
    try:
        plot_pareto(
            plot_folder=plot_folder,
            all_times=all_times,
            color_frame=color_frame,
            compare_size=compare_size,
            compare_title=compare_title,
            order=order,
            env_order=env_order,
            legend_columns=args.legend_columns,
            legend_bottom=args.legend_bottom,
            errors=args.errors,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot pareto.")
        traceback.print_exc()

    if args.plot_in_cell:
        print("\nPlotting in-cell pareto graphs")
        try:
            plot_pareto(
                plot_folder=plot_folder,
                all_times=all_times,
                color_frame=color_frame,
                compare_size=compare_size,
                compare_title=compare_title,
                order=order,
                env_order=env_order,
                legend_columns=args.legend_columns,
                legend_bottom=args.legend_bottom,
                prefixe="in_cell_",
                prefixe_title="In-Cell ",
                errors=args.errors,
            )
        except Exception:
            print("\n!!!WARNING!!! Cannot plot in-cell pareto.")
            traceback.print_exc()

######################
# Load min_max_frame #

if (
    args.plot_archives
    or args.plot_all_archives
    or args.plot_robotics_archives
    or args.plot_robotics_per_approach_archives
    or args.plot_benchmark_archives
    or args.plot_extract_archives
    or args.plot_replay_archive
    or args.plot_replay_trajectory
    or args.plot_archive_profiles
):

    try:
        min_max_frame = find_min_max(
            plot_folder=plot_folder,
            config_frame=config_frame,
            errors=args.errors,
            additional=args.additional,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot get min and max, not using normalisation.")
        min_max_frame = None
        traceback.print_exc()

    if args.plot_in_cell:

        try:
            in_cell_min_max_frame = find_min_max(
                plot_folder=plot_folder,
                config_frame=config_frame,
                prefixe="in_cell_",
                errors=args.errors,
                additional=args.additional,
            )
        except Exception:
            print(
                "\n!!!WARNING!!! Cannot get in-cell min and max, not using normalisation."
            )
            in_cell_min_max_frame = None
            traceback.print_exc()

#####################
# Plot all archives #

if args.plot_all_archives:

    from analysis.plot_all_archives import plot_all_archives

    print("\nPlotting all archives")
    try:
        plot_all_archives(
            plot_folder=plot_folder,
            config_frame=config_frame,
            min_max_frame=min_max_frame,
            compare_size=compare_size,
            compare_title=compare_title,
            errors=args.errors,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot all archives.")
        traceback.print_exc()

    if args.plot_in_cell:

        print("\nPlotting all in-cell archives")
        try:
            plot_all_archives(
                plot_folder=plot_folder,
                config_frame=config_frame,
                min_max_frame=in_cell_min_max_frame,
                compare_size=compare_size,
                compare_title=compare_title,
                prefixe="in_cell_",
                prefixe_title="In-Cell ",
                errors=args.errors,
            )
        except Exception:
            print("\n!!!WARNING!!! Cannot plot in-cell all archives.")
            traceback.print_exc()

#################
# Plot archives #

if args.plot_archives:

    from analysis.plot_archives import plot_archives

    print("\nPlotting archives")
    try:
        plot_archives(
            plot_folder=plot_folder,
            single_compare_size=args.archives_compare_size,
            env_order=env_order,
            config_frame=config_frame,
            min_max_frame=min_max_frame,
            compare_size=compare_size,
            compare_title=compare_title,
            errors=args.errors,
            additional=args.additional,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot archives.")
        traceback.print_exc()

    if args.plot_in_cell:

        print("\nPlotting in-cell archives")
        try:
            plot_archives(
                plot_folder=plot_folder,
                single_compare_size=args.archives_compare_size,
                env_order=env_order,
                config_frame=config_frame,
                min_max_frame=in_cell_min_max_frame,
                compare_size=compare_size,
                compare_title=compare_title,
                prefixe="in_cell_",
                prefixe_title="In-Cell ",
                errors=args.errors,
                additional=args.additional,
            )
        except Exception:
            print("\n!!!WARNING!!! Cannot plot in-cell archives.")
            traceback.print_exc()

#########################
# Plot archive profiles #

if args.plot_archive_profiles:

    from analysis.plot_archive_profiles import plot_archive_profiles

    print("\nPlotting archive profiles")
    try:
        plot_archive_profiles(
            plot_folder=plot_folder,
            config_frame=config_frame,
            min_max_frame=min_max_frame,
            color_frame=color_frame,
            compare_size=compare_size,
            compare_title=compare_title,
            order=order,
            legend_columns=args.legend_columns,
            legend_bottom=args.legend_bottom,
            errors=args.errors,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot archives.")
        traceback.print_exc()

    if args.plot_in_cell:

        print("\nPlotting archive in-cell profiles")
        try:
            plot_archive_profiles(
                plot_folder=plot_folder,
                config_frame=config_frame,
                min_max_frame=in_cell_min_max_frame,
                color_frame=color_frame,
                compare_size=compare_size,
                compare_title=compare_title,
                order=order,
                legend_columns=args.legend_columns,
                legend_bottom=args.legend_bottom,
                prefixe="in_cell_",
                prefixe_title="In-Cell ",
                errors=args.errors,
            )
        except Exception:
            print("\n!!!WARNING!!! Cannot plot in-cell archives.")
            traceback.print_exc()

##########################
# Load reproducibilities #

if (
    args.plot_reproducibility_metrics
    or args.plot_reproducibility_archives
    or args.plot_robotics
    or args.plot_robotics_archives
    or args.plot_benchmark_archives
    or args.plot_qdrl_families_reprod
):

    print("\nLoading reproducibilities.")
    try:
        (all_reproducibilities_data, reprod_min_max_frame,) = load_reproducibilities(
            plot_folder=plot_folder,
            config_frame=config_frame,
            compare_size=compare_size,
            order=order,
            reproducibility_fix=args.reproducibility_fix,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot load reproducibilities.")
        traceback.print_exc()

    if args.plot_in_cell:

        print("\nLoading in-cell reproducibilities.")
        try:
            (
                in_cell_all_reproducibilities_data,
                in_cell_reprod_min_max_frame,
            ) = load_reproducibilities(
                plot_folder=plot_folder,
                config_frame=config_frame,
                compare_size=compare_size,
                order=order,
                reproducibility_fix=args.reproducibility_fix,
                prefixe="in_cell_",
            )
        except Exception:
            print("\n!!!WARNING!!! Cannot load in_cell_reproducibilities.")
            traceback.print_exc()


################################
# Plot reproducibility metrics #

if args.plot_reproducibility_metrics:

    from analysis.plot_reproducibility_metrics import plot_reproducibility_metrics

    print("\nPlotting reproducibility convergence metrics graphs")
    try:
        plot_reproducibility_metrics(
            plot_folder=plot_folder,
            all_reproducibilities_data=all_reproducibilities_data,
            color_frame=color_frame,
            compare_size=compare_size,
            compare_title=compare_title,
            order=order,
            legend_columns=args.legend_columns,
            legend_bottom=args.legend_bottom,
            errors=args.errors,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot reproducibility metrics.")
        traceback.print_exc()

    if args.plot_in_cell:
        print("\nPlotting in-cell reproducibility convergence metrics graphs")
        try:
            plot_reproducibility_metrics(
                plot_folder=plot_folder,
                all_reproducibilities_data=in_cell_all_reproducibilities_data,
                color_frame=color_frame,
                compare_size=compare_size,
                compare_title=compare_title,
                order=order,
                legend_columns=args.legend_columns,
                legend_bottom=args.legend_bottom,
                prefixe="in_cell_",
                prefixe_title="In-Cell ",
                errors=args.errors,
            )
        except Exception:
            print("\n!!!WARNING!!! Cannot plot in-cell reproducibility metrics.")
            traceback.print_exc()


##################################
# Plot reproducibility archives #

if args.plot_reproducibility_archives:

    from analysis.plot_reproducibility_archives import plot_reproducibility_archives

    print("\nPlotting reproducibility archives")
    try:
        plot_reproducibility_archives(
            plot_folder=plot_folder,
            single_compare_size=args.archives_compare_size,
            env_order=env_order,
            config_frame=config_frame,
            reprod_min_max_frame=reprod_min_max_frame,
            compare_size=compare_size,
            compare_title=compare_title,
            errors=args.errors,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot archives.")
        traceback.print_exc()

    if args.plot_in_cell:

        print("\nPlotting in-cell reproducibility archives")
        try:
            plot_reproducibility_archives(
                plot_folder=plot_folder,
                single_compare_size=args.archives_compare_size,
                env_order=env_order,
                config_frame=config_frame,
                reprod_min_max_frame=reprod_min_max_frame,
                compare_size=compare_size,
                compare_title=compare_title,
                prefixe="in_cell_",
                prefixe_title="In-Cell ",
                errors=args.errors,
            )
        except Exception:
            print("\n!!!WARNING!!! Cannot plot in-cell archives.")
            traceback.print_exc()

#########################
# Plot robotics metrics #

if args.plot_robotics:

    from analysis.plot_robotics import plot_robotics

    print("\nPlotting robotics convergence metrics graphs")
    try:
        plot_robotics(
            plot_folder=plot_folder,
            all_finals=all_finals,
            all_reprods=all_reproducibilities_data,
            baselines_list=baselines_list,
            categories_list=categories_list,
            baselines_as_list=baselines_as_list,
            categories_as_list=categories_as_list,
            color_frame=color_frame,
            compare_size=compare_size,
            compare_title=compare_title,
            legend_columns=args.legend_columns,
            legend_bottom=args.legend_bottom,
            p_values=args.p_values,
            errors=args.errors,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot robotics metrics.")
        traceback.print_exc()

    if args.plot_in_cell:
        print("\nPlotting in-cell robotics convergence metrics graphs")
        try:
            plot_robotics(
                plot_folder=plot_folder,
                all_finals=all_finals,
                all_reprods=in_cell_all_reproducibilities_data,
                baselines_list=baselines_list,
                categories_list=categories_list,
                baselines_as_list=baselines_as_list,
                categories_as_list=categories_as_list,
                color_frame=color_frame,
                compare_size=compare_size,
                compare_title=compare_title,
                legend_columns=args.legend_columns,
                legend_bottom=args.legend_bottom,
                p_values=args.p_values,
                prefixe="in_cell_",
                prefixe_title="In-Cell ",
                errors=args.errors,
            )
        except Exception:
            print("\n!!!WARNING!!! Cannot plot in-cell robotics metrics.")
            traceback.print_exc()

##########################
# Plot robotics archives #

if args.plot_robotics_archives:

    from analysis.plot_robotics import plot_robotics_archives

    print("\nPlotting robotics archives")
    try:
        plot_robotics_archives(
            plot_folder=plot_folder,
            single_compare_size=args.archives_compare_size,
            env_order=env_order,
            config_frame=config_frame,
            baselines_list=baselines_list,
            categories_list=categories_list,
            baselines_as_list=baselines_as_list,
            categories_as_list=categories_as_list,
            min_max_frame=min_max_frame,
            compare_size=compare_size,
            compare_title=compare_title,
            errors=args.errors,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot robotics archives.")
        traceback.print_exc()

    if args.plot_in_cell:

        print("\nPlotting in-cell robotics archives")
        try:
            plot_robotics_archives(
                plot_folder=plot_folder,
                single_compare_size=args.archives_compare_size,
                env_order=env_order,
                config_frame=config_frame,
                baselines_list=baselines_list,
                categories_list=categories_list,
                baselines_as_list=baselines_as_list,
                categories_as_list=categories_as_list,
                min_max_frame=in_cell_min_max_frame,
                compare_size=compare_size,
                compare_title=compare_title,
                prefixe="in_cell_",
                prefixe_title="In-Cell ",
                errors=args.errors,
            )
        except Exception:
            print("\n!!!WARNING!!! Cannot plot in-cell robotics archives.")
            traceback.print_exc()

if args.plot_robotics_per_approach_archives:

    from analysis.plot_robotics import plot_robotics_per_approach_archives

    print("\nPlotting robotics per-approach archives")
    try:
        plot_robotics_per_approach_archives(
            plot_folder=plot_folder,
            single_compare_size=args.archives_compare_size,
            env_order=env_order,
            config_frame=config_frame,
            baselines_list=baselines_list,
            categories_list=categories_list,
            baselines_as_list=baselines_as_list,
            categories_as_list=categories_as_list,
            min_max_frame=min_max_frame,
            compare_size=compare_size,
            compare_title=compare_title,
            errors=args.errors,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot robotics archives.")
        traceback.print_exc()

    if args.plot_in_cell:

        print("\nPlotting in-cell robotics per-appraoch archives")
        try:
            plot_robotics_per_approach_archives(
                plot_folder=plot_folder,
                single_compare_size=args.archives_compare_size,
                env_order=env_order,
                config_frame=config_frame,
                baselines_list=baselines_list,
                categories_list=categories_list,
                baselines_as_list=baselines_as_list,
                categories_as_list=categories_as_list,
                min_max_frame=in_cell_min_max_frame,
                compare_size=compare_size,
                compare_title=compare_title,
                prefixe="in_cell_",
                prefixe_title="In-Cell ",
                errors=args.errors,
            )
        except Exception:
            print("\n!!!WARNING!!! Cannot plot in-cell robotics per-appraoch archives.")
            traceback.print_exc()


##################
# Plot benchmark #

if args.plot_benchmark:

    from analysis.plot_benchmark import plot_benchmark

    print("\nPlotting benchmark graphs")
    try:
        plot_benchmark(
            plot_folder=plot_folder,
            all_finals=all_finals,
            baselines_list=baselines_list,
            categories_list=categories_list,
            baselines_as_list=baselines_as_list,
            categories_as_list=categories_as_list,
            color_frame=color_frame,
            compare_size=compare_size,
            compare_title=compare_title,
            legend_columns=args.legend_columns,
            legend_bottom=args.legend_bottom,
            p_values=args.p_values,
            errors=args.errors,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot benchmark.")
        traceback.print_exc()

    if args.plot_in_cell:
        print("\nPlotting in-cell benchmark graphs")
        try:
            plot_benchmark(
                plot_folder=plot_folder,
                all_finals=all_finals,
                baselines_list=baselines_list,
                categories_list=categories_list,
                baselines_as_list=baselines_as_list,
                categories_as_list=categories_as_list,
                color_frame=color_frame,
                compare_size=compare_size,
                compare_title=compare_title,
                legend_columns=args.legend_columns,
                legend_bottom=args.legend_bottom,
                p_values=args.p_values,
                prefixe="in_cell_",
                prefixe_title="In-Cell ",
                errors=args.errors,
            )
        except Exception:
            print("\n!!!WARNING!!! Cannot plot in-cell benchmark.")
            traceback.print_exc()

###########################
# Plot benchmark archives #

if args.plot_benchmark_archives:

    from analysis.plot_benchmark import plot_benchmark_archives

    print("\nPlotting benchmark archives")
    try:
        plot_benchmark_archives(
            plot_folder=plot_folder,
            min_max_frame=min_max_frame,
            single_compare_size=args.archives_compare_size,
            env_order=env_order,
            config_frame=config_frame,
            baselines_list=baselines_list,
            categories_list=categories_list,
            baselines_as_list=baselines_as_list,
            categories_as_list=categories_as_list,
            compare_size=compare_size,
            compare_title=compare_title,
            errors=args.errors,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot benchmark archives.")
        traceback.print_exc()

###########################
# Plot estimation problem #

if args.plot_estimation_problem:

    from analysis.plot_estimation_problem import plot_estimation_problem

    print("\nPlotting estimation_problem graphs")
    try:
        plot_estimation_problem(
            plot_folder=plot_folder,
            all_finals=all_finals,
            baselines_list=baselines_list,
            categories_list=categories_list,
            baselines_as_list=baselines_as_list,
            categories_as_list=categories_as_list,
            color_frame=color_frame,
            compare_size=compare_size,
            compare_title=compare_title,
            legend_columns=args.legend_columns,
            legend_bottom=args.legend_bottom,
            p_values=args.p_values,
            errors=args.errors,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot estimation_problem.")
        traceback.print_exc()

    if args.plot_in_cell:
        print("\nPlotting in-cell estimation_problem graphs")
        try:
            plot_estimation_problem(
                plot_folder=plot_folder,
                all_finals=all_finals,
                baselines_list=baselines_list,
                categories_list=categories_list,
                baselines_as_list=baselines_as_list,
                categories_as_list=categories_as_list,
                color_frame=color_frame,
                compare_size=compare_size,
                compare_title=compare_title,
                legend_columns=args.legend_columns,
                legend_bottom=args.legend_bottom,
                p_values=args.p_values,
                prefixe="in_cell_",
                prefixe_title="In-Cell ",
                errors=args.errors,
            )
        except Exception:
            print("\n!!!WARNING!!! Cannot plot in-cell estimation_problem.")
            traceback.print_exc()

########################
# Plot extract problem #

if args.plot_extract:

    from analysis.plot_extract import plot_extract

    print("\nPlotting extract_problem graphs")
    try:
        plot_extract(
            plot_folder=plot_folder,
            all_finals=all_finals,
            baselines_list=baselines_list,
            categories_list=categories_list,
            baselines_as_list=baselines_as_list,
            categories_as_list=categories_as_list,
            color_frame=color_frame,
            compare_size=compare_size,
            compare_title=compare_title,
            legend_columns=args.legend_columns,
            legend_bottom=args.legend_bottom,
            p_values=args.p_values,
            errors=args.errors,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot extract_problem.")
        traceback.print_exc()

    if args.plot_in_cell:
        print("\nPlotting in-cell extract_problem graphs")
        try:
            plot_extract(
                plot_folder=plot_folder,
                all_finals=all_finals,
                baselines_list=baselines_list,
                categories_list=categories_list,
                baselines_as_list=baselines_as_list,
                categories_as_list=categories_as_list,
                color_frame=color_frame,
                compare_size=compare_size,
                compare_title=compare_title,
                legend_columns=args.legend_columns,
                legend_bottom=args.legend_bottom,
                p_values=args.p_values,
                prefixe="in_cell_",
                prefixe_title="In-Cell ",
                errors=args.errors,
            )
        except Exception:
            print("\n!!!WARNING!!! Cannot plot in-cell extract_problem.")
            traceback.print_exc()

############################
# Plot extract convergence #

if args.plot_extract_convergence:

    from analysis.plot_extract import plot_extract_convergence

    print("\nPlotting extract_problem convergence graphs")
    try:
        plot_extract_convergence(
            plot_folder=plot_folder,
            all_convergence=all_convergence,
            baselines_list=baselines_list,
            categories_list=categories_list,
            baselines_as_list=baselines_as_list,
            categories_as_list=categories_as_list,
            x_column=x_column,
            x_name=x_name,
            color_frame=color_frame,
            compare_size=compare_size,
            compare_title=compare_title,
            legend_columns=args.legend_columns,
            legend_bottom=args.legend_bottom,
            errors=args.errors,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot extract_problem convergence.")
        traceback.print_exc()

    if args.plot_in_cell:
        print("\nPlotting in-cell extract_problem convergence graphs")
        try:
            plot_extract_convergence(
                plot_folder=plot_folder,
                all_convergence=all_convergence,
                baselines_list=baselines_list,
                categories_list=categories_list,
                baselines_as_list=baselines_as_list,
                categories_as_list=categories_as_list,
                x_column=x_column,
                x_name=x_name,
                color_frame=color_frame,
                compare_size=compare_size,
                compare_title=compare_title,
                legend_columns=args.legend_columns,
                legend_bottom=args.legend_bottom,
                prefixe="in_cell_",
                prefixe_title="In-Cell ",
                errors=args.errors,
            )
        except Exception:
            print("\n!!!WARNING!!! Cannot plot in-cell extract_problem convergence.")
            traceback.print_exc()

#####################
# Plot extract time #

if args.plot_extract_time:

    from analysis.plot_extract import plot_extract_time

    print("\nPlotting extract_problem convergence graphs")
    try:
        plot_extract_time(
            plot_folder=plot_folder,
            all_times=all_times,
            baselines_list=baselines_list,
            categories_list=categories_list,
            baselines_as_list=baselines_as_list,
            categories_as_list=categories_as_list,
            color_frame=color_frame,
            compare_size=compare_size,
            compare_title=compare_title,
            legend_columns=args.legend_columns,
            legend_bottom=args.legend_bottom,
            p_values=args.p_values,
            errors=args.errors,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot extract_problem time.")
        traceback.print_exc()

    if args.plot_in_cell:
        print("\nPlotting in-cell extract_problem time graphs")
        try:
            plot_extract_time(
                plot_folder=plot_folder,
                all_times=all_times,
                baselines_list=baselines_list,
                categories_list=categories_list,
                baselines_as_list=baselines_as_list,
                categories_as_list=categories_as_list,
                color_frame=color_frame,
                compare_size=compare_size,
                compare_title=compare_title,
                legend_columns=args.legend_columns,
                legend_bottom=args.legend_bottom,
                p_values=args.p_values,
                prefixe="in_cell_",
                prefixe_title="In-Cell ",
                errors=args.errors,
            )
        except Exception:
            print("\n!!!WARNING!!! Cannot plot in-cell extract_problem time.")
            traceback.print_exc()


#########################
# Plot extract archives #

if args.plot_extract_archives:

    from analysis.plot_extract import plot_extract_archives

    print("\nPlotting extract archives")
    try:
        plot_extract_archives(
            plot_folder=plot_folder,
            single_compare_size=args.archives_compare_size,
            config_frame=config_frame,
            baselines_list=baselines_list,
            categories_list=categories_list,
            baselines_as_list=baselines_as_list,
            categories_as_list=categories_as_list,
            min_max_frame=min_max_frame,
            compare_size=compare_size,
            compare_title=compare_title,
            errors=args.errors,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot extract archives.")
        traceback.print_exc()

    if args.plot_in_cell:

        print("\nPlotting in-cell extract archives")
        try:
            plot_extract_archives(
                plot_folder=plot_folder,
                single_compare_size=args.archives_compare_size,
                config_frame=config_frame,
                baselines_list=baselines_list,
                categories_list=categories_list,
                baselines_as_list=baselines_as_list,
                categories_as_list=categories_as_list,
                min_max_frame=in_cell_min_max_frame,
                compare_size=compare_size,
                compare_title=compare_title,
                prefixe="in_cell_",
                prefixe_title="In-Cell ",
                errors=args.errors,
            )
        except Exception:
            print("\n!!!WARNING!!! Cannot plot in-cell extract archives.")
            traceback.print_exc()

#################################
# Plot extract archive sampling #

if args.plot_extract_archive_sampling:

    from analysis.plot_extract import plot_extract_archive_sampling

    print("\nPlotting archive_sampling extract metrics graphs")
    try:
        plot_extract_archive_sampling(
            plot_folder=plot_folder,
            config_frame=config_frame,
            x_column=x_column,
            x_name=x_name,
            compare_size=compare_size,
            compare_title=compare_title,
            order=order,
            color_frame=color_frame,
            legend_columns=args.legend_columns,
            legend_bottom=args.legend_bottom,
            use_max_xaxis=args.use_max_xaxis,
            env_max_xaxis=env_max_xaxis,
            p_values=args.p_values,
            errors=args.errors,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot archive_sampling extract metrics.")
        traceback.print_exc()


###################################
# Plot partial_evaluation metrics #

if args.plot_partial_evaluation:

    from analysis.plot_partial_evaluation import plot_partial_evaluation

    print("\nPlotting partial_evaluation convergence metrics graphs")
    try:
        plot_partial_evaluation(
            plot_folder=plot_folder,
            config_frame=config_frame,
            all_convergence=all_convergence,
            x_column=x_column,
            x_name=x_name,
            compare_size=compare_size,
            compare_title=compare_title,
            order=order,
            env_order=env_order,
            color_frame=color_frame,
            legend_columns=args.legend_columns,
            legend_bottom=args.legend_bottom,
            use_max_xaxis=args.use_max_xaxis,
            env_max_xaxis=env_max_xaxis,
            errors=args.errors,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot partial_evaluation metrics.")
        traceback.print_exc()

###################################
# Plot archive_sampling metrics #

if args.plot_archive_sampling:

    from analysis.plot_archive_sampling import plot_archive_sampling

    print("\nPlotting archive_sampling convergence metrics graphs")
    try:
        plot_archive_sampling(
            plot_folder=plot_folder,
            config_frame=config_frame,
            x_column=x_column,
            x_name=x_name,
            compare_size=compare_size,
            compare_title=compare_title,
            order=order,
            color_frame=color_frame,
            legend_columns=args.legend_columns,
            legend_bottom=args.legend_bottom,
            use_max_xaxis=args.use_max_xaxis,
            env_max_xaxis=env_max_xaxis,
            errors=args.errors,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot archive_sampling metrics.")
        traceback.print_exc()

##############################
# Plot qdrl_families metrics #

if args.plot_qdrl_families:

    from analysis.plot_qdrl_families import (
        plot_qdrl_families,
        plot_qdrl_families_5pourcent,
        plot_qdrl_families_area,
        plot_qdrl_families_convergence,
        plot_qdrl_families_convergence_split,
        plot_qdrl_families_sample_complexity,
    )

    print("\nPlotting qdrl_families metrics")
    try:
        plot_qdrl_families_convergence(
            plot_folder=plot_folder,
            all_convergence=all_convergence,
            color_frame=color_frame,
            x_column=x_column,
            x_name=x_name,
            compare_size=compare_size,
            compare_title=compare_title,
            order=order,
            env_order=env_order,
            qdrl_families=qdrl_families,
            legend_columns=args.legend_columns,
            legend_bottom=args.legend_bottom,
            errors=args.errors,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot qdrl_families convergence metrics.")
        traceback.print_exc()

    try:
        plot_qdrl_families_convergence_split(
            plot_folder=plot_folder,
            all_convergence=all_convergence,
            color_frame=color_frame,
            x_column=x_column,
            x_name=x_name,
            compare_size=compare_size,
            compare_title=compare_title,
            order=order,
            env_order=env_order,
            qdrl_families=qdrl_families,
            legend_columns=args.legend_columns,
            legend_bottom=args.legend_bottom,
            errors=args.errors,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot qdrl_families convergence metrics.")
        traceback.print_exc()

    try:
        plot_qdrl_families(
            plot_folder=plot_folder,
            all_finals=all_finals,
            color_frame=color_frame,
            compare_size=compare_size,
            compare_title=compare_title,
            env_order=env_order,
            qdrl_families=qdrl_families,
            legend_columns=args.legend_columns,
            legend_bottom=args.legend_bottom,
            p_values=args.p_values,
            errors=args.errors,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot qdrl_families metrics.")
        traceback.print_exc()

    try:
        plot_qdrl_families_area(
            plot_folder=plot_folder,
            all_convergence=all_convergence,
            color_frame=color_frame,
            compare_size=compare_size,
            compare_title=compare_title,
            env_order=env_order,
            qdrl_families=qdrl_families,
            legend_columns=args.legend_columns,
            legend_bottom=args.legend_bottom,
            p_values=args.p_values,
            errors=args.errors,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot qdrl_families area metrics.")
        traceback.print_exc()

    try:
        plot_qdrl_families_5pourcent(
            plot_folder=plot_folder,
            all_times=all_times,
            color_frame=color_frame,
            compare_size=compare_size,
            compare_title=compare_title,
            env_order=env_order,
            qdrl_families=qdrl_families,
            legend_columns=args.legend_columns,
            legend_bottom=args.legend_bottom,
            p_values=args.p_values,
            errors=args.errors,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot qdrl_families 5pourcent metrics.")
        traceback.print_exc()

    try:
        plot_qdrl_families_sample_complexity(
            plot_folder=plot_folder,
            all_convergence=all_convergence,
            all_finals=all_finals,
            color_frame=color_frame,
            compare_size=compare_size,
            compare_title=compare_title,
            order=order,
            env_order=env_order,
            qdrl_families=qdrl_families,
            legend_columns=args.legend_columns,
            legend_bottom=args.legend_bottom,
            p_values=args.p_values,
            errors=args.errors,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot qdrl_families 5pourcent metrics.")
        traceback.print_exc()


if args.plot_qdrl_families_reprod:

    from analysis.plot_qdrl_families import plot_qdrl_families_reprod

    print("\nPlotting qdrl_families reprod metrics")
    try:
        plot_qdrl_families_reprod(
            plot_folder=plot_folder,
            all_reprods=all_reproducibilities_data,
            color_frame=color_frame,
            compare_size=compare_size,
            compare_title=compare_title,
            env_order=env_order,
            qdrl_families=qdrl_families,
            legend_columns=args.legend_columns,
            legend_bottom=args.legend_bottom,
            p_values=args.p_values,
            errors=args.errors,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot qdrl_families reprod metrics.")
        traceback.print_exc()

if args.plot_qdrl_families_pe:

    from analysis.plot_qdrl_families import plot_qdrl_families_partial_evaluation

    print("\nPlotting partial_evaluation extract metrics graphs")
    try:
        plot_qdrl_families_partial_evaluation(
            plot_folder=plot_folder,
            config_frame=config_frame,
            x_column=x_column,
            x_name=x_name,
            compare_size=compare_size,
            compare_title=compare_title,
            order=order,
            env_order=env_order,
            color_frame=color_frame,
            legend_columns=args.legend_columns,
            legend_bottom=args.legend_bottom,
            use_max_xaxis=args.use_max_xaxis,
            env_max_xaxis=env_max_xaxis,
            p_values=args.p_values,
            errors=args.errors,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot partial_evaluation extract metrics.")
        traceback.print_exc()

#######################################
# Plot qdrl_families chaining metrics #

if args.plot_qdrl_families_chaining:

    from analysis.load_chaining import load_chaining

    print("\nLoading qdrl_families chaining.")
    try:
        config_frame, all_chaining_data, all_data = load_chaining(
            plot_folder=plot_folder,
            compare_size=compare_size,
            always_name=always_name,
            not_name=not_name,
            replace_name=replace_name,
            order=order,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot load qdrl_families chaining.")
        traceback.print_exc()

    from analysis.plot_qdrl_families import plot_qdrl_families_chaining

    print("\nPlotting qdrl_families chaining metrics")
    try:
        plot_qdrl_families_chaining(
            plot_folder=plot_folder,
            all_chaining_data=all_chaining_data,
            color_frame=color_frame,
            compare_size=compare_size,
            compare_title=compare_title,
            env_order=env_order,
            qdrl_families=qdrl_families,
            legend_columns=args.legend_columns,
            legend_bottom=args.legend_bottom,
            p_values=args.p_values,
            errors=args.errors,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot qdrl_families convergence metrics.")
        traceback.print_exc()


#####################
# Plot replay final #

if args.plot_replay_archive:

    from analysis.plot_replay_archive import plot_replay_archive

    print("\nPlotting replay final")
    try:
        plot_replay_archive(
            plot_folder=plot_folder,
            config_frame=config_frame,
            min_max_frame=min_max_frame,
            compare_size=compare_size,
            deterministic=args.deterministic,
            replications=args.replications,
            bd=bd,
            paper_plot=args.paper_plot,
            errors=args.errors,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot replay final.")
        traceback.print_exc()

##########################
# Plot replay trajectory #

if args.plot_replay_trajectory:

    from analysis.plot_replay_trajectory import plot_replay_trajectory

    print("\nPlotting replay trajectory")
    try:
        plot_replay_trajectory(
            plot_folder=plot_folder,
            config_frame=config_frame,
            compare_size=compare_size,
            min_max_frame=min_max_frame,
            deterministic=args.deterministic,
            replications=args.replications,
            bd=bd,
            paper_plot=args.paper_plot,
            save_html=args.html,
            errors=args.errors,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot replay trajectory.")
        traceback.print_exc()

######################
# Plot replay reward #

if args.plot_replay_reward:

    from analysis.plot_replay_reward import plot_replay_reward

    print("\nPlotting replay reward")
    try:
        plot_replay_reward(
            plot_folder=plot_folder,
            config_frame=config_frame,
            compare_size=compare_size,
            deterministic=args.deterministic,
            replications=args.replications,
            bd=bd,
            paper_plot=args.paper_plot,
            save_html=args.html,
            errors=args.errors,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot replay reward.")
        traceback.print_exc()

############################
# Plot replay gelman rubin #

if args.plot_replay_gr:

    from analysis.plot_replay_gelman_rubin import plot_replay_gelman_rubin

    print("\nPlotting replay gelman rubin")
    try:
        plot_replay_gelman_rubin(
            plot_folder=plot_folder,
            config_frame=config_frame,
            compare_size=compare_size,
            deterministic=args.deterministic,
            replications=args.replications,
            bd=bd,
            paper_plot=args.paper_plot,
            save_html=args.html,
            errors=args.errors,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot replay gelman rubin.")
        traceback.print_exc()
