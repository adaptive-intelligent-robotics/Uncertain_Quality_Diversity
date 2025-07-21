import argparse
import os
import traceback

import matplotlib as mpl
import pandas as pd
import seaborn as sns

from analysis.load_archives import find_min_max
from analysis.load_p_values import p_values
from analysis.load_reproducibilities import load_reproducibilities
from analysis.load_results import load_results, sort_data
from analysis.plot_algo_properties import plot_algo_properties
from analysis.plot_all_archives import plot_all_archives
from analysis.plot_archive_profiles import plot_archive_profiles
from analysis.plot_archives import plot_archives
from analysis.plot_benchmark import plot_benchmark, plot_benchmark_archives
from analysis.plot_metrics import plot_convergence_metrics, plot_metrics
from analysis.plot_pareto import plot_pareto
from analysis.plot_reproducibility_archives import plot_reproducibility_archives
from analysis.plot_reproducibility_metrics import plot_reproducibility_metrics
from analysis.plot_robotics import plot_robotics, plot_robotics_archives
from analysis.plot_summary import plot_summary
from analysis.plot_visualisations import plot_visualisations

#########
# Input #

parser = argparse.ArgumentParser()

# Folder
parser.add_argument("--results", default="results", type=str)
parser.add_argument("--plots", default="plots", type=str)

# Choose what to plot
parser.add_argument("--plot-metrics", action="store_true")
parser.add_argument("--plot-convergence-metrics", action="store_true")
parser.add_argument("--plot-visualisations", action="store_true")
parser.add_argument("--plot-archives", action="store_true")
parser.add_argument("--plot-all-archives", action="store_true")
parser.add_argument("--plot-archive-profiles", action="store_true")
parser.add_argument("--plot-algo-properties", action="store_true")
parser.add_argument("--plot-reproducibility-metrics", action="store_true")
parser.add_argument("--plot-reproducibility-archives", action="store_true")

# UQD-paper plots
parser.add_argument("--plot-summary", action="store_true")
parser.add_argument("--plot-pareto", action="store_true")

# Fit-reprod paper plots
parser.add_argument("--plot-benchmark", action="store_true")
parser.add_argument("--plot-benchmark-archives", action="store_true")
parser.add_argument("--plot-robotics", action="store_true")
parser.add_argument("--plot-robotics-archives", action="store_true")

# Configuration
parser.add_argument("--p-values", action="store_true")
parser.add_argument("--plot-in-cell", action="store_true")
parser.add_argument("--algos", default="", type=str)
parser.add_argument("--exclude-algos", default="", type=str)
parser.add_argument("--exclude-sizes", default="", type=str)
parser.add_argument("--compare-batch-size", action="store_true")
parser.add_argument("--time-pourcent", default=1.0, type=float)
parser.add_argument("--pourcent-value", action="store_true")
parser.add_argument("--legend-columns", default=2, type=int)
parser.add_argument("--legend-bottom", default=0.2, type=float)
parser.add_argument("--paper-plot", action="store_true")
parser.add_argument("--archives-compare-size", default=16384, type=int)

# Visualisation configuration
parser.add_argument("--save-html", action="store_true")
parser.add_argument("--deterministic", action="store_true")
parser.add_argument("--best-indiv", action="store_true")
parser.add_argument("--indiv", default=0, type=int)
parser.add_argument("--replications", default=256, type=int)

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
assert os.path.exists(save_folder), "\n!!!ERROR!!! Empty result folder.\n"

####################
# Other parameters #

not_name = [
    "folder",
    "run",
    "seed",
    "env",
    "num_iterations",
    "batch_size",
    "sampling_size",
    "sampling_use",
    "init_time",
    "run_time",
    "metrics_file",
    "in_cell_metrics_file",
    "repertoire_folder",
    "projected_repertoire_folder",
    "target_repertoire_folder",
    "reeval_repertoire_folder",
    "average_repertoire_folder",
    "fit_reeval_repertoire_folder",
    "fit_average_repertoire_folder",
    "desc_reeval_repertoire_folder",
    "desc_average_repertoire_folder",
    "fit_var_repertoire_folder",
    "desc_var_repertoire_folder",
    "additional_folder",
    "reeval_additional_folder",
    "in_cell_reeval_repertoire_folder",
    "in_cell_fit_reeval_repertoire_folder",
    "in_cell_desc_reeval_repertoire_folder",
    "in_cell_fit_var_repertoire_folder",
    "in_cell_desc_var_repertoire_folder",
    "reeval_fit_var_repertoire_folder",
    "reeval_desc_var_repertoire_folder",
    "in_cell_reeval_fit_var_repertoire_folder",
    "in_cell_reeval_desc_var_repertoire_folder",
    "in_cell_reeval_additional_repertoire_folder",
    "in_cell_reeval_additional_repertoire",
    "min_bd",
    "max_bd",
    "depth",
    "num_samples",
    "episode_length",
    "params_variance",
    "desc_variance",
    "fit_variance",
    "params_std",
    "desc_std",
    "fit_std",
    "delta_fitness",
    "delta_reproducibility",
]

replace_name = {
    "Extended-Adaptive-Sampling-median": "Parallel-Adaptive-Sampling",
    "MAP-Sampling": "Vanilla-Archive-Sampling",
    "Estimator-Archive-Sampling": "Vanilla-Archive-Sampling",
    "MAP-Elites": "ME",
    "ME-sampling-": "ME-Sampling-sampling-",
    "ME-reproducibility-sampling-": "ME-Sampling-Reproducibility-sampling-",
    "Low-Spread": "LS",
    "Archive-Sampling": "AS",
    "Parallel-Adaptive-Sampling": "PAS",
    "DominanceReprod": "Dominance (ours)",
    "WeightedReprod": "Weighted (ours)",
    "MOME-Reprod-biased": "MOME-X (ours)",
    "-depth-": " ",
    "-sampling-": " smpl",
    "-archive-out-sampling-": "-out-smpl",
    "Mutation": "Mut",
    "-decay": "-d",
    "MAD-fitreprodextract-MAD-fitreprodextract": "MAD-fitreprodextract",
    "fitreprodextract": "reprod",
    "fitextract": "perf",
    "-f0.01-r0.01-0.0001": "",
    "-f0.01-r0.01": "",
    "-f0.02-r0.02-0.0001": "",
    "-f0.02-r0.02": "",
    "-f0.2-r0.02-0.0001": "",
    "-f0.2-r0.02": "",
    "-f0.05-r0.05-0.0001": "",
    "-f0.05-r0.05": "",
    "-f0.2-r0.0-0.0001": "",
    "-f0.2-r0.0": "",
    "-f260.0-r0.04-0.0001": "",
    "-f260.0-r0.04": "",
    "-f140.0-r0.14-0.0001": "",
    "-f140.0-r0.14": "",
    "-f220.0-r6.0-0.0001": "",
    "-f220.0-r6.0": "",
    "-proj": "",
    "  ": " ",
}
if args.paper_plot:
    replace_name = {
        **replace_name,
        " smpl1": "",
        " smpl2": "",
        " smpl32": "",
        " 2": "",
    }

baselines_list = [
    "Vanilla-ME",
    "Random",
    "ME-Sampling",
    "ME-LS",
    "ME-Sampling-Reproducibility",
]
categories_list = [
    "MOME-X",
    "ME-Weighted",
    "ME-Dominance",
    "MOME-Reprod-biased",
    # "Deep-Grid ",
    # "Deep-Grid-sampling",
    # "Deep-Bias",
    # "Deep-Target",
    # "Archive-Sampling",
    # "Extended-Adaptive-Sampling",
    # "Parallel-Adaptive-Sampling",
    # "MAP-Elites-sampling",
    # "PGA",
]
baselines_as_list = [
    "Vanilla-AS",
]
categories_as_list = [
    "AS-Weighted",
    "AS-Dominance",
]

order = baselines_list + categories_list + baselines_as_list + categories_as_list

env_order = {
    "arm_fit0.01_desc0.01_params0.0": "Arm",
    "hexapod_sin_omni_fit0.05_desc0.05_params0.0": "Hexapod",
    "hexapod_sin_omni": "Hexapod",
    "walker2d_uni": "Walker",
    "ant_omni": "Ant",
    "anttrap": "Ant-Trap",
    "arm_gaussian_desc_bi_variance_fit0.01_desc0.1_params0.0": "Arm Bi-variance",
    "arm_gaussian_desc_fitprop_variance_nonoise": "Arm Fit-prop-variance",
    "direct_mapping_perfect_trade_off_0.02_nonoise": "Linear trade-off",
    "direct_mapping_deceptive_0.1_nonoise": "Deceptive",
    "direct_mapping_sharp_peak_bigger_0.2_nonoise": "Avoidable Peak",
    "direct_mapping_sharp_peak_smaller_0.02_nonoise": "Unavoidable Peak",
}

params = {
    "axes.labelsize": 26,
    "axes.titlesize": 26,
    "legend.fontsize": 22,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "text.usetex": False,
    "axes.titlepad": 10,
    "lines.linewidth": 2,
    "lines.markersize": 8,
}
mpl.rcParams.update(params)

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
config_frame, all_data, all_losses, all_times, all_var = load_results(
    save_folder=save_folder,
    plot_folder=plot_folder,
    plot_algos=plot_algos,
    exclude_algos=exclude_algos,
    exclude_sizes=exclude_sizes,
    not_name=not_name,
    replace_name=replace_name,
    time_pourcent=args.time_pourcent,
    pourcent_value=args.pourcent_value,
    compare_size=compare_size,
    order=order,
)

# Create a color frame
config_frame = sort_data(config_frame, ["algo", compare_size], order)
labels = config_frame["algo"].drop_duplicates().values
colors = sns.color_palette("colorblind", len(labels))
color_frame = pd.DataFrame(data={"Label": labels, "Color": colors})


############################
# Plot convergence metrics #

if args.plot_convergence_metrics:

    print("\nPlotting convergence metrics graphs")
    try:
        plot_convergence_metrics(
            plot_folder=plot_folder,
            all_data=all_data,
            all_times=all_times,
            color_frame=color_frame,
            compare_size=compare_size,
            compare_title=compare_title,
            order=order,
            legend_columns=args.legend_columns,
            legend_bottom=args.legend_bottom,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot metrics.")
        traceback.print_exc()

    if args.plot_in_cell:
        try:
            plot_convergence_metrics(
                plot_folder=plot_folder,
                all_data=all_data,
                all_times=all_times,
                color_frame=color_frame,
                compare_size=compare_size,
                compare_title=compare_title,
                order=order,
                legend_columns=args.legend_columns,
                legend_bottom=args.legend_bottom,
                prefixe="in_cell_",
                prefixe_title="In-Cell ",
            )
        except Exception:
            print("\n!!!WARNING!!! Cannot plot in-cell metrics.")
            traceback.print_exc()

################
# Plot metrics #

if args.plot_metrics:

    print("\nPlotting metrics graphs")
    try:
        plot_metrics(
            plot_folder=plot_folder,
            all_losses=all_losses,
            all_times=all_times,
            color_frame=color_frame,
            compare_size=compare_size,
            compare_title=compare_title,
            order=order,
            legend_columns=args.legend_columns,
            legend_bottom=args.legend_bottom,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot metrics.")
        traceback.print_exc()

    if args.plot_in_cell:
        try:
            plot_metrics(
                plot_folder=plot_folder,
                all_losses=all_losses,
                all_times=all_times,
                color_frame=color_frame,
                compare_size=compare_size,
                compare_title=compare_title,
                order=order,
                legend_columns=args.legend_columns,
                legend_bottom=args.legend_bottom,
                prefixe="in_cell_",
                prefixe_title="In-Cell ",
            )
        except Exception:
            print("\n!!!WARNING!!! Cannot plot in-cell metrics.")
            traceback.print_exc()


################
# Plot summary #

if args.plot_summary:

    print("\nPlotting summary graphs")
    try:
        plot_summary(
            plot_folder=plot_folder,
            all_losses=all_losses,
            all_times=all_times,
            all_var=all_var,
            color_frame=color_frame,
            compare_size=compare_size,
            compare_title=compare_title,
            order=order,
            env_order=env_order,
            legend_columns=args.legend_columns,
            legend_bottom=args.legend_bottom,
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
                dataframe=all_losses,
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
                all_losses=all_losses,
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
                    dataframe=all_losses,
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
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot pareto.")
        traceback.print_exc()

    if args.plot_in_cell:
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
            )
        except Exception:
            print("\n!!!WARNING!!! Cannot plot in-cell pareto.")
            traceback.print_exc()

########################
# Plot algo properties #

if args.plot_algo_properties:

    print("\nPlotting algo properties graphs")
    try:
        plot_algo_properties(
            plot_folder=plot_folder,
            all_data=all_data,
            all_times=all_times,
            color_frame=color_frame,
            compare_size=compare_size,
            compare_title=compare_title,
            order=order,
            legend_columns=args.legend_columns,
            legend_bottom=args.legend_bottom,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot metrics.")
        traceback.print_exc()

######################
# Load min_max_frame #

if (
    args.plot_archives
    or args.plot_all_archives
    or args.plot_robotics_archives
    or args.plot_visualisations
    or args.plot_archive_profiles
):

    try:
        min_max_frame = find_min_max(
            plot_folder=plot_folder,
            config_frame=config_frame,
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

    print("\nPlotting all archives")

    try:
        plot_all_archives(
            plot_folder=plot_folder,
            config_frame=config_frame,
            min_max_frame=min_max_frame,
            compare_size=compare_size,
            compare_title=compare_title,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot all archives.")
        traceback.print_exc()

    if args.plot_in_cell:

        try:
            plot_all_archives(
                plot_folder=plot_folder,
                config_frame=config_frame,
                min_max_frame=in_cell_min_max_frame,
                compare_size=compare_size,
                compare_title=compare_title,
                prefixe="in_cell_",
                prefixe_title="In-Cell ",
            )
        except Exception:
            print("\n!!!WARNING!!! Cannot plot in-cell all archives.")
            traceback.print_exc()

#################
# Plot archives #

if args.plot_archives:

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
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot archives.")
        traceback.print_exc()

    if args.plot_in_cell:

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
            )
        except Exception:
            print("\n!!!WARNING!!! Cannot plot in-cell archives.")
            traceback.print_exc()

#########################
# Plot archive profiles #

if args.plot_archive_profiles:

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
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot archives.")
        traceback.print_exc()

    if args.plot_in_cell:

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
):

    print("\nLoading reproducibilities.")
    try:
        (
            all_reproducibilities_data,
            reprod_repertoire_folders,
            reprod_min_max_frame,
        ) = load_reproducibilities(
            plot_folder=plot_folder,
            config_frame=config_frame,
            compare_size=compare_size,
            order=order,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot load reproducibilities.")
        traceback.print_exc()

    if args.plot_in_cell:

        try:
            (
                in_cell_all_reproducibilities_data,
                in_cell_reprod_repertoire_folders,
                in_cell_reprod_min_max_frame,
            ) = load_reproducibilities(
                plot_folder=plot_folder,
                config_frame=config_frame,
                compare_size=compare_size,
                order=order,
                prefixe="in_cell_",
            )
        except Exception:
            print("\n!!!WARNING!!! Cannot load in_cell_reproducibilities.")
            traceback.print_exc()

################################
# Plot reproducibility metrics #

if args.plot_reproducibility_metrics:

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
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot reproducibility metrics.")
        traceback.print_exc()

    if args.plot_in_cell:
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
            )
        except Exception:
            print("\n!!!WARNING!!! Cannot plot in-cell reproducibility metrics.")
            traceback.print_exc()


##################################
# Plot reproducibility archives #

if args.plot_reproducibility_archives:

    print("\nPlotting reproducibility archives")

    try:
        plot_reproducibility_archives(
            plot_folder=plot_folder,
            reprod_repertoire_folders=reprod_repertoire_folders,
            reprod_min_max_frame=reprod_min_max_frame,
            compare_size=compare_size,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot archives.")
        traceback.print_exc()

    if args.plot_in_cell:

        try:
            plot_reproducibility_archives(
                plot_folder=plot_folder,
                reprod_repertoire_folders=in_cell_reprod_repertoire_folders,
                reprod_min_max_frame=in_cell_reprod_min_max_frame,
                compare_size=compare_size,
                prefixe="in_cell_",
                prefixe_title="In-Cell ",
            )
        except Exception:
            print("\n!!!WARNING!!! Cannot plot in-cell archives.")
            traceback.print_exc()

#########################
# Plot robotics metrics #

if args.plot_robotics:

    print("\nPlotting robotics convergence metrics graphs")
    try:
        plot_robotics(
            plot_folder=plot_folder,
            all_times=all_times,
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
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot robotics metrics.")
        traceback.print_exc()

    if args.plot_in_cell:
        try:
            plot_robotics(
                plot_folder=plot_folder,
                all_times=all_times,
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
            )
        except Exception:
            print("\n!!!WARNING!!! Cannot plot in-cell robotics metrics.")
            traceback.print_exc()

##########################
# Plot robotics archives #

if args.plot_robotics_archives:

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
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot robotics archives.")
        traceback.print_exc()

    if args.plot_in_cell:

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
            )
        except Exception:
            print("\n!!!WARNING!!! Cannot plot in-cell robotics archives.")
            traceback.print_exc()

##################
# Plot benchmark #

if args.plot_benchmark:

    print("\nPlotting benchmark graphs")
    try:
        plot_benchmark(
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
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot benchmark.")
        traceback.print_exc()

###########################
# Plot benchmark archives #

if args.plot_benchmark_archives:

    print("\nPlotting benchmark archives")
    try:
        plot_benchmark_archives(
            plot_folder=plot_folder,
            single_compare_size=args.archives_compare_size,
            env_order=env_order,
            config_frame=config_frame,
            baselines_list=baselines_list,
            categories_list=categories_list,
            baselines_as_list=baselines_as_list,
            categories_as_list=categories_as_list,
            compare_size=compare_size,
            compare_title=compare_title,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot benchmark archives.")
        traceback.print_exc()

#######################
# Plot visualisations #

if args.plot_visualisations:

    print("\nPlotting visualisations")

    try:
        plot_visualisations(
            plot_folder=plot_folder,
            config_frame=config_frame,
            min_max_frame=min_max_frame,
            compare_size=compare_size,
            deterministic=args.deterministic,
            replications=args.replications,
            indiv=args.indiv,
            best_indiv=args.best_indiv,
            save_html=args.save_html,
            paper_plot=args.paper_plot,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot visualisations.")
        traceback.print_exc()
