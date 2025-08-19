import matplotlib as mpl

always_name = [
    "num_samples",
    "depth",
]

not_name = [
    "seed",
    # Already part of the name
    "run",
    "name",
    "container",
    "emitter",
    "suffixe",
    # Batch-size and Sampling-size
    "batch_size",
    "effective_batch_size",
    "real_evals_per_iter",
    "init_batch_size",
    "sampling_size",
    "sampling_use",
    # Metrics logging
    "size_log_period",
    "log_period",
    "archive_log_period",
    "qd_offset",
    "mome_projected_delta_fitness",
    "mome_projected_delta_reproducibility",
    # Task settings
    "env",
    "env_name",
    "policy_hidden_layer_sizes",
    "num_timesteps",
    "num_iterations",
    "episode_length",
    "min_bd",
    "max_bd",
    "params_std",
    "desc_std",
    "fit_std",
    "num_centroids",
    "euclidean_centroids",
    "euclidean_grid_shape",
    # Files and folders
    "folder",
    "results",
    "metrics_file",
    "in_cell_metrics_file",
    "results_repertoire",
    "results_projected_repertoire",
    "results_target_repertoire",
    "results_reeval_repertoire",
    "results_average_repertoire",
    "results_fit_reeval_repertoire",
    "results_fit_average_repertoire",
    "results_desc_reeval_repertoire",
    "results_desc_average_repertoire",
    "results_fit_var_repertoire",
    "results_desc_var_repertoire",
    "results_additional_repertoire",
    "results_reeval_additional_repertoire",
    "results_in_cell_reeval_repertoire",
    "results_in_cell_fit_reeval_repertoire",
    "results_in_cell_desc_reeval_repertoire",
    "results_in_cell_fit_var_repertoire",
    "results_in_cell_desc_var_repertoire",
    "results_reeval_fit_var_repertoire",
    "results_reeval_desc_var_repertoire",
    "results_in_cell_reeval_fit_var_repertoire",
    "results_in_cell_reeval_desc_var_repertoire",
    "results_in_cell_reeval_additional_repertoire",
    "results_in_cell_reeval_additional_repertoire",
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
    "as_max_number_evals",
    "delta_fitness",
    "delta_reproducibility",
    "mer_delta_fitness",
    "mer_delta_reproducibility",
]

replace_name = {
    # Emitter params
    "MAP-Elites": "ME",
    "-Mixing": "",
    # Container params
    "num_samples": "smpl",
    "smpl:0": "smpl:1",  # Old data have 0 as default
    "depth:0": "depth:1",  # Old data have 0 as default
    # Vanilla-ME
    "ME smpl:1 depth:1": "Vanilla-ME",
    # ME-Random
    "ME-Random smpl:1 depth:1": "ME-Random",
    # ME-Sampling
    "ME-sampling-32 smpl:32 depth:1": "ME-Sampling-32",
    "ME smpl:32 depth:1": "ME-Sampling-32",
    "ME-sampling-64 smpl:64 depth:1": "ME-Sampling-64",
    "ME smpl:64 depth:1": "ME-Sampling-64",
    "ME-sampling-128 smpl:128 depth:1": "ME-Sampling-128",
    "ME smpl:128 depth:1": "ME-Sampling-128",
    "ME-sampling-256 smpl:256 depth:1": "ME-Sampling-256",
    "ME smpl:256 depth:1": "ME-Sampling-256",
    "ME-Sampling-32": "ME-Sampling",  # smpl 32 as default
    # ME-Sampling-Reprod
    "ME-Sampling-Reprod smpl:32 depth:1": "ME-Sampling-Reprod-32",
    "ME-Sampling-Reprod-32": "ME-Sampling-Reprod",  # smpl 32 as default
    # ME-LS
    "ME-Low-Spread smpl:32 depth:1": "ME-LS-32",
    "ME-LS-32": "ME-LS",  # smpl 32 as default
    # Deep-Grid
    "Deep-Grid smpl:1 depth:32": "Deep-Grid-32",
    "Deep-Grid smpl:1 depth:64": "Deep-Grid-64",
    "Deep-Grid smpl:1 depth:128": "Deep-Grid-128",
    "Deep-Grid-32": "Deep-Grid",  # depth 32 as default
    # Deep-Grid-sampling
    "Deep-Grid-depth-32 smpl:8 depth:32": "Deep-Grid-Sampling-8",
    "Deep-Grid-depth-32 smpl:32 depth:32": "Deep-Grid-Sampling-32",
    "Deep-Grid-depth-32 smpl:64 depth:32": "Deep-Grid-Sampling-64",
    "Deep-Grid-depth-32 smpl:128 depth:32": "Deep-Grid-Sampling-128",
    "Deep-Grid-Sampling-8": "Deep-Grid-Sampling",  # smpl 8 as default
    # Adaptive-Sampling
    "Adaptive-Sampling smpl:1 depth:10": "Adapt-ME",  # depth 10 as default
    "Adaptive-Sampling smpl:1 depth:8": "Adapt-ME-8",
    "Adaptive-Sampling": "Adapt-ME",
    # PAS
    "Extended-Adaptive-Sampling-median": "PAS",
    "Parallel-Adaptive-Sampling": "PAS",
    # AS
    "Archive-Sampling": "Vanilla-AS",
    "Vanilla-AS smpl:1 depth:1": "Vanilla-AS-1",
    "Vanilla-AS smpl:1 depth:2": "Vanilla-AS-2",
    "Vanilla-AS smpl:1 depth:4": "Vanilla-AS-4",
    "Vanilla-AS smpl:2 depth:2": "Vanilla-AS-2-2",
    # "Vanilla-AS-2-2": "Vanilla-AS", # smpl 2 depth 2 as default
    "Vanilla-AS-2": "Vanilla-AS",  # smpl 1 depth 2 as default
    # "Vanilla-AS": "AS", # remove Vanilla
    # AS-Reprod
    "AS-Reprod smpl:2 depth:2": "AS-Reprod-2-2",
    "AS-Reprod-2-2": "AS-Reprod",  # smpl 2 depth 2 as default
    "Vanilla-AS-Reprod": "AS-Reprod",  # remove Vanilla
    # All delta parametrisation
    "-f140.0-r0.14-0.0001": "",  # Hexapod
    "-f140-r0.14-0.0001": "",
    "-f140.0-r0.14": "",
    "-f220.0-r6.0-0.0001": "",  # Ant Omni
    "-f220-r6.0-0.0001": "",
    "-f220.0-r6.0": "",
    "-f260.0-r0.04-0.0001": "",  # Walker
    "-f260-r0.04-0.0001": "",
    "-f260.0-r0.04": "",
    "-f0.05-r0.05-0.0001": "",  # Deceptive
    "-f0.05-r0.05": "",
    "-f0.02-r0.02-0.0001": "",  # Perfect Trade-Off / Smaller Peak
    "-f0.02-r0.02": "",
    "-f0.2-r0.02-0.0001": "",  # Bigger Peak
    "-f0.2-r0.02": "",
    "-f0.2-r0.0-0.0001": "",  # Bi-Variance / Fit-Prop
    "-f0.2-r0.0": "",
    # ME-Weighted
    "ME-WeightedReprod smpl:32 depth:1": "ME-Weighted-32",
    "ME-Weighted smpl:32 depth:1": "ME-Weighted-32",
    "ME-Weighted-32": "ME-Weighted",  # smpl 32 as default
    # AS-Weighted
    "AS-Weighted smpl:2 depth:2": "AS-Weighted-2-2",
    "AS-Weighted-2-2": "AS-Weighted",  # smpl 2 depth 2 as default
    "Vanilla-AS-Weighted": "AS-Weighted",  # remove Vanilla
    # ME-Delta
    "ME-Delta smpl:32 depth:1": "ME-Delta-32",
    "ME-Delta-32": "ME-Delta",  # smpl 32 as default
    # AS-Delta
    "AS-Delta smpl:2 depth:2": "AS-Delta-2-2",
    "AS-Delta-2-2": "AS-Delta",  # smpl 2 depth 2 as default
    "Vanilla-AS-Delta": "AS-Delta",  # remove Vanilla
    # MOME-R
    "mome_max_pareto_front_length": "pareto",
    "mome_biased_sampling:True": "biased",
    "mome_biased_sampling:False": "",
    "pareto:50 ": "",
    "MOME-Reprod-proj-ffit-rfit-0.0001": "MOME-R-fit",
    "MOME-Reprod-proj-freprod-rreprod-0.0001": "MOME-R-reprod",
    "MOME-Reprod-proj": "MOME-R",
    "MOME-Reprod-biased-proj": "MOME-R-biased",
    "MOME-R smpl:32 depth:1": "MOME-R-32",
    "MOME-R-fit smpl:32 depth:1": "MOME-R-32-fit",
    "MOME-R-reprod smpl:32 depth:1": "MOME-R-32-reprod",
    "MOME-R-32": "MOME-R",  # smpl 32 as default
    # EME
    "extract_proportion_resample": "p_extract",
    "extract_type:proportional_harmonic": "PropHarm",
    "extract_type:proportional_linear": "PropLinear",
    "extract_type:proportional_exponential": "Prop",
    "extract_type:proportional": "Prop",
    "extract_type:uniform": "Uniform",
    "Extract-ME smpl:2 depth:8": "EME",  # smpl 2 depth 8 as default
    # EPGA
    "Extract-ME-PGA smpl:1 depth:8": "EPGA",
    # Clean up
    "  ": " ",
    " .0": " ",
    "ME-Random-Random": "Random",
    "Vanilla-Vanilla": "Vanilla",
    "": "",
}

# Env order
env_order = {
    "arm_fit0.01_desc0.01_params0.0": "Arm",
    "hexapod_sin_omni_fit0.05_desc0.05_params0.0": "Hexapod",
    "hexapod_sin_omni": "Hexapod",
    "walker2d_uni": "Walker",
    "ant_omni": "Ant Omni",
    "ant_uni": "Ant Uni",
    "anttrap": "Ant-Trap",
    "halfcheetah_uni": "HalfCheetah Feet",
    "arm_gaussian_desc_bi_variance_fit0.01_desc0.1_params0": "Arm Two Var",
    "arm_gaussian_desc_fitprop_variance_nonoise": "Arm Continuous Var",
    "direct_mapping_perfect_trade_off_0.02_nonoise": "Linear trade-off",
    "direct_mapping_deceptive_0.1_nonoise": "Deceptive",
    "direct_mapping_sharp_peak_bigger_0.2_nonoise": "Avoidable Peak",
    "direct_mapping_sharp_peak_smaller_0.02_nonoise": "Unavoidable Peak",
    "walker2d_uni_generalised": "Walker Feet",
    "ant_uni_generalised": "Ant Feet",
    "ant_velocity_generalised": "Ant Velocity",
    "ant_angle_generalised": "Ant Angle",
    "humanoid_uni_generalised": "Humanoid Feet",
}

# Env max xaxis
env_max_xaxis = {
    "arm_gaussian_desc_fit0_desc0.05_params0": 4000,
    "arm_gaussian_fit_fit0.1_desc0_params0": 4000,
    "arm_multi_modal_fit_fit0.01_desc0.01_params0": 4000,
    "arm_fit0.01_desc0.01_params0.0": 2000,
    "arm_gaussian_fit0.01_desc0.01_params0": 2000,
    "arm_gaussian_desc_bi_variance_fit0.01_desc0.1_params0": 8000,
    "direct_mapping_deceptive_0.1_nonoise": 8000,
    "direct_mapping_perfect_trade_off_0.02_nonoise": 8000,
    "direct_mapping_sharp_peak_smaller_0.02_nonoise": 8000,
    "direct_mapping_sharp_peak_bigger_0.2_nonoise": 8000,
    "arm_gaussian_desc_fitprop_variance_nonoise": 8000,
    "ant_omni": 3000,
    "hexapod_sin_omni_fit0.05_desc0.05_params0.0": 3000,
    "walker2d_uni": 2000,
}

# Display parameters
params = {
    "font.size": 28,
    "axes.labelsize": 28,
    "axes.titlesize": 28,
    "legend.fontsize": 28,
    "xtick.labelsize": 28,
    "ytick.labelsize": 28,
    "text.usetex": False,
    "axes.titlepad": 10,
    "lines.linewidth": 3.5,
    "lines.markersize": 8,
}
mpl.rcParams.update(params)
