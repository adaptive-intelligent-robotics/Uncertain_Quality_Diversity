BASELINES = ["Vanilla", "Random"]
CATEGORIES = [
    "Deep-Grid ",
    "Deep-Grid-sampling",
    "Deep-Bias",
    "Deep-Target",
    "Archive-Sampling",
    "Extended-Adaptive-Sampling",
    "Parallel-Adaptive-Sampling",
    "MAP-Elites-sampling",
    "PGA",
    "MAP-Elites-2Calls",
    "All-ES",
    "ESSimultaneous",
    "ESGA",
    "ES",
]

ENV_NAME_DIFFICULTY = {
    "arm_fit0.01_desc0.01_params0.0": "Arm",
    "hexapod_sin_omni_fit0.05_desc0.05_params0.0": "Hexapod",
    "hexapod_sin_omni": "Hexapod",
    "walker2d_uni": "Walker",
    "ant_omni": "Ant",
}
ENV_NO_VAR = [
    "arm_fit0.01_desc0.01_params0.0",
    "hexapod_sin_omni_fit0.05_desc0.05_params0.0",
]
