import os
from typing import List, Tuple


def save_config(
    save_folder: str,
    name: str,
    seed: int,
    env_name: str,
    episode_length: int,
    params_std: float,
    min_bd: List,
    max_bd: List,
    batch_size: int,
    sampling_size: int,
    sampling_use: int,
    num_iterations: int,
    policy_hidden_layer_sizes: Tuple,
    num_init_cvt_samples: int,
    num_centroids: int,
    num_samples: int,
    num_reevals: int,
    depth: int,
    delta_fitness: float,
    delta_reproducibility: float,
    metrics_file: str = "",
    in_cell_metrics_file: str = "",
    save_folder_repertoire: str = "",
    save_folder_projected_repertoire: str = "",
    save_folder_reeval_repertoire: str = "",
    save_folder_fit_reeval_repertoire: str = "",
    save_folder_desc_reeval_repertoire: str = "",
    save_folder_fit_var_repertoire: str = "",
    save_folder_reeval_fit_var_repertoire: str = "",
    save_folder_desc_var_repertoire: str = "",
    save_folder_reeval_desc_var_repertoire: str = "",
    save_folder_additional_repertoire: str = "",
    save_folder_reeval_additional_repertoire: str = "",
    save_folder_in_cell_reeval_repertoire: str = "",
    save_folder_in_cell_fit_reeval_repertoire: str = "",
    save_folder_in_cell_desc_reeval_repertoire: str = "",
    save_folder_in_cell_fit_var_repertoire: str = "",
    save_folder_in_cell_reeval_fit_var_repertoire: str = "",
    save_folder_in_cell_desc_var_repertoire: str = "",
    save_folder_in_cell_reeval_desc_var_repertoire: str = "",
) -> str:
    """Save the current config in the config.csv file.
    Create it if necessary."""

    # Create results folder if needed
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # Opening config file and writing header
    file_name = f"{save_folder}/config.csv"
    if not os.path.exists(file_name):
        config = open(file_name, "w")
        header = (
            "run,seed,env,episode_length,params_std,min_bd,max_bd,"
            + "batch_size,sampling_size,sampling_use,"
            + "num_iterations,policy_hidden_layer_sizes,"
            + "num_init_cvt_samples,num_centroids,"
            + "num_samples,num_reevals,depth,"
            + "delta_fitness,delta_reproducibility,"
            + "metrics_file,in_cell_metrics_file,"
            + "repertoire_folder,"
            + "projected_repertoire_folder,"
            + "reeval_repertoire_folder,"
            + "fit_reeval_repertoire_folder,desc_reeval_repertoire_folder,"
            + "fit_var_repertoire_folder,reeval_fit_var_repertoire_folder,"
            + "desc_var_repertoire_folder,reeval_desc_var_repertoire_folder,"
            + "additional_folder,reeval_additional_folder,"
            + "in_cell_reeval_repertoire_folder,"
            + "in_cell_fit_reeval_repertoire_folder,in_cell_desc_reeval_repertoire_folder,"
            + "in_cell_fit_var_repertoire_folder,in_cell_reeval_fit_var_repertoire_folder,"
            + "in_cell_desc_var_repertoire_folder,in_cell_reeval_desc_var_repertoire_folder"
        )
        config.write(f"{header}\n")
    else:
        config = open(file_name, "a")

    # Writting config
    layer = "_".join(map(str, policy_hidden_layer_sizes))
    cvt = num_init_cvt_samples
    params = (
        f"{name},{seed},{env_name},{episode_length},{params_std},{min_bd},{max_bd},"
        + f"{batch_size},{sampling_size},{sampling_use},"
        + f"{num_iterations},{layer},{cvt},{num_centroids},"
        + f"{num_samples},{num_reevals},{depth},"
        + f"{delta_fitness},{delta_reproducibility},"
        + f"{metrics_file},{in_cell_metrics_file},"
        + f"{save_folder_repertoire},"
        + f"{save_folder_projected_repertoire},"
        + f"{save_folder_reeval_repertoire},"
        + f"{save_folder_fit_reeval_repertoire},{save_folder_desc_reeval_repertoire},"
        + f"{save_folder_fit_var_repertoire},{save_folder_reeval_fit_var_repertoire},"
        + f"{save_folder_desc_var_repertoire},{save_folder_reeval_desc_var_repertoire},"
        + f"{save_folder_additional_repertoire},{save_folder_reeval_additional_repertoire},"
        + f"{save_folder_in_cell_reeval_repertoire},"
        + f"{save_folder_in_cell_fit_reeval_repertoire},{save_folder_in_cell_desc_reeval_repertoire},"
        + f"{save_folder_in_cell_fit_var_repertoire},{save_folder_in_cell_reeval_fit_var_repertoire},"
        + f"{save_folder_in_cell_desc_var_repertoire},{save_folder_in_cell_reeval_desc_var_repertoire}"
    )
    config.write(f"{params}\n")
    config.close()

    return file_name


def create_metrics_csv(file_name: str, prefixe: str) -> None:

    # Opening metrics file
    file_metrics = open(file_name, "w")

    # Writing top line
    file_metrics.write(
        f"epoch,eval,time,"
        + f"{prefixe}qd_score,{prefixe}coverage,{prefixe}max_fitness,{prefixe}min_fitness,"
        + f"{prefixe}reeval_qd_score,{prefixe}reeval_coverage,"
        + f"{prefixe}reeval_max_fitness,{prefixe}reeval_min_fitness,"
        + f"{prefixe}fit_reeval_qd_score,{prefixe}fit_reeval_coverage,"
        + f"{prefixe}fit_reeval_max_fitness,{prefixe}fit_reeval_min_fitness,"
        + f"{prefixe}desc_reeval_qd_score,{prefixe}desc_reeval_coverage,"
        + f"{prefixe}desc_reeval_max_fitness,{prefixe}desc_reeval_min_fitness,"
        + f"{prefixe}fit_var_qd_score,{prefixe}fit_var_coverage,"
        + f"{prefixe}fit_var_max_fitness,{prefixe}desc_reeval_min_fitness,"
        + f"{prefixe}reeval_fit_var_qd_score,{prefixe}reeval_fit_var_coverage,"
        + f"{prefixe}reeval_fit_var_max_fitness,{prefixe}desc_reeval_min_fitness,"
        + f"{prefixe}desc_var_qd_score,{prefixe}desc_var_coverage,"
        + f"{prefixe}desc_var_max_fitness,{prefixe}desc_var_min_fitness,"
        + f"{prefixe}reeval_desc_var_qd_score,{prefixe}reeval_desc_var_coverage,"
        + f"{prefixe}reeval_desc_var_max_fitness,{prefixe}reeval_desc_var_min_fitness,"
        + f"{prefixe}additional_qd_score,{prefixe}additional_coverage,"
        + f"{prefixe}additional_max_fitness,{prefixe}additional_min_fitness,"
        + f"{prefixe}reeval_additional_qd_score,{prefixe}reeval_additional_coverage,"
        + f"{prefixe}reeval_additional_max_fitness,{prefixe}reeval_additional_min_fitness,"
        + "num_samples_per_indiv,num_samples_per_iter,batch_size\n"
    )

    # Closing file
    file_metrics.flush()
    file_metrics.close()


def write_metrics_csv(
    file_name: str,
    epch: float,
    evl: float,
    time: float,
    qds: float,
    cov: float,
    maxf: float,
    minf: float,
    rqds: float,
    rcov: float,
    rmaxf: float,
    rminf: float,
    rfqds: float,
    rfcov: float,
    rfmaxf: float,
    rfminf: float,
    rdqds: float,
    rdcov: float,
    rdmaxf: float,
    rdminf: float,
    vfqds: float,
    vfcov: float,
    vfmaxf: float,
    vfminf: float,
    rvfqds: float,
    rvfcov: float,
    rvfmaxf: float,
    rvfminf: float,
    vdqds: float,
    vdcov: float,
    vdmaxf: float,
    vdminf: float,
    rvdqds: float,
    rvdcov: float,
    rvdmaxf: float,
    rvdminf: float,
    addqds: float,
    addcov: float,
    addmaxf: float,
    addminf: float,
    raddqds: float,
    raddcov: float,
    raddmaxf: float,
    raddminf: float,
    nsamples: int,
    isamples: int,
    batch: int,
) -> None:

    # Opening metrics file
    file_metrics = open(file_name, "a")

    # Saving metrics
    file_metrics.write(
        f"{epch},{evl},{time},{qds},{cov},{maxf},{minf},"
        + f"{rqds},{rcov},{rmaxf},{rminf},"
        + f"{rfqds},{rfcov},{rfmaxf},{rfminf},"
        + f"{rdqds},{rdcov},{rdmaxf},{rdminf},"
        + f"{vfqds},{vfcov},{vfmaxf},{vfminf},"
        + f"{rvfqds},{rvfcov},{rvfmaxf},{rvfminf},"
        + f"{vdqds},{vdcov},{vdmaxf},{vdminf},"
        + f"{rvdqds},{rvdcov},{rvdmaxf},{rvdminf},"
        + f"{addqds},{addcov},{addmaxf},{addminf},"
        + f"{raddqds},{raddcov},{raddmaxf},{raddminf},"
        + f"{nsamples},{isamples},{batch}\n"
    )

    # Closing file
    file_metrics.flush()
    file_metrics.close()
