import os
from typing import Tuple


def save_config(
    save_folder: str,
    name: str,
    seed: int,
    env_name: str,
    episode_length: int,
    min_bd: float,
    max_bd: float,
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
    metrics_file: str = "",
    in_cell_metrics_file: str = "",
    save_folder_repertoire: str = "",
    save_folder_reeval_repertoire: str = "",
    save_folder_fit_reeval_repertoire: str = "",
    save_folder_desc_reeval_repertoire: str = "",
    save_folder_fit_var_repertoire: str = "",
    save_folder_desc_var_repertoire: str = "",
    save_folder_in_cell_reeval_repertoire: str = "",
    save_folder_in_cell_fit_reeval_repertoire: str = "",
    save_folder_in_cell_desc_reeval_repertoire: str = "",
    save_folder_in_cell_fit_var_repertoire: str = "",
    save_folder_in_cell_desc_var_repertoire: str = "",
) -> str:

    # Create results folder if needed
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # Opening config file
    file_name = f"{save_folder}/config.csv"
    if not os.path.exists(file_name):
        config = open(file_name, "w")
        env_header = "env,episode_length,min_bd,max_bd"
        algo_header = (
            "batch_size,sampling_size,sampling_use,"
            + "num_iterations,policy_hidden_layer_sizes,"
            + "num_init_cvt_samples,num_centroids"
        )
        stoch_header = "num_samples,num_reevals,depth"
        run_header = (
            "metrics_file,in_cell_metrics_file,"
            + "repertoire_folder,"
            + "reeval_repertoire_folder,"
            + "fit_reeval_repertoire_folder,desc_reeval_repertoire_folder,"
            + "fit_var_repertoire_folder,desc_var_repertoire_folder,"
            + "in_cell_reeval_repertoire_folder,"
            + "in_cell_fit_reeval_repertoire_folder,in_cell_desc_reeval_repertoire_folder,"
            + "in_cell_fit_var_repertoire_folder,in_cell_desc_var_repertoire_folder"
        )
        config.write(
            f"run,seed,{env_header},{algo_header},{stoch_header},{run_header}\n"
        )
    else:
        config = open(file_name, "a")

    # Saving config
    env_params = f"{env_name},{episode_length},{min_bd},{max_bd}"
    layer = "_".join(map(str, policy_hidden_layer_sizes))
    cvt = num_init_cvt_samples
    algo_params = (
        f"{batch_size},{sampling_size},{sampling_use},"
        + f"{num_iterations},{layer},{cvt},{num_centroids}"
    )
    stoch_params = f"{num_samples},{num_reevals},{depth}"
    run_objects = (
        f"{metrics_file},{in_cell_metrics_file},"
        + f"{save_folder_repertoire},"
        + f"{save_folder_reeval_repertoire},"
        + f"{save_folder_fit_reeval_repertoire},{save_folder_desc_reeval_repertoire},"
        + f"{save_folder_fit_var_repertoire},{save_folder_desc_var_repertoire},"
        + f"{save_folder_in_cell_reeval_repertoire},"
        + f"{save_folder_in_cell_fit_reeval_repertoire},{save_folder_in_cell_desc_reeval_repertoire},"
        + f"{save_folder_in_cell_fit_var_repertoire},{save_folder_in_cell_desc_var_repertoire}"
    )
    config.write(
        f"{name},{seed},{env_params},{algo_params},{stoch_params},{run_objects}\n"
    )
    config.close()

    return file_name


def create_metrics_csv(
    save_folder: str,
    name: str,
    seed: int,
    epch: float,
    evl: float,
    time: float,
    qds: float,
    cov: float,
    maxf: float,
    rqds: float,
    rcov: float,
    rmaxf: float,
    rfqds: float,
    rfcov: float,
    rfmaxf: float,
    rdqds: float,
    rdcov: float,
    rdmaxf: float,
    vfqds: float,
    vfcov: float,
    vfmaxf: float,
    vdqds: float,
    vdcov: float,
    vdmaxf: float,
    nsamples: int,
    isamples: int,
    batch: int,
    prefixe: str = "",
) -> str:

    # Create results folder if needed
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # Opening metrics file
    file_name = (
        f"{save_folder}/" + prefixe + "metrics_" + name + "_" + str(seed) + ".csv"
    )
    file_metrics = open(file_name, "w")
    file_metrics.write(
        f"epoch,eval,time,{prefixe}qd_score,{prefixe}coverage,{prefixe}max_fitness,"
        + f"{prefixe}reeval_qd_score,{prefixe}reeval_coverage,{prefixe}reeval_max_fitness,"
        + f"{prefixe}fit_reeval_qd_score,{prefixe}fit_reeval_coverage,{prefixe}fit_reeval_max_fitness,"
        + f"{prefixe}desc_reeval_qd_score,{prefixe}desc_reeval_coverage,{prefixe}desc_reeval_max_fitness,"
        + f"{prefixe}fit_var_qd_score,{prefixe}fit_var_coverage,{prefixe}fit_var_max_fitness,"
        + f"{prefixe}desc_var_qd_score,{prefixe}desc_var_coverage,{prefixe}desc_var_max_fitness,"
        + "num_samples_per_indiv,num_samples_per_iter,batch_size\n"
    )

    # Saving metrics
    file_metrics.write(
        f"{epch},{evl},{time},{qds},{cov},{maxf},"
        + f"{rqds},{rcov},{rmaxf},"
        + f"{rfqds},{rfcov},{rfmaxf},"
        + f"{rdqds},{rdcov},{rdmaxf},"
        + f"{vfqds},{vfcov},{vfmaxf},"
        + f"{vdqds},{vdcov},{vdmaxf},"
        + f"{nsamples},{isamples},{batch}\n"
    )
    file_metrics.flush()
    file_metrics.close()

    return file_name


def write_metrics_csv(
    file_name: str,
    epch: float,
    evl: float,
    time: float,
    qds: float,
    cov: float,
    maxf: float,
    rqds: float,
    rcov: float,
    rmaxf: float,
    rfqds: float,
    rfcov: float,
    rfmaxf: float,
    rdqds: float,
    rdcov: float,
    rdmaxf: float,
    vfqds: float,
    vfcov: float,
    vfmaxf: float,
    vdqds: float,
    vdcov: float,
    vdmaxf: float,
    nsamples: int,
    isamples: int,
    batch: int,
) -> str:

    file_metrics = open(file_name, "a")
    file_metrics.write(
        f"{epch},{evl},{time},{qds},{cov},{maxf},"
        + f"{rqds},{rcov},{rmaxf},"
        + f"{rfqds},{rfcov},{rfmaxf},"
        + f"{rdqds},{rdcov},{rdmaxf},"
        + f"{vfqds},{vfcov},{vfmaxf},"
        + f"{vdqds},{vdcov},{vdmaxf},"
        + f"{nsamples},{isamples},{batch}\n"
    )
    file_metrics.flush()
    file_metrics.close()
    return file_name
