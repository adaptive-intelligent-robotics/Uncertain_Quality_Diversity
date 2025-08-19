# Uncertain Quality-Diversity

This repository contains the code for the following papers:
- [Uncertain Quality-Diversity: Evaluation methodology and new methods for Quality-Diversity in Uncertain Domains](https://ieeexplore.ieee.org/abstract/document/10120985), introducing the Uncertain Quality-Diversity (UQD) framework.
- [Benchmark tasks for Quality-Diversity applied to Uncertain domains](https://dl.acm.org/doi/abs/10.1145/3583133.3596326), introducing benchmark tasks for UQD.
- [Exploring the Performance-Reproducibility Trade-off in Quality-Diversity](https://ieeexplore.ieee.org/document/10912727), identifying the problem of fitness-reproducibility trade-off in UQD.
- [Extract-QD Framework: A Generic Approach forQuality-Diversity in Noisy, Stochastic or Uncertain Domains](https://dl.acm.org/doi/epdf/10.1145/3712256.3726404), introducing Extract-MAP-Elites and the Extract-QD Framework.
As all files evolve with papers, we provide Git tags to point to the exact code of each paper.

This repository builds on top of the [QDax library](https://github.com/adaptive-intelligent-robotics/QDax), following its structure and defining new classes from the existing QDax classes.

## Content

### Algorithms

On top of all the algorithms already included in QDax, this repository provides implementations for the following UQD algorithms, all implemented as QDax-compatible containers:
- ME-Sampling
- ME-Sampling-Reprod
- Deep-Grid (from [Fast and stable MAP-Elites in noisy domains using deep grids](https://direct.mit.edu/isal/proceedings/isal2020/32/273/98397))
- Archive-Sampling (AS) (from [Uncertain Quality-Diversity: Evaluation methodology and new methods for Quality-Diversity in Uncertain Domains](https://ieeexplore.ieee.org/abstract/document/10120985))
- AS-Reprod
- Parallel-Adaptive-Sampling (from [Uncertain Quality-Diversity: Evaluation methodology and new methods for Quality-Diversity in Uncertain Domains](https://ieeexplore.ieee.org/abstract/document/10120985))
- ME-Weighted (from [Exploring the Performance-Reproducibility Trade-off in Quality-Diversity](https://ieeexplore.ieee.org/document/10912727))
- ME-Delta (from [Exploring the Performance-Reproducibility Trade-off in Quality-Diversity](https://ieeexplore.ieee.org/document/10912727))
- AS-Weighted (from [Exploring the Performance-Reproducibility Trade-off in Quality-Diversity](https://ieeexplore.ieee.org/document/10912727))
- AS-Delta (from [Exploring the Performance-Reproducibility Trade-off in Quality-Diversity](https://ieeexplore.ieee.org/document/10912727))
- MOME-R (from [Exploring the Performance-Reproducibility Trade-off in Quality-Diversity](https://ieeexplore.ieee.org/document/10912727))
- Extract-ME (from [Extract-QD Framework: A Generic Approach forQuality-Diversity in Noisy, Stochastic or Uncertain Domains](https://dl.acm.org/doi/epdf/10.1145/3712256.3726404))

It also provides the code for the following baselines:
- ME-Random: a variant of ME that generates random offspring instead of selecting parents from the grid.
- ME with a depth: a variant of ME that stores $d$ solutions per cell of the grid, $d$ being the depth. This is not expected to perform differently from ME, as the final returned grid only contains the top layer, it is only provided to faciliate later developments.
- Adaptive-Sampling: a jax-compatible implementation of the first UQD algorithm proposed in [Map-elites for noisy domains by adaptive sampling](https://dl.acm.org/doi/abs/10.1145/3319619.3321904?casa_token=CJVMQp-r43kAAAAA:jFHeZl4uSZFriJOJi60xCAW1_X1e7BtKgCFm_9J4gTUe3SyxRDL1MsuLDLssO35GRY1LeTVPINQ). Note that this algorithm is intrinsically non-parallelisable because part of the evaluations are performed sequentially during the addition to the archive. While this code is written in Jax, it follows the original algorithm and is thus sequential and as a consequence is slower than all other algortihms. We recommand running this specific algorithm on CPU as it induces many data tranferts between GPU and CPU.

### Metrics

This repository also provides implementations for the UQD metrics as follow:
- Corrected Repertoire to compute Corrected Metrics: get the groundth truth fitness and descriptor of all solutions in the current repertoire and place them in a new repertoire called Corrected Reperoire. To evaluate the ground truth, use directly the real value for tasks where it is accessible (for example optimisation tasks), or approximate it using N reevaluations for tasks where it cannot be easily acess (for example complex robotics tasks with initial configuration noise).
- Variance Repertoire to compute Reproducibility Metrics: get the reproducibility of all solutions and store it as the fitness in the Repertoire, if not available approximate it using the same N reevaluation as the Corrected Repertoire.
- Cell-aware Corrected Repertoire to compute Corrected Metrics: specific implementation of the Corrected Metrics for the Deep-Grid algorithm, using cell-selector, as defined in its original paper.

### Tasks

This repository also provides compatible implementations of the following tasks that are not included in QDax:
- The [UQD Benchmark tasks](https://arxiv.org/abs/2304.12454) based on the arm and direct mapping tasks
- Noisy Rastrigin and noisy Sphere, adding Gaussian noise on the fitness and descriptor
- Hexapod Omni with reward from [this paper](https://direct.mit.edu/evco/article/24/1/59/1004?casa_token=bZvw7OK9LDoAAAAA:sAvo7rRM3pCk3he3ZW_v_KSPQ44ESToFbDm0-A_s179y8o6RfowglpsTnQDJwlXlOjKIke3o) controlled using open-loop controllers as described in the paper
- Hexapod Omni with reward from [this paper](https://direct.mit.edu/evco/article/24/1/59/1004?casa_token=bZvw7OK9LDoAAAAA:sAvo7rRM3pCk3he3ZW_v_KSPQ44ESToFbDm0-A_s179y8o6RfowglpsTnQDJwlXlOjKIke3o) controlled using closed-loop NN policies

## Structure

This repository builds on QDax and follows the same structure:
- `core` contains the main file definition, in particular:
	- `core` itself contains the main file for each algorithm listed above as well as two files defining the metrics.
	- `core/containers` defines the necessary containers for the UQD algorithms, in particular, the Depth container
	- `core/emitter` defines the necessary emitters
- `tasks` contains the definition of the additional tasks
- `analysis` contains the files necessary to plot the graphs used in the paper
- `singularity` contains all the files necessary to build and execute a singularity container to reproduce the paper results.

## Running the code

### Requirements

To run the code in this repository, you would need the libraries given in `requirement.txt`. In particular, you would need the QDax library, as this repository is built on top of it.
To install all packages directly on your computer, you can run:

```
pip install -r requirements.txt
```

However, to avoid version issues, we recommend using a virtual machine, a pyenv, or a singularity container (we provide instructions for singularity below).

### Running the code

To run any algorithm, you would need to choose its different components:
1. Container: parameter `--container`, for example `Extract-MAP-Elites` or `Archive-Sampling`.
2. Emitter: parameter `--emitter`, most UQD algorithm do not modify the emitter.
3. Specific parameters: most UQD algorihtm uses the `--depth` and `--num-samples` parameters for example.

You would also need to specify a sampling-size (i.e. an evaluation budget per evaluations), using the `--sampling-size` argument and a number of generation, using the `--num-iterations` arguments. To change the environment, use the parameter `--env-name`, by default, it will run a Redundant Arm task with Gaussian Fitness.

For example, to run Extract-MAP-Elites with the same parameters as the paper:
```
python3 main.py --num-iterations 1000 --sampling-size 4096 --container Extract-MAP-Elites --depth 8 --num-samples 2
```

Here are more details on where to find the list of all available configurations:
- Environments: `environments_manager/set_up_environment.py` list all available environments at the top.
- Containers: `set_up_container.py` list all available containers in `CONTAINER_LIST`.
- Emitters: `set_up_emitter.py` list all available emitters in `EMITTER_LIST`.
- Parameters: `main.py` list all available parameters in `0. Input` section in teh argparse definition.

### Parameters for standard algorithms

Here are the standard configuration of UQD algorithms:

- ME-Sampling: `--container MAP-Elites --num-samples 32`
- ME-Sampling-Reprod: `--container MAP-Elites-Sampling-Reprod --num-samples 32`
- Adaptive-Sampling: `--container Adaptive-Sampling --depth 10`
- Deep-Grid: `--container Deep-Grid --depth 32`
- Archive-Sampling (AS): `--container Archive-Sampling --depth 2 --num-samples 2`
- AS-Reprod: `--container Archive-Sampling-Reprod --depth 2 --num-samples 2`
- Parallel-Adaptive-Sampling: `--container Parallel-Adaptive-Sampling --depth 2 --pas-max-samples 8`
- ME-LS: `--container MAP-Elites-Low-Spread --num-samples 32`
- ME-Weighted: `--container MAP-Elites-Weighted --num-samples 32`
- ME-Delta: `--container MAP-Elites-Delta --num-samples 32`
- AS-Weighted: `--container Archive-Sampling-Weighted --depth 2 --num-samples 2`
- AS-Delta: `--container Archive-Sampling-Delta --depth 2 --num-samples 2`
- MOME-R: `--container MOME-Reprod --num-samples 32`
- Extract-ME: `--container Extract-MAP-Elites --depth 8 --num-samples 2`


### Using the singularity containers

This repository also contains the recipe to build a singularity container with all the required libraries.
This container can either be built as a final container (it would then be an executable file) or as a sandbox container (that can then be used to develop code interactively within the container).

To create a final container, go inside `singularity/` and run:
```
python3 build_final_image.py
```

When you get the final image, you can execute it.
For example, if you want to run [REPLICATIONS] replications of Archive-Sampling with depth `2` as in the paper for `1000` generations on the `ant_omni` task, with sampling-size `4096`, and save the results in results_[RESULTSNAME] you would run:

```
singularity run --nv [IMAGE_NAME] [RESULTNAME] [REPLICATIONS] --container Archive-Sampling --emitter Mixing --num-iterations 1000 --sampling-size 4096 --env-name ant_omni
```

The `--nv` option allows you to use the Nvidia drivers from your computer within the container.



To build a sandbox container, go inside `singularity/` and run:
```
python3 start_container.py -n
```

Again, the `-n` option allows using the Nvidia drivers from your computer within the container.
You would then enter the sandbox and be able to develop your code in this environment.
