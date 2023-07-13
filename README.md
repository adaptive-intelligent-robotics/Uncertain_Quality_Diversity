This repository contains the code for the paper [Uncertain Quality-Diversity: Evaluation methodology and new methods for Quality-Diversity in Uncertain Domains](https://ieeexplore.ieee.org/abstract/document/10120985), introducing the Uncertain Quality-Diversity (UQD) framework, as well as the paper [Benchmark tasks for Quality-Diversity applied to Uncertain domains](https://arxiv.org/abs/2304.12454), introducing benchmark tasks for UQD. 

This repository builds on top of the [QDax library](https://github.com/adaptive-intelligent-robotics/QDax), following its structure and defining new classes from the existing QDax classes.
In particular, it provides the QDax code for:
- The following UQD algorithms:
	- MAP-Elites-sampling
	- MAP-Elites with a depth
	- Deep-Grid
	- Archive-Sampling
	- Parallel-Adaptive-Sampling
	- MAP-Elites-Random
- The following mechanisms for UQD setting:
	- reevaluation of the archive, either using the average or the median, giving back Corrected repertoire and Variance repertoire
	- reevaluation of the archive in the specific case of Deep-Grid using in-cell selector, either using the average or the median, giving back Corrected repertoire and Variance repertoire
- The following tasks that are not provided in the original QDax library:
	- The [UQD Benchmark tasks](https://arxiv.org/abs/2304.12454) based on the arm tasks
	- Noisy Rastrigin and noisy Sphere, adding Gaussian noise on the fitness and descriptor
	- Hexapod Omni with reward from [this paper](https://direct.mit.edu/evco/article/24/1/59/1004?casa_token=bZvw7OK9LDoAAAAA:sAvo7rRM3pCk3he3ZW_v_KSPQ44ESToFbDm0-A_s179y8o6RfowglpsTnQDJwlXlOjKIke3o) controlled using open-loop controllers as described in the paper
	- Hexapod Omni with reward from [this paper](https://direct.mit.edu/evco/article/24/1/59/1004?casa_token=bZvw7OK9LDoAAAAA:sAvo7rRM3pCk3he3ZW_v_KSPQ44ESToFbDm0-A_s179y8o6RfowglpsTnQDJwlXlOjKIke3o) controlled using closed-loop NN policies


## Structure

This repository builds on QDax and follows the same structure:
- `core` contains the main file definition, in particular:
	- `core` itself contains the main file for each UQD algorithm listed above
	- `core` itself also contains the two files defining the reevaluation mechanisms described above
	- `core/containers` defines the necessary containers for the UQD algorithms, in particular, the Depth container
	- `core/emitter` defines the necessary emitters
- `tasks` contains the definition of the additional tasks, in particular uqd_benchmark.py contains the definition of the UQD Benchmark tasks proposed in [Benchmark tasks for Quality-Diversity applied to Uncertain domains](https://arxiv.org/abs/2304.12454)
- `analysis` contains the files necessary to plot the graphs used in the paper
- `singularity` contains all the files necessary to build and execute a singularity container to reproduce the paper results.


## Necessary libraries

To run the code in this repository, you would need the libraries given in `requirement.txt`. In particular, you would need the QDax library, as this repository is built on top of it.
To install all packages directly on your computer, you can run:

```
pip install -r requirements.txt
```

However, to avoid version issues, we recommend using a virtual machine, a pyenv, or a singularity container as described below.

## Running the code

To run an algorithm from the paper, you would need to choose its container and its emitter. For example, to run Archive-Sampling, you would use an Archive-Sampling container (of depth `2` as in the paper) with a Mixing (standard) emitter. 
For example, to run it for `1000` generations on the `ant_omni` task, with sampling-size `4096` you would run:

```
python3 main.py --container Archive-Sampling --emitter Mixing --num-iterations 1000 --sampling-size 4096 --env-name ant_omni
```

To run only the benchmark task example from [Benchmark tasks for Quality-Diversity applied to Uncertain domains](https://arxiv.org/abs/2304.12454), you can use directly the correpsonding main file:

```
python3 main_uqd_benchmark_task.py alg_name=me noise_type=gaussian_fit
```

alg_name allows to specify the algorithm to run and noise_type the type of noise from the Benchmark paper.

## Using the singularity containers

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
