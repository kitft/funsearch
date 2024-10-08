# FunSearch

Forked from https://github.com/google-deepmind/funsearch via https://github.com/jonppe/funsearch

## Overview

FunSearch is an implementation of an evolutionary algorithm for program search using large language models. This fork provides an asynchronous and parallel implementation with the following features:

- Asynchronous main loop using `asyncio`
- Evaluators run in separate processes via `multiprocessing`
- Thread-safe communication using `multiprocessing.Queue`
- Concurrent sampling with multiple asynchronous API calls
- Asynchronous database operations for non-blocking access
- Adaptive sampling rate based on CPU/evaluator capacity

## Installation and Setup

### Using Docker (Recommended)

1. Install Docker. Make sure the docker service is running.
   
2. Clone the repository and navigate to the project directory

`git clone https://github.com/kitft/funsearch.git` (or `gh repo clone kitft/funsearch` if you have GitHub CLI)
`cd funsearch`
   
3. Set your Mistral API key:
   ```
   export MISTRAL_API_KEY=<your_key_here>
   ```
4. Build the Docker image. This may take a while, especially the first time. You will need to re-build if you update the Dockerfile or edit the code. You should not need to re-build if you are only updating the input python file or input data.

   ```
   docker build . -t funsearch
   ```
5. Create a data folder and run the container:
   ```
   mkdir data
   docker run -it -v ./data:/workspace/data -p 6006:6006 -e MISTRAL_API_KEY=$MISTRAL_API_KEY funsearch


```
# Run the search with the desired model using the '--model_name' attribute.
# A good one to use for testing, as an extremely cheap model, is 'mistral-tiny-latest'. 
# The best one for our use case is probably 'codestral-latest', which is 4x more expensive per output token.
# Either model is still relatively cheap: with codestral, 1 million output tokens is $0.6.
# We will need to manually add different API calls for OpenAI, Anthropic, etc. This is not difficult.

`funsearch run` takes two arguments:

1. `SPEC_FILE`: A Python module that provides the basis of the LLM prompt as well as the evaluation metric.
   - Example: See `examples/cap_set_spec.py`

2. `INPUTS`: Can be one of the following:
   - A filename ending in .json or .pickle
   - Comma-separated input data
   - The files are expected to contain a list with at least one element
   - Elements will be passed to the `solve()` method one by one
   - As it stands, we simply sum over the scores of each input. We may want to consider changing this, to appropriately weight the different results.

Examples of valid INPUTS:
- 8
- 8,9,10
- ./examples/cap_set_input_data.json`

funsearch runasync examples/cap_set_spec.py 8 --sandbox_type ExternalProcessSandbox --model_name codestral-latest --samplers 20 --num_islands 10  --run_duration 3000

This implementation limits the number of evaluators to the number of cores minus 1, or the number of evaluators specified in the config file, whichever is smaller.
We choose the number of islands via --num_islands. 10 is typically a good default.
Any parameters not listed here can be modified in funsearch/config.py. Particular parameters of interest are the LLM top_p and temperature (as distinct from the temperature used for sampling from the islands). We set these to the same values as funsearch, 0.95 and 1 respectively. This probably requires some tuning for different problems/LLM models.


Here are the rest of the run params:

- `spec_file`: A file containing the specification for the problem to be solved. This includes the base prompt for the LLM and the evaluation metric.
- `inputs`: The input data for the problem. This can be a filename ending in .json or .pickle, or comma-separated values.
- `--model_name`: The name of the language model to use. Default is "gpt-3.5-turbo-instruct".
- `--output_path`: The directory where logs and data will be stored. Default is "./data/".
- `--load_backup`: Path to a backup file of a previous program database to continue from a previous run.
- `--iterations`: The maximum number of iterations per sampler. Default is -1 (unlimited).
- `--samplers`: The number of sampler threads to run. Default is 15.
- `--sandbox_type`: The type of sandbox to use for code execution. Default is "ContainerSandbox".
- `--num_islands`: The number of islands for the island model in the genetic algorithm. Default is 10.
- `--run_duration`: The duration in seconds for which the search should run. Default is 3600 (1 hour).
- `--num_evaluators`: The number of evaluator processes to run. Default is the number of CPU cores minus 1.
- `--lm`: The language model configuration to use. This can be specified multiple times for different samplers.
- `--function_to_evolve`: The name of the function to evolve in the spec file. Default is "priority".
- `--function_to_run`: The name of the function to run for evaluation in the spec file. Default is "solve".
You can adjust these parameters to customize your FunSearch run. For example:


```
Here, we are searching for the algorithm to find maximum cap sets for dimension 11.
You should see something like:
```
root@11c22cd7aeac:/workspace# funsearch run examples/cap_set_spec.py 11 --sandbox_type ExternalProcessSandbox --model_name codestral-latest
INFO:root:Writing logs to data/1704956206
INFO:absl:Best score of island 0 increased to 2048
INFO:absl:Best score of island 1 increased to 2048
INFO:absl:Best score of island 2 increased to 2048
INFO:absl:Best score of island 3 increased to 2048
INFO:absl:Best score of island 4 increased to 2048
INFO:absl:Best score of island 5 increased to 2048
INFO:absl:Best score of island 6 increased to 2048
INFO:absl:Best score of island 7 increased to 2048
INFO:absl:Best score of island 8 increased to 2048
INFO:absl:Best score of island 9 increased to 2048
INFO:absl:Best score of island 5 increased to 2053
INFO:absl:Best score of island 1 increased to 2049
INFO:absl:Best score of island 8 increased to 2684
^C^CINFO:root:Keyboard interrupt. Stopping.
INFO:absl:Saving backup to data/backups/program_db_priority_1704956206_0.pickle.
```

You may also see `INFO:httpx:HTTP Request: POST https://api.mistral.ai/v1/chat/completions "HTTP/1.1 200 OK"` for each successful API call.

Note that in the last command, we use the ExternalProcessSandbox. This is not fully 'safe', but does make it a bit less likely that invalid code from LLM could break things. The default is ContainerSandbox. However, as we are running the entire thing inside a Docker container, this is not strictly necessary.

Alternatively, you can run the main Python process on a host computer outside of any container and let
the process build and run separate sandbox containers (still requires Docker(/Podman)).
This variant could be also used, e.g., in Colab quite safely since the environment is some kind of container itself.

```
pip install .

funsearch run examples/cap_set_spec.py 11
```

You can monitor the progress of the search using tensorboard: in a separate terminal, run `tensorboard --logdir ./data/tensorboard_logs/ --host 0.0.0.0` to see score graphs for each island. You will need to access the tensorboard server via `http://localhost:6006` on your host machine. Run this command from outside the docker! You may need to install tensorboard first: `pip install tensorboard`.

Once a run is complete - or after interrupting using Ctrl-C, we can analyse our results using the backups file:

```
INFO:absl:Saving backup to data/backups/program_db_priority_1727991117_0.pickle.

>>> funsearch ls data/backups/program_db_priority_1727991117_0.pickle
```

Additionally, the scores are logged to a csv file in `./data/scores/`, and at the end of the `runasync` a graph of the best scores per island over time is generated in `./data/graphs/` (at this point, matplotlib and pandas are installed - this is just to improve docker compilation time). This can be generated by hand using `funsearch makegraphs <timestamp>`.



---

Adding additional programs:
To add additional programs, add .py files to the examples/ directory. These should follow the same structure as the other examples - a priority function with an @funsearch.evolve decorator, and an evaluation function which returns a score decorated with @funsearch.run. See `examples/cap_set_spec.py` for a simple example, and see `examples/Example_Implementation_SPEC.py` for template which you can fill in. I have also been writing playground `*.ipynb` files in the `examples/playgrounds/` directory, where new environments can be developed and tested.

This repository accompanies the publication

> Romera-Paredes, B. et al. [Mathematical discoveries from program search with large language models](https://www.nature.com/articles/s41586-023-06924-6). *Nature* (2023)

There are 6 independent directories:

- `cap_set` contains functions discovered by FunSearch that construct large cap
sets, and we also provide those cap sets in a numerical format for convenience.

- `admissible_set` contains functions discovered by FunSearch that construct
large admissible sets, and we also provide those admissible sets in a numerical
format for convenience.

- `bin_packing` contains heuristics discovered by FunSearch for online 1D bin
packing problems, and an evaluation suite to reproduce the results reported in
the paper.

- `cyclic_graphs` contains functions discovered by FunSearch that construct
large independent sets in strong products of cyclic graphs, and we also provide
those sets in a numerical format for convenience.

- `corner_free_sets` contains the discovered sets of indices, in numerical
format, satisfying the combinatorial degeneration constraints described for the
corners-free problem in the Supplementary Information.

- `implementation` contains an implementation of the evolutionary algorithm,
code manipulation routines, and a single-threaded implementation of the
FunSearch pipeline. It does not contain language models for generating new
programs, the sandbox for executing untrusted code, nor the infrastructure for
running FunSearch on our distributed system. This directory is intended to be
useful for understanding the details of our method, and for adapting it for use
with any available language models, sandboxes, and distributed systems.

## Installation

No installation is required. All notebooks can be opened and run in Google
Colab.

## Usage

- `cap_set`: The notebook `cap_set.ipynb` can be opened via
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/funsearch/blob/master/cap_set/cap_set.ipynb).

- `admissible_set`: The notebook `admissible_set.ipynb` can be opened
via
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/funsearch/blob/master/admissible_set/admissible_set.ipynb).

- `bin_packing`: The notebook `bin_packing.ipynb` can be opened via
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/funsearch/blob/master/bin_packing/bin_packing.ipynb).

- `cyclic_graphs`: The notebook `cyclic_graphs.ipynb` can be opened via
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/funsearch/blob/master/cyclic_graphs/cyclic_graphs.ipynb).

## Citing this work

If you use the code or data in this package, please cite:

```bibtex
@Article{FunSearch2023,
  author  = {Romera-Paredes, Bernardino and Barekatain, Mohammadamin and Novikov, Alexander and Balog, Matej and Kumar, M. Pawan and Dupont, Emilien and Ruiz, Francisco J. R. and Ellenberg, Jordan and Wang, Pengming and Fawzi, Omar and Kohli, Pushmeet and Fawzi, Alhussein},
  journal = {Nature},
  title   = {Mathematical discoveries from program search with large language models},
  year    = {2023},
  doi     = {10.1038/s41586-023-06924-6}
}
```

## License and disclaimer

Copyright 2023 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
