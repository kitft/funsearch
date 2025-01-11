
# Funsearch

A high-performance implementation of program search using LLMs, built for discovering mathematical algorithms and solving optimization problems. We essentially run a genetic algorithm with LLMs as the mutation/breeding operator.

This version adds parallel processing, updated tooling, and many more features to the original DeepMind implementation.

## Features

### Modified to support non-priority-function program search
- Search for any type signature, not just priority functions

### Integration with Weights & Biases
- Real-time monitoring and logging
- Automatic score logging and graphing

### Multi-Model Support
- Use any combination of modern LLMs including Claude/GPT-4o/Gemini/Mistral/Deepseek/etc through direct APIs or OpenRouter
- Use multiple models in the same run:  multiple models with different temperatures, etc
- Adaptive sampling rates based on LLM/CPU/evaluator capacity

### Parallel Processing
- Runs multiple program evaluations simultaneously across CPU cores
- Makes asynchronous API calls to sampler/evaluator/database agents

### Safety Features
- Sandboxed code execution (container or process-based) *in progress* - see discussion below
- Model validation before search starts/graceful handling of API failures

For implementation details and example problems, see the examples directory, particularly the section on *Adding additional programs*.





## Installation and Setup

### Using Docker (Recommended)

1. Install Docker. Make sure the docker service is running.
   
2. Clone the repository and navigate to the project directory

`git clone https://github.com/kitft/funsearch.git`
`cd funsearch`
   
3. Set your API keys by creating a `.env` file in the project root (this file will be automatically gitignored):
   ```
   # Create a .env file in the project root
   touch .env
   
   # Add your API keys to the .env file:
   MISTRAL_API_KEY=<your_key_here>
   GOOGLE_API_KEY=<your_key_here>
   OPENAI_API_KEY=<your_key_here>
   ANTHROPIC_API_KEY=<your_key_here>
   OPENROUTER_API_KEY=<your_key_here>
   WANDB_API_KEY=<your_wandb_key_here>
   DEEPINFRA_API_KEY=<your_deepinfra_key_here>
   ```
4. If you're using Docker desktop, open the app. Then, build the Docker image with the command below. This may take a while, especially the first time. You will need to re-build if you update the Dockerfile or edit the code. You should not need to re-build if you are only updating the input python file or input data.

   ```
   docker build . -t funsearch
   ```
5. Create a data folder and run the container:
   ```
   mkdir data
   docker run -it -v ./data:/workspace/data -v ./examples:/workspace/examples --env-file .env funsearch
   ```

## Model Selection
- Any model name containing a slash (/) will automatically use the OpenRouter API. For example:
  - 'anthropic/claude-3-opus'
  - 'google/gemini-pro'
  - 'meta-llama/llama-2-70b-chat'
  - 'mistralai/mixtral-8x7b'
- You can find the full list of available models at https://openrouter.ai/docs
- Multiple models can be specified using comma separation with optional counts and key numbers:
  - 'model1*count1*key1,model2*count2*key2'
  - Example: 'codestral-latest*10*0,mistralai/codestral_mamba*5*1' will use 10 codestrals from mistral, and 5 mistral-tinys from openrouter.
- Note the key numbers are used to differentiate between different API keys for the same model. These can be passed as separate environment variables with integers on the end. The separate ones will only be used if the key number is greater than 0.
- You can also specify a model from OpenAI or Anthropic, etc. If this doesn't work, have a look at models.py
- Setting model temperatures: pass a list of temperatures, one for each model. Format: "temperature1,temperature2,..."
- Before starting the search, each model is validated to ensure it's working correctly:

## Advice
- Run the search with the desired model using the '--model' attribute.
- A good one to use for testing, as an extremely cheap model, is 'mistral-tiny-latest'. 
- The best one for our use case is probably 'codestral-latest', which is 4x more expensive per output token.
- Another great one might be deepseek/deepseek-chat, which is competitive with claude-3-5-sonnet and much cheaper.
- Either model is still relatively cheap: with codestral, 1 million output tokens is $0.6.
- You can also pass in a model name from OpenAI or Anthropic, etc. If this doesn't work, have a look at models.py

## Running the search
`funsearch runasync` takes two arguments and several options:

1. `SPEC_FILE`: A Python module that provides the basis of the LLM prompt as well as the evaluation metric.
   - Example: See `examples/cap_set_spec.py`

2. `INPUTS`: Can be one of the following:
   - A filename ending in .json or .pickle
   - Comma-separated input data
   - The files are expected to contain a list with at least one element
   - Elements will be passed to the `solve()` method one by one

Examples of valid INPUTS:
- 8
- 8,9,10
- ./examples/cap_set_input_data.json

Key options:
- `--model`: The LLM model to use (default: "codestral-latest","mistralai/codestral-mamba")
- `--samplers`: Number of sampler threads (default: 15)
- `--evaluators`: Number of evaluator processes (default: CPU cores - 1)
- `--islands`: Number of islands for genetic algorithm (default: 10)
- `--duration`: Run duration in seconds (default: 3600)
- `--team`: Wandb team/entity for logging (optional)
- `--name`: Wandb ID for run

Example command:
```bash
funsearch runasync /workspace/examples/cap_set_spec.py 8 --sandbox ExternalProcessSandbox --model mistralai/codestral-mamba --samplers 20 --islands 10 --duration 3000 --team <team>
```

### Weights & Biases Integration

The search progress is automatically logged to Weights & Biases (wandb). You can specify a team/entity for logging in two ways:

1. Using the `--team` option:
   - If provided, will use that team as default with a 10-second countdown
   - During countdown, you can press Enter to enter a different entity
   - After countdown, automatically proceeds with the specified team

2. Without `--team`:
   - Prompts for entity name
   - Press Enter to use wandb default (no entity)

To use wandb:
1. Set your wandb API key:
   ```bash
   export WANDB_API_KEY=<your_wandb_key_here>
   ```
2. Run funsearch with optional team:
   ```bash
   funsearch runasync ... --team myteam
   ```

The following metrics are logged to wandb:
- Best and average scores for each island
- Overall best and average scores
- Queue sizes
- API call counts
- Other relevant metrics

Additionally, scores are logged to CSV files in `./data/scores/`, and graphs are generated in `./data/graphs/` showing the progression of scores over time.

The number of evaluator processes is limited to the number of CPU cores minus 1, or the number specified in --evaluators, whichever is smaller.

The number of islands is controlled via --islands. 10 is typically a good default as it provides a good balance between exploration and exploitation.

Any parameters not listed here can be modified in funsearch/config.py. Particular parameters of interest are the LLM top_p and temperature (as distinct from the temperature used for sampling from the islands). We set these to the same values as the original funsearch paper: 0.95 and 1 respectively. This probably requires some tuning for different problems/LLM models.

The search will automatically stop if:
- The eval queue size exceeds 500 items
- The result queue size exceeds 50 items 
- The specified duration is reached
- A keyboard interrupt is received

Here are the available run parameters:

- `spec_file`: A file containing the specification for the problem to be solved. This includes the base prompt for the LLM and the evaluation metric.
- `inputs`: The input data for the problem. This can be a filename ending in .json or .pickle, or comma-separated values.
- `--model`: The name of the language model (or models) to use. Default is "codestral-latest", which uses the Mistral api. Format: "model1*count1*key1,model2*count2*key2",...etc
- `--output_path`: The directory where logs and data will be stored. Default is "./data/".
- `--load_backup`: Path to a backup file of a previous program database to continue from a previous run: e.g. "./data/backups/program_db_priority_identifier_0.pickle"
- `--iterations`: The maximum number of iterations per sampler. Default is -1 (unlimited).
- `--sandbox`: The type of sandbox to use for code execution. Default is "ContainerSandbox".
- `--samplers`: The number of sampler threads to run. Default is 15.
- `--evaluators`: The number of evaluator processes to run. Default is the number of CPU cores minus 1.
- `--islands`: The number of islands for the island model in the genetic algorithm. Default is 10.
- `--reset`: The time between island resets in seconds. Default is 600 (10 minutes).
- `--duration`: The duration in seconds for which the search should run. Default is 3600 (1 hour).
- `--temperature`: LLM temperature. Default is 1.0. Can also give a list of temperatures - one for each entry in --model. Format: "temperature1,temperature2,..."
- `--team`: Wandb team/entity for logging (optional)
- `--envfile`: Path to a .env file to load environment variables from. This is only useful if you are running the search outside of a container.


## Example
Here, we are searching for the algorithm to find maximum cap sets for dimension 11.
You should see something like:
```
root@11c22cd7aeac:/workspace# funsearch runasync /workspace/examples/cap_set_spec.py 11 --sandbox ExternalProcessSandbox --model codestral-latest
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
source .env #need to source the .env file to get the API keys
funsearch runasync examples/cap_set_spec.py 11
```

# PROGRAM DATABASE BACKUPS


The program database is automatically backed up every 500 programs, with 5 rotating backups labelled 0-4, where the most recent backup is not necessarily labelled 0. These are stored in the `./data/backups/` directory. You can resume from a backup with the `--load_backup` option. 


To inspect a backup of a program database, you can use the `funsearch ls./data/backups/program_db_priority_<identifier>/program_db_priority_<identifier>_<backup_number>.pickle` command.

# LOGS AND GRAPHS

You can monitor the progress of the search using Weights & Biases (wandb). First, make sure you have set your wandb API key (in the .env file), or otherwise set the WANDB_API_KEY environment variable inside the container.

```
export WANDB_API_KEY=<your_wandb_key_here>
```

The search progress will be automatically logged to your wandb project "funsearch". You can view the results in real-time by:
1. Going to https://wandb.ai
2. Logging in with your account
3. Opening the "funsearch" project
4. Selecting your run

Each run will be named with a timestamp (e.g., "run_<model_name>_1704956206") and will track:
- Best and average scores for each island
- Overall best and average scores
- Queue sizes
- API call counts
- Other relevant metrics

Additionally, the scores are logged to a csv file in `./data/scores/`, and at the end of the `runasync` a graph of the best scores per island over time is generated in `./data/graphs/` (at this point, matplotlib and pandas are installed - this is just to improve docker compilation time). This can be generated by hand using `funsearch makegraphs <timestamp>`.

To remove/delete all data associated with a timestamp(s), use the following command. This is not reversible, so be careful!
```
funsearch removetimestamp <timestamp1> <timestamp2> ...
```

# ERRORS

PLEASE ENSURE YOU HAVE UPDATED TO THE LATEST VERSION OF THIS REPO

If you are getting OPENAI Async doesn't exist errors, run `pip install openai>=1.2` in your Docker environment. This should happen on Dockerfile creation, but could be a problem if you have some legacy docker/pdm files.




# Adding additional programs:
To add additional programs, add .py files to the examples/ directory. These should follow the same structure as the other examples - a priority function with an @funsearch.evolve decorator, and an evaluation function which returns a score decorated with @funsearch.run. See `examples/cap_set_spec.py` for a simple example, and see `examples/Example_Implementation_SPEC.py` for template which you can fill in. I have also been writing playground `*.ipynb` files in the `examples/playgrounds/` directory, where new environments can be developed and tested.

---
# Original Google DeepMind FunSearch Repository Data


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

Here, we are searching for the algorithm to find maximum cap sets for dimension 11.
You should see something like:
```
root@11c22cd7aeac:/workspace# funsearch runasync /workspace/examples/cap_set_spec.py 11 --sandbox_type ExternalProcessSandbox --model_name codestral-latest
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

funsearch runasync /workspace/examples/cap_set_spec.py 8 --sandbox ExternalProcessSandbox --model mistralai/codestral-mamba --samplers 20 --islands 10 --duration 3000 --team <team>
```

# LOGS AND GRAPHS

You can monitor the progress of the search using Weights & Biases (wandb). First, make sure you have set your wandb API key:
```
export WANDB_API_KEY=<your_wandb_key_here>
```

The search progress will be automatically logged to your wandb project "funsearch". You can view the results in real-time by:
1. Going to https://wandb.ai
2. Logging in with your account
3. Opening the "funsearch" project
4. Selecting your run

Each run will be named with a timestamp (e.g., "run_1704956206") and will track:
- Best and average scores for each island
- Overall best and average scores
- Queue sizes
- API call counts
- Other relevant metrics

Additionally, the scores are logged to a csv file in `./data/scores/`, and at the end of the `runasync` a graph of the best scores per island over time is generated in `./data/graphs/` (at this point, matplotlib and pandas are installed - this is just to improve docker compilation time). This can be generated by hand using `funsearch makegraphs <timestamp>`.

To remove/delete all data associated with a timestamp. This is not reversible, so be careful!
```
funsearch removetimestamp <timestamp>
```

# ERRORS

PLEASE ENSURE YOU HAVE UPDATED TO THE LATEST VERSION OF THIS REPO

If you are getting OPENAI Async doesn't exist errors, run `pip install openai>=1.2` in your Docker environment. This should happen on Dockerfile creation, but could be a problem if you have some legacy docker/pdm files.




# Adding additional programs:
To add additional programs, add .py files to the examples/ directory. These should follow the same structure as the other examples - a priority function with an @funsearch.evolve decorator, and an evaluation function which returns a score decorated with @funsearch.run. See `examples/cap_set_spec.py` for a simple example, and see `examples/Example_Implementation_SPEC.py` for template which you can fill in. I have also been writing playground `*.ipynb` files in the `examples/playgrounds/` directory, where new environments can be developed and tested.

---

forked from https://github.com/google-deepmind/funsearch via https://github.com/jonppe/funsearch 

---
# Original Google DeepMind FunSearch Repository Data


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
