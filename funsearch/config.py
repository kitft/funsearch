# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Configuration of a FunSearch experiment."""
import dataclasses
from funsearch import sandbox

@dataclasses.dataclass(frozen=True)
class ProgramsDatabaseConfig:
  """Configuration of a ProgramsDatabase.

  Attributes:
    functions_per_prompt: Number of previous programs to include in prompts.
    num_islands: Number of islands to maintain as a diversity mechanism.
    reset_period: How often (in seconds) the weakest islands should be reset.
    cluster_sampling_temperature_init: Initial temperature for softmax sampling
        of clusters within an island.
    cluster_sampling_temperature_period: Period of linear decay of the cluster
        sampling temperature.
    backup_period: Number of iterations before backing up the program database on disk
    backup_folder: Path for automatic backups
  """
  functions_per_prompt: int = 2
  num_islands: int = 10  # Default value, can be overridden during initialization
  reset_period: int =  60 * 60 #4*60*60
  cluster_sampling_temperature_init: float = 0.1
  cluster_sampling_temperature_period: int = 30_000
  length_sample_temperature: float = 1.0
  backup_period: int = 30
  backup_folder: str = './data/backups'

  def __init__(self, **kwargs):
    # Use object.__setattr__ to set attributes on a frozen dataclass
    #object.__setattr__(self, 'num_islands', num_islands)
    # Set other attributes from kwargs
    for key, value in kwargs.items():
      if hasattr(self, key):
        object.__setattr__(self, key, value)


@dataclasses.dataclass(frozen=True)
class Config:
  """Configuration of a FunSearch experiment.

  Attributes:
    programs_database: Configuration of the evolutionary algorithm.
    num_samplers: Number of independent Samplers in the experiment. A value
        larger than 1 only has an effect when the samplers are able to execute
        in parallel, e.g. on different matchines of a distributed system.
    num_evaluators: Number of independent program Evaluators in the experiment.
        A value larger than 1 is only expected to be useful when the Evaluators
        can execute in parallel as part of a distributed system.
    samples_per_prompt: How many independently sampled program continuations to
        obtain for each prompt.
    num_islands: Number of islands to maintain as a diversity mechanism.
    llm_temperature: Temperature for the LLM.
  """
  num_islands: int = 10
  programs_database: ProgramsDatabaseConfig = dataclasses.field(
      default_factory=ProgramsDatabaseConfig)
  num_samplers: int = 15
  num_evaluators: int = 10
  samples_per_prompt: int = 4
  num_batches=2
  run_duration: int = 10000000
  top_p: float = 0.95
  llm_temperature: float = 1.0


  def __init__(self, num_islands: int = 10, **kwargs):
    object.__setattr__(self, 'num_islands', num_islands)
    object.__setattr__(self, 'programs_database', ProgramsDatabaseConfig(num_islands=num_islands))
    for key, value in kwargs.items():
      if hasattr(self, key):
        object.__setattr__(self, key, value)

#config for multi-testing - not actually designed to be modified by users
@dataclasses.dataclass(frozen=False)
class MultiTestingConfig:
  """Configuration for multi-testing."""
  log_path: str
  sandbox_class: type[sandbox.DummySandbox]
  parsed_inputs: list
  template: None
  function_to_evolve: None
  function_to_run: None
  lm: None
  timestamp: str


