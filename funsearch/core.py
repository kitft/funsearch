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

"""A single-threaded implementation of the FunSearch pipeline."""
import logging

from funsearch import code_manipulation


def _extract_function_names(specification: str) -> tuple[str, str]:
  """Returns the name of the function to evolve and of the function to run."""
  run_functions = list(
      code_manipulation.yield_decorated(specification, 'funsearch', 'run'))
  if len(run_functions) != 1:
    raise ValueError('Expected 1 function decorated with `@funsearch.run`.')
  evolve_functions = list(
      code_manipulation.yield_decorated(specification, 'funsearch', 'evolve'))
  if len(evolve_functions) != 1:
    raise ValueError('Expected 1 function decorated with `@funsearch.evolve`.')
  return evolve_functions[0], run_functions[0]


def run(samplers, database, iterations: int = -1):
  """Launches a FunSearch experiment."""

  try:
    # This loop can be executed in parallel on remote sampler machines. As each
    # sampler enters an infinite loop, without parallelization only the first
    # sampler will do any work. 
    # This is the original comment on the code - I'm not sure it's correct. Depending on
    # implementation, s.sample() should not give an infinite loop - not as the code is 
    # currently written. In any case, it needs to be significantly modified for parallelisation.
    while iterations != 0:
      for s in samplers:
        s.sample()
      #if iterations > 0:
      iterations -= 1
      if iterations % 5 == 0:
        #print("best scores unique: ", list(set(database._best_scores_per_test_per_island)))
        L = database._best_scores_per_test_per_island
        print("best scores unique: ", [dict(sorted(dict(s).items(), key=lambda x: x[0])) for s in set(frozenset(d.items()) for d in L)])
  except KeyboardInterrupt:
    logging.info("Keyboard interrupt. Stopping.")
  database.backup()



