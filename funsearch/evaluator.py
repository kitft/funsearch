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

"""Class for evaluating programs proposed by the Sampler."""
import ast
import re
import inspect
from collections.abc import Sequence
import copy
from typing import Any, Tuple
import logging

from funsearch import code_manipulation
from funsearch import programs_database
from funsearch import sandbox
from funsearch import logging_stats
"""
  Regex to find all methods named 'priority_vX'.
  With each match, start from the 'def priority_vX(' and continue until there's a new line with any of
  - a new 'def'
  - ` or ' or # without indentation
"""
#METHOD_MATCHER = re.compile(r"def priority_v\d\(.*?\) -> float:(?:\s*(?:[ \t]*(?!def|#|`|').*(?:\n|$)))+")
# DDED ANY TYPE SIGNATURE OUTPUT, NOT JUST FLOAT
METHOD_MATCHER = re.compile(r"def priority_v\d\(.*?\)(?:\s*->.*?)?:(?:\s*(?:[ \t]*(?!def|#|`|').*(?:\n|$)))+")
METHOD_NAME_MATCHER = re.compile(r"priority_v\d+")

ALLOWED_FUNCTIONS = {'itertools', 'numpy', 'np', 'math', 'functools'}
DISALLOWED_BUILTINS = {'print','__import__','breakpoint','compile','open','dir','eval','exec','globals','input','repr'}

class FunctionChecker(ast.NodeVisitor):
    def __init__(self):
        self.is_safe = True
        self.vars = []

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name not in ALLOWED_FUNCTIONS:
                self.is_safe = False
            self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module not in ALLOWED_FUNCTIONS:
            self.is_safe = False
        self.generic_visit(node)
    
    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.vars.append(target.id)
        self.generic_visit(node)

    def visit_Call(self, node):
        # Check for disallowed built-in function calls
        if isinstance(node.func, ast.Name):
            if node.func.id in DISALLOWED_BUILTINS:
                self.is_safe = False
        # Check if function calls are from allowed modules
        elif isinstance(node.func, ast.Attribute):
            # if node.func.value.id not in ALLOWED_FUNCTIONS:
            #     self.is_safe = False
            func_value = ast.unparse(node.func.value)
            if func_value.split('.')[0] not in ALLOWED_FUNCTIONS and func_value.split('.')[0] not in self.vars:
                self.is_safe = False
        self.generic_visit(node)

def is_function_safe(func:code_manipulation.Function)->bool:
    #function_code = inspect.getsource(func)
    function_code = func.__str__()
    tree = ast.parse(function_code)
    checker = FunctionChecker()
    checker.visit(tree)
    return checker.is_safe


class _FunctionLineVisitor(ast.NodeVisitor):
  """Visitor that finds the last line number of a function with a given name."""

  def __init__(self, target_function_name: str) -> None:
    self._target_function_name: str = target_function_name
    self._function_end_line: int | None = None

  def visit_FunctionDef(self, node: Any) -> None:  # pylint: disable=invalid-name
    """Collects the end line number of the target function."""
    if node.name == self._target_function_name:
      self._function_end_line = node.end_lineno
    self.generic_visit(node)

  @property
  def function_end_line(self) -> int:
    """Line number of the final line of function `target_function_name`."""
    assert self._function_end_line is not None  # Check internal correctness.
    return self._function_end_line


def _find_method_implementation(generated_code: str) -> Tuple[str, str]:
  """Find the last 'def priority_vX()' method from generated code.

  Return the code and the name of the method.
  """
  matches = METHOD_MATCHER.findall(generated_code)
  if not matches:
    return "", ""
  last_match = matches[-1]
  name = METHOD_NAME_MATCHER.search(last_match).group()
  return last_match, name


def _trim_function_body(generated_code: str) -> str:
  """Extracts the body of the generated function, trimming anything after it. Includes docstring."""
  if not generated_code:
    return ''
  if not type(generated_code) is str:
    generated_code = str(generated_code)

  # 3 lines below added by AVS on 10/23/2024 to ignore lines that are not indented (or defining a function)
  # This helps deal with some of the cases where the LLM is adding commentary that is not valid code
  lines = generated_code.split('\n')
  lines = [s for s in lines if (len(s) > 1 and s[:2] in ["de", "  "])]
  generated_code = '\n'.join(lines)

  method_name = "fake_function_header"
  # Check is the response only a continuation for our prompt or full method implementation with header
  if "def priority_v" in generated_code:
    code, method_name = _find_method_implementation(generated_code)
  else:
    code = f'def {method_name}():\n{generated_code}' #this is temporary and doesn't matter that it's incorrect typing?

  # Finally parse the code to make sure it's valid Python
  tree = None
  # We keep trying and deleting code from the end until the parser succeeds.
  while tree is None:
    try:
      tree = ast.parse(code)
    except SyntaxError as e:
      code = '\n'.join(code.splitlines()[:e.lineno - 1])
  if not code:
    # Nothing could be saved from `generated_code`
    return ''

  visitor = _FunctionLineVisitor(method_name)
  visitor.visit(tree)
  body_lines = code.splitlines()[1:visitor.function_end_line]
  return '\n'.join(body_lines) + '\n\n'


def _sample_to_program(
    generated_code: str,
    version_generated: int | None,
    template: code_manipulation.Program,
    function_to_evolve: str,
) -> tuple[code_manipulation.Function, str]:
  """Returns the compiled generated function and the full runnable program."""
  doc_and_body = _trim_function_body(generated_code)
  if not doc_and_body:
    return None, None
  if version_generated is not None:
    doc_and_body = code_manipulation.rename_function_calls(
        doc_and_body,
        f'{function_to_evolve}_v{version_generated}',
        function_to_evolve)

  program = copy.deepcopy(template)
  evolved_function = program.get_function(function_to_evolve)
  evolved_function.body = doc_and_body
  return evolved_function, str(program)



def _calls_ancestor(program: str, function_to_evolve: str) -> bool:
  """Returns whether the generated function is calling an earlier version."""
  for name in code_manipulation.get_functions_called(program):
    # In `program` passed into this function the most recently generated
    # function has already been renamed to `function_to_evolve` (wihout the
    # suffix). Therefore any function call starting with `function_to_evolve_v`
    # is a call to an ancestor function.
    if name.startswith(f'{function_to_evolve}_v'):
      return True
  return False


class Evaluator:
  """Class that analyses functions generated by LLMs."""

  def __init__(
      self,
      database: programs_database.ProgramsDatabase or multi_testing.AsyncProgramsDatabase, #undefined name 'multi_testing'
      sbox: sandbox.DummySandbox,
      template: code_manipulation.Program,
      function_to_evolve: str,
      function_to_run: str,
      inputs: Sequence[Any],
      timeout_seconds: int = 30,
      id: int = 0,
      log_path: str = None,
  ):
    self.log_path = log_path
    self._database = database
    self._template = template
    self._function_to_evolve = function_to_evolve
    self._function_to_run = function_to_run
    self._inputs = inputs
    self._timeout_seconds = timeout_seconds
    self._sandbox = sbox
    self._id = id
    #self._sandbox.id = id
    #print("eval id: ", self._id)
    #print("sandbox id: ", self._sandbox.id)
    self.usage_logger = logging_stats.UsageLogger(self._id,log_dir = self.log_path)
  def analyse(
      self,
      sample: str,
      usage_stats: logging_stats.UsageStats | None,
  ) -> tuple[Any, int | None, dict, int | None, str | None] | None:
    """Compiles the sample into a program and executes it on test inputs.
    
    Returns:
        A tuple of (function, scores, usage_stats) if successful,
        or None if compilation/execution failed.
        The scores dict contains evaluation results for each test input.
    """
    island_id = usage_stats.island_id
    version_generated = usage_stats.version_generated
    island_version = usage_stats.island_version
    model = usage_stats.model

    new_function, program = _sample_to_program(
        sample, version_generated, self._template, self._function_to_evolve)
    if new_function is None:
      usage_stats.eval_state = 'parse_failed'
      usage_stats.sandbox_current_call_count = -1
      logging.info(f"eval:parse-failed: {model}, island_id: {island_id}, file:s{usage_stats.sampler_id}p{usage_stats.prompt_count}e{self._id}c{usage_stats.sandbox_current_call_count} version_gen: {version_generated}, island_version: {island_version}")
      self._log(usage_stats)
      return (None, {}, usage_stats)
    
    safe = is_function_safe(new_function)
    if not safe:
      usage_stats.eval_state = 'unsafe'
      usage_stats.sandbox_current_call_count = -1
      logging.info(f"eval:unsafe: {model}, island_id: {island_id}, file:s{usage_stats.sampler_id}p{usage_stats.prompt_count}e{self._id}c{usage_stats.sandbox_current_call_count} version_gen: {version_generated}, island_v: {island_version}")
      self._log(usage_stats)
      return (None, {}, usage_stats)

    scores_per_test = {}
    for current_input in self._inputs:
      usage_stats.sandbox_current_call_count = self._sandbox.call_count 

      test_output, runs_ok = self._sandbox.run(
          program, self._function_to_run, current_input, self._timeout_seconds)
      if (runs_ok and not _calls_ancestor(program, self._function_to_evolve)
          and test_output is not None):
        if not isinstance(test_output, (int, float)):
          raise ValueError('@function.run did not return an int/float score.')
        scores_per_test[current_input] = test_output
      elif runs_ok:
        pass
      elif not runs_ok:
        usage_stats.std_err = test_output

    if scores_per_test:
      #print("Putting in queue inside evaluator")
      #self._database.register_program(new_function, island_id, scores_per_test)
      #db_queue.put((new_function, island_id, scores_per_test))
      logging.debug(f"eval:success {model} {scores_per_test}")
      usage_stats.eval_state = 'success'
      usage_stats.scores_per_test = scores_per_test
      self._log(usage_stats)
      return (new_function, scores_per_test,  usage_stats)
    else:
      logging.info(f"eval:didnotrun: {model}, island_id: {island_id}, file:s{usage_stats.sampler_id}p{usage_stats.prompt_count}e{self._id}c{usage_stats.sandbox_current_call_count} version_gen: {version_generated}, island_version: {island_version}")
      usage_stats.eval_state = 'did_not_run'
      self._log(usage_stats)
      return (None, {}, usage_stats)

  def _log(self,usage_stats):
    if usage_stats.id is None:
      #this is the initial evaluation
      return
    prompt = usage_stats.prompt
    response = usage_stats.response
    model = usage_stats.model
    sampler_id = usage_stats.sampler_id
    prompt_count = usage_stats.prompt_count
    sandbox_current_call_count = usage_stats.sandbox_current_call_count

    model_name_replaced = model.replace("/", "_")
    name_for_log = f"{model_name_replaced}_s{sampler_id}_p{prompt_count}_e{self._id}_c{sandbox_current_call_count}.log"

    if self.log_path is not None:
      with open(self.log_path / name_for_log, "a") as f:
        f.write("=== PROMPT ===\n")
        f.write(prompt)
        f.write("\n=== RESPONSE ===\n")
        f.write(str(response))
        f.write("\n================\n")
        f.write(f"eval_state: {usage_stats.eval_state}\n")
        f.write(f"Model: {model}\n")
        f.write(f"Total tokens: {usage_stats.total_tokens}\n")
        f.write(f"Prompt tokens: {usage_stats.tokens_prompt}\n")
        f.write(f"Completion tokens: {usage_stats.tokens_completion}\n")
        f.write(f"generation_time: {usage_stats.generation_time}\n")
        f.write(f"scores_per_test: {usage_stats.scores_per_test}\n")
        if usage_stats.std_err is not None:
          f.write(f"std_err: {usage_stats.std_err}\n")
    self.usage_logger.log_usage(usage_stats)
    
  # async def analyse_sync(
  #     self,
  #     sample: str,
  #     island_id: int | None,
  #     version_generated: int | None,
  # ) -> dict:
  #   """Compiles the sample into a program and executes it on test inputs."""
  #   new_function, program = _sample_to_program(
  #       sample, version_generated, self._template, self._function_to_evolve)

  #   scores_per_test = {}
  #   for current_input in self._inputs:
  #     test_output, runs_ok = self._sandbox.run(
  #         program, self._function_to_run, current_input, self._timeout_seconds)
  #     if (runs_ok and not _calls_ancestor(program, self._function_to_evolve)
  #         and test_output is not None):
  #       if not isinstance(test_output, (int, float)):
  #         raise ValueError('@function.run did not return an int/float score.')
  #       scores_per_test[current_input] = test_output
  #   if scores_per_test:
  #     await self._database.register_program(new_function, island_id, scores_per_test)
  #     #db_queue.put((new_function, island_id, scores_per_test))

