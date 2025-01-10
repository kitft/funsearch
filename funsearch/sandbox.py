import logging

import ast
import os
import pathlib
import sys
from typing import Any

import cloudpickle

CONTAINER_MAIN = (pathlib.Path(__file__).parent / "container" / "container_main.py").absolute()

IMAGE_NAME = "funsearch_sandbox"


class DummySandbox:
  """Base class for Sandboxes that execute the generated code.

  Note: this base class executes the code but does not offer any sandboxing!!!
  It should be only used in unit testing or debugging, and not with real LLM
  unless the host environment is in some kind of sandbox itself.
  Even in sandboxed host, the executed code could theoretically affect later executions.
  """

  sandboxes = 0

  def __init__(self, **kwargs):
    self.id = DummySandbox.sandboxes

    DummySandbox.sandboxes += 1

  def run(
          self,
          program: str,
          function_to_run: str,
          test_input,
          timeout_seconds: int,
  ) -> tuple[Any, bool]:
    """Returns `function_to_run(test_input)` and whether execution succeeded."""

    # The same "program" seems to be now repeatedly parsed using AST and then compiled.
    # This could probably be simplified quite a bit.
    namespace = DummySandbox.compile_code(program)
    return namespace[function_to_run](test_input)

  @staticmethod
  def compile_code(program: str):
    namespace = {}

    parsed_code = ast.parse(program)
    compiled_code = compile(parsed_code, filename="<ast>", mode="exec")
    exec(compiled_code, namespace)
    return namespace


class ExternalProcessSandbox(DummySandbox):
  """Sandbox that executes the code in a separate Python process in the same host.

  Note: This does not provide real safety and should be only used in an environment where the host process is
  in some kind of safe sandbox itself (i.e., a container).
  This kind of sandbox merely makes it more probable that single invalid call does not break the whole
  funsearch algorithm. It might be easier to set up and thus nice environment to tune the prompts and other code.
  """

  def __init__(self, base_path: pathlib.Path, timeout_secs: int = 30, python_path: str = "python", id: int = 0):
    super(ExternalProcessSandbox, self).__init__()

    self.id = id
    self.output_path = pathlib.Path(base_path) / f"sandbox{self.id}"
    self.timeout_secs = timeout_secs
    self.python_path = python_path
    self.call_count = 0

    self.input_path = self.output_path / "inputs"
    for p in [self.output_path, self.input_path]:
      if not p.exists():
        p.mkdir(parents=True)

  def _exec(self, call_data_path: pathlib.Path, input_path: pathlib.Path, error_file_path: pathlib.Path):
    """Use podman/docker to execute python in a container.
    - The main.py shall execute the LLM generated method from prog.pickle file providing
      input.pickle as the input for the method.
    - main.py writes the output of the method into output.pickle.
    Everything except the /workspace folder will be read-only so that the environment remains good
    for future runs.
    """
    prog_path = call_data_path / "prog.pickle"
    output_file = call_data_path / "output.pickle"
    cmd = (f"{self.python_path} {CONTAINER_MAIN} {prog_path} {input_path} {output_file}"
           f"  2> {error_file_path}")
    logging.debug(f"Executing: {cmd}")
    return os.system(cmd)

  def run(
          self,
          program: str,
          function_to_run: str,
          test_input,
          timeout_seconds: int,
  ) -> tuple[Any, bool]:

    call_data_folder = (self.output_path / f"call{self.call_count}").absolute()
    if not call_data_folder.exists():
      call_data_folder.mkdir()

    input_hash = hash(test_input)
    input_path = (self.input_path / f"{input_hash}.pickle").absolute()

    if not input_path.exists():
      with open(input_path, "wb") as f:
        cloudpickle.dump(test_input, f)
    try:
      namespace = DummySandbox.compile_code(program)

      prog_file = (call_data_folder / "prog.pickle").absolute()
      with open(prog_file, "wb+") as f:
        cloudpickle.dump(namespace[function_to_run], f)

      error_file = self.output_path / f"stderr_{self.call_count}.log"

      retcode = self._exec(call_data_folder, input_path, error_file)
      self.call_count += 1

      if retcode != 0:
        self._save_diagnostics(program, call_data_folder)
        with open(error_file, "r") as f:
          outerr = f.read()
        return outerr, False

      output_file = call_data_folder / "output.pickle"
      with open(output_file, "rb") as f:
        out = cloudpickle.load(f)
        return out, True
    except Exception as e:
      logging.debug(f"Could not execute code: {e}")
    self._save_diagnostics(program, call_data_folder)
    return None, False

  @staticmethod
  def _save_diagnostics(program: str, output_path: pathlib.Path):
    filepath = output_path / "program.py"
    logging.debug(f"Writing program to {filepath}")
    with open(filepath, "w+") as f:
      f.write(program)

class ContainerSandbox(ExternalProcessSandbox):
  """Basic sandbox that runs unsafe code in Podman or Docker container.
  - the sandbox should be safe against inadvertent bad code by LLM but not against malicious attacks.
  - does not require any other dependencies on the host than Podman/Docker
  - does not support multithreading
  - might provide easier or more lightweight debugging experience than some other fancier sandbox environments
  """
  executable = "podman"
  image_built = False

  @classmethod
  def build_image(cls, extra_pip_packages):
    version = sys.version.split(" ")[0]
    ret = os.system("podman --version")
    if ret != 0:
      ret = os.system("docker --version")
      if ret != 0:
        raise Exception("Could not find Podman or Docker. Can not use ContainerSandbox.")
      else:
        cls.executable = "docker"

    dockerfile = pathlib.Path(__file__).parent / "container" / "Dockerfile"
    logging.debug("Building container image")
    extra = ""
    if extra_pip_packages:
      extra = f"--build-arg INSTALL_PACKAGES=\"{extra_pip_packages}\""

    cmd = (f"{cls.executable} build --build-arg PYTHON_VERSION={version} {extra} "
           f"-t {IMAGE_NAME} -f {dockerfile} {CONTAINER_MAIN.parent}")
    logging.debug(f"Executing: {cmd}")
    os.system(cmd)
    cls.image_built = True

  def __init__(self, base_path: pathlib.Path, extra_pip_packages: str = "numpy", timeout_secs=30, id: int = 0):
    super(ContainerSandbox, self).__init__(base_path, timeout_secs)

    if not ContainerSandbox.image_built:
      ContainerSandbox.build_image(extra_pip_packages)

  def _exec(self, call_data_path: pathlib.Path, input_path: pathlib.Path, error_file_path: pathlib.Path):
    """Use podman/docker to execute python in a container.
    - The main.py shall execute the LLM generated method from prog.pickle file providing
      input.pickle as the input for the method.
    - main.py writes the output of the method into output_{id}.pickle.
    Everything except the /workspace folder will be read-only so that the environment remains good
    for future runs.
    """
    output_file = call_data_path / f"output_{self.id}.pickle"
    cmd = (f"{self.executable} run "
           f"--stop-timeout={self.timeout_secs} "
           f"-v {CONTAINER_MAIN}:/main.py:ro "
           f"-v {call_data_path}:/workspace "
           f"-v {input_path}:/input.pickle:ro "
           f"{IMAGE_NAME}:latest /usr/local/bin/python3 "
           f"/main.py /workspace/prog.pickle /input.pickle {output_file}"
           f" 2> {error_file_path}")
    logging.debug(f"Executing: {cmd}")
    return os.system(cmd)

