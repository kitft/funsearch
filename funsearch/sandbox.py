import ast
from typing import Any

import ast
import os
import pathlib
import sys
from typing import Any

import cloudpickle

from funsearch import evaluator

CONTAINER_MAIN = (pathlib.Path(__file__).parent / "container" / "container_main.py").absolute()

IMAGE_NAME = "funsearch_sandbox"


class Sandbox:
  """Base class for Sandboxes that execute the generated code.

  Note: this base class executes the code but does not offer any sandboxing!!!
  It should be only used in unit testing or debugging, and not with real LLM
  unless the host environment is in some kind of sandbox itself.
  Even in sandboxed host, the executed code could theoretically affect later executions.
  """

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
    namespace = Sandbox.compile_code(program)
    return namespace[function_to_run](test_input)

  @staticmethod
  def compile_code(program: str):
    namespace = {}

    parsed_code = ast.parse(program)
    compiled_code = compile(parsed_code, filename="<ast>", mode="exec")
    exec(compiled_code, namespace)
    return namespace


class ContainerSandbox(Sandbox):
  """Basic sandbox that runs unsafe code in Podman or Docker container.
  - the sandbox should be safe against inadvertent bad code by LLM but not against malicious attacks.
  - does not require any other dependencies on the host than Podman/Docker
  - does not support multithreading
  - might provide easier or more lightweight debugging experience than some other fancier sandbox environments
  """
  executable = "podman"
  containers = 0
  image_built = False

  @classmethod
  def build_image(cls, extra_pip_packages: str = ""):
    version = sys.version.split(" ")[0]
    ret = os.system("podman --version")
    if ret != 0:
      ret = os.system("docker --version")
      if ret != 0:
        raise Exception("Could not find Podman or Docker. Can not use ContainerSandbox.")
      else:
        cls.executable = "docker"

    dockerfile = pathlib.Path(__file__).parent / "container" / "Dockerfile"
    print("Building container image")
    extra = ""
    if extra_pip_packages:
      extra = f"--build-arg INSTALL_PACKAGES={extra_pip_packages}"
    os.system(f"{cls.executable} build --build-arg PYTHON_VERSION={version} {extra} -t {IMAGE_NAME} -f {dockerfile}")
    cls.image_built = True

  def __init__(self, base_path: pathlib.Path, extra_pip_packages: str = "", timeout_secs=30):
    if not ContainerSandbox.image_built:
      ContainerSandbox.build_image(extra_pip_packages)

    super(ContainerSandbox).__init__()
    self.timeout_secs = timeout_secs
    self.id = ContainerSandbox.containers
    ContainerSandbox.containers += 1
    self.call_count = 0

    self.output_path = pathlib.Path(base_path) / f"sandbox{self.id}"
    if not self.output_path.exists():
      self.output_path.mkdir(parents=True)

  def run(
          self,
          program: str,
          function_to_run: str,
          test_input,
          timeout_seconds: int,
  ) -> tuple[Any, bool]:

    mount_path = (self.output_path / f"call{self.call_count}").absolute()
    if not mount_path.exists():
      mount_path.mkdir()

    input_hash = hash(test_input)
    input_path = (self.output_path / f"{input_hash}.pickle").absolute()
    if not input_path.exists():
      with open(input_path, "wb") as f:
        cloudpickle.dump(test_input, f)
    try:
      namespace = Sandbox.compile_code(program)

      prog_file = (mount_path / f"prog.pickle").absolute()
      with open(prog_file, "wb+") as f:
        cloudpickle.dump(namespace[function_to_run], f)

      error_file = self.output_path / f"stderr_{self.call_count}.log"

      cmd = (f"/usr/bin/podman run "
             f"--timeout={self.timeout_secs} "
             f"-v {CONTAINER_MAIN}:/main.py "
             f"-v {mount_path}:/workspace "
             f"-v {input_path}:/input.pickle "
             f"{IMAGE_NAME}:latest /usr/local/bin/python3 "
             f"/main.py /workspace/prog.pickle /input.pickle /workspace/output.pickle"
             f"  2> {error_file}")
      print(f"Executing: {cmd}")
      retcode = os.system(cmd)
      self.call_count += 1

      if retcode != 0:
        self._save_diagnostics(program, mount_path)
        return None, False

      output_file = mount_path / f"output.pickle"
      with open(output_file, "rb") as f:
        out = cloudpickle.load(f)
        return out, True
    except Exception as e:
      print(f"Could not execute code: {e}")
      self._save_diagnostics(program, mount_path)

  def _save_diagnostics(self, program: str, output_path: pathlib.Path):
    filepath = output_path / "program.py"
    print(f"Writing program to {filepath}")
    with open(filepath, "w+") as f:
      f.write(program)

def test_container_sandbox():
  test_prog = """
print("running!")
def x(y):
  print(f"Received {y}")
  return y + 1
   """
  sandbox = ContainerSandbox("/tmp/")
  ret, success = sandbox.run(test_prog, "x", 10, 1)
  assert success
  assert ret == 11


if __name__ == "__main__":
  test_container_sandbox()
