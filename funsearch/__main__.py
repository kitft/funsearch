import json
import logging
import os
import pathlib
import pickle
import time

import click
import llm
from dotenv import load_dotenv


from funsearch import config, core, sandbox, sampler, programs_database, code_manipulation, evaluator

LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)


def get_all_subclasses(cls):
  all_subclasses = []

  for subclass in cls.__subclasses__():
    all_subclasses.append(subclass)
    all_subclasses.extend(get_all_subclasses(subclass))

  return all_subclasses


SANDBOX_TYPES = get_all_subclasses(sandbox.DummySandbox) + [sandbox.DummySandbox]
SANDBOX_NAMES = [c.__name__ for c in SANDBOX_TYPES]


def parse_input(filename_or_data: str):
  if len(filename_or_data) == 0:
    raise Exception("No input data specified")
  p = pathlib.Path(filename_or_data)
  if p.exists():
    if p.name.endswith(".json"):
      return json.load(open(filename_or_data, "r"))
    if p.name.endswith(".pickle"):
      return pickle.load(open(filename_or_data, "rb"))
    raise Exception("Unknown file format or filename")
  if "," not in filename_or_data:
    data = [filename_or_data]
  else:
    data = filename_or_data.split(",")
  if data[0].isnumeric():
    f = int if data[0].isdecimal() else float
    data = [f(v) for v in data]
  return data

@click.group()
@click.pass_context
def main(ctx):
  pass


@main.command()
@click.argument("spec_file", type=click.File("r"))
@click.argument('inputs')
@click.option('--model_name', default="mistral/codestral-latest", help='LLM model')
@click.option('--output_path', default="./data/", type=click.Path(file_okay=False), help='path for logs and data')
@click.option('--load_backup', default=None, type=click.File("rb"), help='Use existing program database')
@click.option('--iterations', default=-1, type=click.INT, help='Max iterations per sampler')
@click.option('--samplers', default=1, type=click.INT, help='Samplers: 1 due to single-threaded implementation')
@click.option('--sandbox_type', default="ContainerSandbox", type=click.Choice(SANDBOX_NAMES), help='Sandbox type')
@click.option('--num_islands', default=10, type=click.INT, help='Number of islands')
def run(spec_file, inputs, model_name, output_path, load_backup, iterations, samplers, sandbox_type, num_islands):
  """ Execute function-search algorithm:

\b
  SPEC_FILE is a python module that provides the basis of the LLM prompt as
            well as the evaluation metric.
            See examples/cap_set_spec.py for an example.\n
\b
  INPUTS    input filename ending in .json or .pickle, or a comma-separated
            input data. The files are expected contain a list with at least
            one element. Elements shall be passed to the solve() method
            one by one. Examples
              8
              8,9,10
              ./examples/cap_set_input_data.json
"""

  # Load environment variables from .env file.
  #
  # Using OpenAI APIs with 'llm' package requires setting the variable
  # OPENAI_API_KEY=sk-...
  # See 'llm' package on how to use other providers.
  load_dotenv()

  timestamp = str(int(time.time()))
  log_path = pathlib.Path(output_path) / timestamp
  if not log_path.exists():
    log_path.mkdir(parents=True)
    logging.info(f"Writing logs to {log_path}")

  model = llm.get_model(model_name)
  #model.key = model.get_key()
  model.key = os.environ.get('MISTRAL_API_KEY')
  lm = sampler.LLM(2, model, log_path)

  specification = spec_file.read()
  function_to_evolve, function_to_run = core._extract_function_names(specification)
  template = code_manipulation.text_to_program(specification)

  conf = config.Config(num_evaluators=1, num_islands=num_islands)
  database = programs_database.ProgramsDatabase(
    conf.programs_database, template, function_to_evolve, identifier=timestamp)
  if load_backup:
    database.load(load_backup)

  inputs = parse_input(inputs)

  sandbox_class = next(c for c in SANDBOX_TYPES if c.__name__ == sandbox_type)
  evaluators = [evaluator.Evaluator(
    database,
    sandbox_class(base_path=log_path),
    template,
    function_to_evolve,
    function_to_run,
    inputs,
  ) for _ in range(conf.num_evaluators)]

  # We send the initial implementation to be analysed by one of the evaluators.
  initial = template.get_function(function_to_evolve).body
  evaluators[0].analyse(initial, island_id=None, version_generated=None)
  assert len(database._islands[0]._clusters) > 0, ("Initial analysis failed. Make sure that Sandbox works! "
                                                   "See e.g. the error files under sandbox data.")

  samplers = [sampler.Sampler(database, evaluators, lm)
              for _ in range(samplers)]

  core.run(samplers, database, iterations)


@main.command()
@click.argument("db_file", type=click.File("rb"))
def ls(db_file):
  """List programs from a stored database (usually in data/backups/ )"""
  conf = config.Config(num_evaluators=1)

  # A bit silly way to list programs. This probably does not work if config has changed any way
  database = programs_database.ProgramsDatabase(
    conf.programs_database, None, "", identifier="")
  database.load(db_file)

  progs = database.get_best_programs_per_island()
  print("Found {len(progs)} programs")
  for i, (prog, score) in enumerate(progs):
    print(f"{i}: Program with score {score}")
    print(prog)
    print("\n")


if __name__ == '__main__':
  main()
