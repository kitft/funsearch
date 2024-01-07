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


@click.command()
@click.argument("spec_file", type=click.File("r"))
@click.argument('inputs')
@click.option('--model_name', default="gpt-3.5-turbo-instruct", help='LLM model')
@click.option('--output_path', default="./data/", type=click.Path(file_okay=False), help='path for logs and data')
@click.option('--load_backup', default=None, type=click.File("rb"), help='Use existing program database')
@click.option('--iterations', default=-1, type=click.INT, help='Max iterations per sampler')
@click.option('--samplers', default=15, type=click.INT, help='Samplers')
def main(spec_file, inputs, model_name, output_path, load_backup, iterations, samplers):
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
  model.key = model.get_key()
  lm = sampler.LLM(2, model, log_path)

  specification = spec_file.read()
  function_to_evolve, function_to_run = core._extract_function_names(specification)
  template = code_manipulation.text_to_program(specification)

  conf = config.Config(num_evaluators=1)
  database = programs_database.ProgramsDatabase(
      conf.programs_database, template, function_to_evolve)
  if load_backup:
    database.load(load_backup)

  inputs = parse_input(inputs)

  evaluators = [evaluator.Evaluator(
    database,
    sandbox.ContainerSandbox(log_path, "numpy", 20),
    template,
    function_to_evolve,
    function_to_run,
    inputs,
  ) for _ in range(conf.num_evaluators)]

  # We send the initial implementation to be analysed by one of the evaluators.
  initial = template.get_function(function_to_evolve).body
  evaluators[0].analyse(initial, island_id=None, version_generated=None)

  samplers = [sampler.Sampler(database, evaluators, lm)
              for _ in range(samplers)]

  core.run(samplers, database, iterations)


if __name__ == '__main__':
  main()
