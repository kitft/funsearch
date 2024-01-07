import logging
import os
import pathlib
import sys
import time
from pathlib import Path

import click
import llm
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))

from funsearch import config, core, sandbox, sampler, programs_database, code_manipulation, evaluator


LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)


@click.command()
@click.argument("spec_file", type=click.File("r"))
@click.option('--model_name', default="gpt-3.5-turbo-instruct", help='LLM model')
@click.option('--output_path', default="./data/", type=click.Path(file_okay=False), help='path for logs and data')
@click.option('--load_backup', default=None, type=click.File("rb"), help='Use existing program database')
@click.option('--iterations', default=-1, type=click.INT, help='Max iterations per sampler')
@click.option('--samplers', default=15, type=click.INT, help='Samplers')
def main(spec_file, model_name, output_path, load_backup, iterations, samplers):
  # Load environment variables from .env file.
  #
  # Using OpenAI APIs with 'llm' package requires setting the variable
  # OPENAI_API_KEY=sk-...
  # See 'llm' package on how to use other providers.
  load_dotenv()

  specification = spec_file.read()

  timestamp = int(time.time())

  log_path = pathlib.Path(output_path) / str(timestamp)
  if not log_path.exists():
    log_path.mkdir(parents=True)
    logging.info(f"Writing logs to {log_path}")

  model = llm.get_model(model_name)
  model.key = model.get_key()
  lm = sampler.LLM(2, model, log_path)

  function_to_evolve, function_to_run = core._extract_function_names(specification)
  template = code_manipulation.text_to_program(specification)

  conf = config.Config(num_evaluators=1)
  database = programs_database.ProgramsDatabase(
      conf.programs_database, template, function_to_evolve)
  if load_backup:
    database.load(load_backup)

  inputs = [4]

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
