import json
import logging
import os
import pathlib
import pickle
import time
import asyncio

import click
#import llm
from dotenv import load_dotenv

from funsearch import config, core, sandbox, sampler, programs_database, code_manipulation, evaluator, multi_testing

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
@click.option('--output_path', default="./data/", type=click.Path(file_okay=False), help='Path for logs and data')
@click.option('--load_backup', default=None, type=click.File("rb"), help='Use existing program database')
@click.option('--iterations', default=-1, type=click.INT, help='Max iterations per sampler')
@click.option('--samplers', default=1, type=click.INT, help='Number of samplers (1 due to single-threaded implementation)')
@click.option('--sandbox_type', default="ContainerSandbox", type=click.Choice(SANDBOX_NAMES), help='Sandbox type')
@click.option('--num_islands', default=10, type=click.INT, help='Number of islands')
def run(spec_file, inputs, model_name, output_path, load_backup, iterations, samplers, sandbox_type, num_islands):
    """Execute the function-search algorithm.

    SPEC_FILE: A Python module providing the basis of the LLM prompt and the evaluation metric.
               Example: examples/cap_set_spec.py

    INPUTS: 
        - A filename ending in .json or .pickle
        - Comma-separated input data
        - Files should contain a list with at least one element
        - Elements are passed to the solve() method one by one
        Examples:
            8
            8,9,10
            ./examples/cap_set_input_data.json
    """

    # Load environment variables from the .env file.
    load_dotenv()

    timestamp = str(int(time.time()))
    log_path = pathlib.Path(output_path) / timestamp
    if not log_path.exists():
        log_path.mkdir(parents=True)
        logging.info(f"Writing logs to {log_path}")

    model = llm.get_model(model_name)
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

    parsed_inputs = parse_input(inputs)

    sandbox_class = next(c for c in SANDBOX_TYPES if c.__name__ == sandbox_type)
    evaluators = [evaluator.Evaluator(
        database,
        sandbox_class(base_path=log_path),
        template,
        function_to_evolve,
        function_to_run,
        parsed_inputs,
    ) for _ in range(conf.num_evaluators)]

    # Register the initial implementation for analysis.
    initial = template.get_function(function_to_evolve).body
    evaluators[0].analyse(initial, island_id=None, version_generated=None)
    assert len(database._islands[0]._clusters) > 0, (
        "Initial analysis failed. Ensure that the Sandbox is operational. "
        "Check the error files under sandbox data."
    )

    samplers = [sampler.Sampler(database, evaluators, lm)
                for _ in range(samplers)]

    async def initiate_search():
        async_database = multi_testing.AsyncProgramsDatabase(database)
        await multi_testing.runAsync(conf, async_database)

    try:
        asyncio.run(initiate_search())
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt. Stopping.")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
    finally:
        database.backup()
        # Ensure all pending tasks are cancelled
        for task in asyncio.all_tasks():
            task.cancel()
        # Wait for all tasks to be cancelled
        asyncio.get_event_loop().run_until_complete(
            asyncio.gather(*asyncio.all_tasks(), return_exceptions=True)
        )


@main.command()
@click.argument("db_file", type=click.File("rb"))
def ls(db_file):
    """List programs from a stored database (usually in data/backups/)"""
    conf = config.Config(num_evaluators=1)

    # Initialize the programs database.
    database = programs_database.ProgramsDatabase(
        conf.programs_database, None, "", identifier="")
    database.load(db_file)

    best_programs = database.get_best_programs_per_island()
    print(f"Found {len(best_programs)} programs")
    for i, (prog, score) in enumerate(best_programs):
        print(f"{i}: Program with score {score}")
        print(prog)
        print("\n")


if __name__ == "__main__":
    main()
