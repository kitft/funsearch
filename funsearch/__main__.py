# import json
# import logging
# import os
# import pathlib
# import pickle
# import time

# import click
# import llm
# from dotenv import load_dotenv


# from funsearch import config, core, sandbox, sampler, programs_database, code_manipulation, evaluator

# LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
# logging.basicConfig(level=LOGLEVEL)


# def get_all_subclasses(cls):
#   all_subclasses = []

#   for subclass in cls.__subclasses__():
#     all_subclasses.append(subclass)
#     all_subclasses.extend(get_all_subclasses(subclass))

#   return all_subclasses


# SANDBOX_TYPES = get_all_subclasses(sandbox.DummySandbox) + [sandbox.DummySandbox]
# SANDBOX_NAMES = [c.__name__ for c in SANDBOX_TYPES]


# def parse_input(filename_or_data: str):
#   if len(filename_or_data) == 0:
#     raise Exception("No input data specified")
#   p = pathlib.Path(filename_or_data)
#   if p.exists():
#     if p.name.endswith(".json"):
#       return json.load(open(filename_or_data, "r"))
#     if p.name.endswith(".pickle"):
#       return pickle.load(open(filename_or_data, "rb"))
#     raise Exception("Unknown file format or filename")
#   if "," not in filename_or_data:
#     data = [filename_or_data]
#   else:
#     data = filename_or_data.split(",")
#   if data[0].isnumeric():
#     f = int if data[0].isdecimal() else float
#     data = [f(v) for v in data]
#   return data

# @click.group()
# @click.pass_context
# def main(ctx):
#   pass


# @main.command()
# @click.argument("spec_file", type=click.File("r"))
# @click.argument('inputs')
# @click.option('--model_name', default="mistral/codestral-latest", help='LLM model')
# @click.option('--output_path', default="./data/", type=click.Path(file_okay=False), help='path for logs and data')
# @click.option('--load_backup', default=None, type=click.File("rb"), help='Use existing program database')
# @click.option('--iterations', default=-1, type=click.INT, help='Max iterations per sampler')
# @click.option('--samplers', default=1, type=click.INT, help='Samplers: 1 due to single-threaded implementation')
# @click.option('--sandbox_type', default="ContainerSandbox", type=click.Choice(SANDBOX_NAMES), help='Sandbox type')
# @click.option('--num_islands', default=10, type=click.INT, help='Number of islands')
# def run(spec_file, inputs, model_name, output_path, load_backup, iterations, samplers, sandbox_type, num_islands):
#   """ Execute function-search algorithm:

# \b
#   SPEC_FILE is a python module that provides the basis of the LLM prompt as
#             well as the evaluation metric.
#             See examples/cap_set_spec.py for an example.\n
# \b
#   INPUTS    input filename ending in .json or .pickle, or a comma-separated
#             input data. The files are expected contain a list with at least
#             one element. Elements shall be passed to the solve() method
#             one by one. Examples
#               8
#               8,9,10
#               ./examples/cap_set_input_data.json
# """

#   # Load environment variables from .env file.
#   #
#   # Using OpenAI APIs with 'llm' package requires setting the variable
#   # OPENAI_API_KEY=sk-...
#   # See 'llm' package on how to use other providers.
#   load_dotenv()

#   timestamp = str(int(time.time()))
#   log_path = pathlib.Path(output_path) / timestamp
#   if not log_path.exists():
#     log_path.mkdir(parents=True)
#     logging.info(f"Writing logs to {log_path}")

#   model = llm.get_model(model_name)
#   #model.key = model.get_key()
#   model.key = os.environ.get('MISTRAL_API_KEY')
#   lm = sampler.LLM(2, model, log_path)

#   specification = spec_file.read()
#   function_to_evolve, function_to_run = core._extract_function_names(specification)
#   template = code_manipulation.text_to_program(specification)

#   conf = config.Config(num_evaluators=1, num_islands=num_islands)
#   database = programs_database.ProgramsDatabase(
#     conf.programs_database, template, function_to_evolve, identifier=timestamp)
#   if load_backup:
#     database.load(load_backup)

#   inputs = parse_input(inputs)

#   sandbox_class = next(c for c in SANDBOX_TYPES if c.__name__ == sandbox_type)
#   evaluators = [evaluator.Evaluator(
#     database,
#     sandbox_class(base_path=log_path),
#     template,
#     function_to_evolve,
#     function_to_run,
#     inputs,
#   ) for _ in range(conf.num_evaluators)]

#   # We send the initial implementation to be analysed by one of the evaluators.
#   initial = template.get_function(function_to_evolve).body
#   evaluators[0].analyse(initial, island_id=None, version_generated=None)
#   assert len(database._islands[0]._clusters) > 0, ("Initial analysis failed. Make sure that Sandbox works! "
#                                                    "See e.g. the error files under sandbox data.")

#   samplers = [sampler.Sampler(database, evaluators, lm)
#               for _ in range(samplers)]

#   core.run(samplers, database, iterations)


# @main.command()
# @click.argument("db_file", type=click.File("rb"))
# def ls(db_file):
#   """List programs from a stored database (usually in data/backups/ )"""
#   conf = config.Config(num_evaluators=1)

#   # A bit silly way to list programs. This probably does not work if config has changed any way
#   database = programs_database.ProgramsDatabase(
#     conf.programs_database, None, "", identifier="")
#   database.load(db_file)

#   progs = database.get_best_programs_per_island()
#   print("Found {len(progs)} programs")
#   for i, (prog, score) in enumerate(progs):
#     print(f"{i}: Program with score {score}")
#     print(prog)
#     print("\n")


# if __name__ == '__main__':
#   main()





import json
import logging
import os
import pathlib
import pickle
import time
import asyncio

import click
import llm
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
@click.option('--samplers', default=1, type=click.INT, help='Number of samplers')
@click.option('--sandbox_type', default="ContainerSandbox", type=click.Choice(SANDBOX_NAMES), help='Sandbox type')
@click.option('--num_evaluators', default=1, type=click.INT, help='Number of evaluators')
@click.option('--num_islands', default=10, type=click.INT, help='Number of islands')
@click.option('--run_duration', default=10000, type=click.INT, help='Run duration')
def runAsync(spec_file, inputs, model_name, output_path, load_backup, iterations, samplers, sandbox_type, num_islands,num_evaluators,run_duration):
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
    conf = config.Config(num_evaluators=num_evaluators, num_islands=num_islands, sandbox=sandbox_type,num_samplers=samplers,run_duration=run_duration)

    #model = [llm.get_model(model_name) for _ in range(samplers)]
    model = [sampler.MistralModel(model_name, top_p=conf.top_p, temperature=conf.temperature) for _ in range(samplers)]
    for m in model:
        m.key = os.environ.get('MISTRAL_API_KEY')
    lm = [sampler.LLM(2, m, log_path) for m in model]

    specification = spec_file.read()
    function_to_evolve, function_to_run = core._extract_function_names(specification)
    template = code_manipulation.text_to_program(specification)

    database = programs_database.ProgramsDatabase(
        conf.programs_database, template, function_to_evolve, identifier=timestamp)
    if load_backup:
        database.load(load_backup)

    parsed_inputs = parse_input(inputs)
    sandbox_class = next(c for c in SANDBOX_TYPES if c.__name__ == sandbox_type)
    multitestingconfig = config.MultiTestingConfig(log_path=log_path, sandbox_class=sandbox_class, parsed_inputs=parsed_inputs,
                                                    template=template, function_to_evolve=function_to_evolve, function_to_run=function_to_run, lm=lm,timestamp=timestamp)

    async def initiate_search():
        async_database = multi_testing.AsyncProgramsDatabase(database)
        await multi_testing.runAsync(conf, async_database, multitestingconfig)

    try:
        asyncio.run(initiate_search())
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt. Stopping.")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
    finally:
        database.backup()
        # Ensure all pending tasks are cancelled
        try:
            loop = asyncio.get_running_loop()
            for task in asyncio.all_tasks(loop=loop):
                task.cancel()
            # Wait for all tasks to be cancelled
            loop.run_until_complete(
                asyncio.gather(*asyncio.all_tasks(loop=loop), return_exceptions=True)
            )
        except RuntimeError:
            # No running event loop
            pass
        # make plots
        plotscores(str(timestamp))

@main.command()
@click.argument("db_file", type=click.File("rb"))
def lsAsync(db_file):
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


##### TEMPORARY FUNCTION FOR OLD SCORING TRACKER
# def generate_score_graph(timestamp):
#     """
#     Generate a graph of average and best scores for each island based on the given timestamp.
    
#     Args:
#     timestamp (str): The timestamp of the data directory to analyze.
    
#     Returns:
#     None (saves the graph as a PNG file)
#     """
#     data_dir = f"./data/{timestamp}"
#     scores = []

#     # Collect scores from all sandboxes and calls
#     for sandbox in os.listdir(data_dir):
#         if sandbox.startswith("sandbox"):
#             sandbox_dir = os.path.join(data_dir, sandbox)
#             for call in os.listdir(sandbox_dir):
#                 if call.startswith("call"):
#                     output_file = os.path.join(sandbox_dir, call, "output.pickle")
#                     if os.path.exists(output_file):
#                         with open(output_file, "rb") as f:
#                             try:
#                                 score = pickle.load(f)
#                                 #print(score)
#                                 #break
#                                 if isinstance(score, (int, float)):
#                                     sandbox_num = int(sandbox.split("sandbox")[1])
#                                     call_num = int(call.split("call")[1])
#                                     scores.append((sandbox_num, call_num, score))
#                             except:
#                                 pass  # Skip if unable to load or if it's not a number

#     # Convert to DataFrame for easier manipulation
#     # Pad with zeros if ragged
#     print(scores)
#     #max_sandbox = max(score[0] for score in scores)
#     #max_call = max(score[1] for score in scores)
#     #padded_scores.extend((i, j, 0) for i in range(max_sandbox + 1) for j in range(max_call + 1) if (i, j) not in {(s[0], s[1]) for s in scores})
#     df = pd.DataFrame(scores, columns=['sandbox', 'call', 'score'])
#     print(df.shape)

#     # Calculate average and best scores for each call
#     avg_scores = df.groupby('call')['score'].mean()
#     best_scores = df.groupby('call')['score'].max()
#     print(best_scores.shape)

#     # Create the graph
#     plt.figure(figsize=(12, 6))
#     plt.plot(avg_scores.index, avg_scores.values, label='Average Score', marker='o')
#     plt.plot(best_scores.index, best_scores.values, label='Best Score', marker='s')
#     plt.xlabel('Call Number')
#     plt.ylabel('Score')
#     plt.title(f'Average and Best Scores per Call (Timestamp: {timestamp})')
#     plt.legend()
#     plt.grid(True)

#     # Save the graph
#     plt.savefig(f'./data/score_graph_{timestamp}.png')
#     plt.close()

#     print(f"Graph saved as score_graph_{timestamp}.png")

# @main.command()
# @click.argument("timestamp")
# def plotscoresold(timestamp):
#     """Generate a graph of scores for the given timestamp."""
#     import subprocess
#     subprocess.check_call(["pip", "install", "pandas", "matplotlib"])
#     generate_score_graph(timestamp)

def plotscores(timestamp):
    """Generate a graph of best overall score and best scores per island over time."""
    #install dependencies
    import subprocess
    subprocess.check_call(["pip", "install", "pandas", "matplotlib"])
    import pandas as pd
    import matplotlib.pyplot as plt
    #from pathlib import Path

    # Read the CSV file
    timestamp = str(timestamp)
    csv_filename = f"./data/scores/scores_log_{timestamp}.csv"
    df = pd.read_csv(csv_filename)

    # Convert Time to seconds (assuming it's already in seconds)
    df['Time'] = pd.to_numeric(df['Time'])

    # Calculate the best overall score and average score at each time point
    df['Best Overall'] = df.groupby('Time')['Best Score'].transform('max')
    df['Average Overall'] = df.groupby('Time')['Best Score'].transform('mean')

    # Create the plot for best scores
    plt.figure(figsize=(12, 6))

    # Plot best overall score
    plt.plot(df['Time'].unique(), df.groupby('Time')['Best Overall'].first(), 
             label='Best Overall', linewidth=2, color='black')

    # Plot best scores per island
    for island in df['Island'].unique():
        island_data = df[df['Island'] == island]
        plt.plot(island_data['Time'], island_data['Best Score'], 
                 label=f'Island {island}', alpha=0.7)

    plt.xlabel('Time (seconds)')
    plt.ylabel('Score')
    plt.title(f'Best Scores Over Time (Timestamp: {timestamp})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

    # Adjust layout to prevent cutting off labels
    plt.tight_layout()

    # Save the best scores graph
    plt.savefig(f'./data/graphs/best_scores_over_time_{timestamp}.png', bbox_inches='tight')
    plt.close()

    print(f"Best scores graph saved as best_scores_over_time_{timestamp}.png")

    # Create a new plot for average scores
    plt.figure(figsize=(12, 6))

    # Plot average overall score
    plt.plot(df['Time'].unique(), df.groupby('Time')['Average Overall'].first(), 
             label='Average Overall', linewidth=2, color='black')

    # Plot average scores per island
    for island in df['Island'].unique():
        island_data = df[df['Island'] == island]
        plt.plot(island_data['Time'], island_data.groupby('Time')['Best Score'].mean(), 
                 label=f'Island {island}', alpha=0.7)

    plt.xlabel('Time (seconds)')
    plt.ylabel('Score')
    plt.title(f'Average Scores Over Time (Timestamp: {timestamp})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

    # Adjust layout to prevent cutting off labels
    plt.tight_layout()

    # Save the average scores graph
    plt.savefig(f'./data/graphs/average_scores_over_time_{timestamp}.png', bbox_inches='tight')
    plt.close()

    print(f"Average scores graph saved as average_scores_over_time_{timestamp}.png")

@main.command()
@click.argument("timestamp")
def makegraphs(timestamp):
    plotscores(timestamp)
