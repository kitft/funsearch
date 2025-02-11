import json
import logging
import os
import pathlib
import pickle
import time
import asyncio
from typing import Union, Optional

import click
#import llm
from dotenv import load_dotenv

from funsearch import async_agents, config, core, sandbox, sampler, programs_database, code_manipulation, models, logging_stats
from funsearch.utilities import oeis_util

LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(format='%(asctime)s.%(msecs)03d:%(levelname)s:%(message)s',level=LOGLEVEL,datefmt='%Y-%m-%d-%H-%M-%S')

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
@click.option('--model', default="mistral/codestral-latest", help='LLM model (or comma-separated list of models or model:count entries)')
@click.option('--output_path', default="./data/", type=click.Path(file_okay=False), help='Path for logs and data')
@click.option('--load_backup', default=None, type=click.File("rb"), help='Use existing program database')
@click.option('--iterations', default=-1, type=click.INT, help='Max iterations per sampler')
@click.option('--sandbox', default="ExternalProcessSandbox", type=click.Choice(SANDBOX_NAMES), help='Sandbox type')
@click.option('--samplers', default=1, type=click.INT, help='Number of samplers')
@click.option('--evaluators', default=10, type=click.INT, help='Number of evaluators')
@click.option('--islands', default=10, type=click.INT, help='Number of islands')
@click.option('--reset', default=600, type=click.INT, help='Reset period in seconds')
@click.option('--duration', default=3600, type=click.INT, help='Duration in seconds')
@click.option('--temperature', default="1.0", type=str, help='LLM temperature or comma-separated list of temperatures')
@click.option('--team', default=None, type=str, help='wandb team name')
@click.option('--envfile', default=None, type= str, help='path to .env file')
@click.option('--name', default=None, help='Unique ID for wandb. Default is timestamp')
@click.option('--tag', default=None, type=str, help='Tag for wandb. Default is None')
def runAsync(spec_file, inputs, model, output_path, load_backup, iterations, sandbox, samplers, evaluators, islands, reset, duration, temperature, team, envfile, name, tag):
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
    if envfile is not None:
        logging.info(f"Loading environment variables from {envfile}")
        if not os.path.exists(envfile):
            raise Exception(f"Environment file {envfile} does not exist")
        load_dotenv(envfile)
        if envfile is not None:
            with open(envfile, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        key = line.split('=')[0]
                        print(f"Found environment variable: {key}")
    else:
        logging.info("No .env file specified, using environment variables")
    

    names_of_models = model
    timestamp = str(int(time.time()))
    problem_name = spec_file.name.split("/")[-1].split(".")[0]

    if name == None:
        timestamp = timestamp
    else:
        timestamp = name
    problem_identifier = problem_name + "_" + timestamp
    model_identifier =  names_of_models + "_T"+ temperature

    name_for_saving = model_identifier + "_" + problem_identifier
    #names_of_models = model
    #name_of_run = "run_" + names_of_models + "_" + name_val

    log_path = pathlib.Path(output_path) / problem_name / timestamp
    if not log_path.exists():
        log_path.mkdir(parents=True)
        logging.info(f"Writing logs to {log_path}")
    model_list = model.split(",")
    model_counts = [int(m.split('*')[1]) if '*' in m else 1 for m in model_list]
    model_keys = [int(m.split('*')[2]) if m.count('*') > 1 else 0 for m in model_list]
    model_list = [m.split('*')[0] for m in model_list]
    if ',' in temperature:
        temperature_list = temperature.split(",")
    else:
        temperature_list = [temperature] * len(model_list)
    if len(temperature_list) != len(model_list):
        raise Exception(f"Temperature list length {len(temperature_list)} does not match model list length {len(model_list)}")

    if sum(model_counts) > samplers:
        samplers = sum(model_counts)
        logging.info(f"Setting samplers to {samplers}")
    elif sum(model_counts) < samplers:
        i = 0
        while sum(model_counts) < samplers:
            model_counts[i] += 1
            i = (i+1) % len(model_counts)
    logging.info(f"Sampling with {model_counts} copies of model(s): {model_list}")
    logging.info(f"Using LLM temperature(s): {temperature}")

    conf = config.Config(sandbox=sandbox, num_samplers=samplers, num_evaluators=evaluators, num_islands=islands, reset_period=reset, run_duration=duration,llm_temperature=temperature_list)
    logging.info(f"run_duration = {conf.run_duration}, reset_period = {conf.reset_period}")

    temperature_list = sum([model_counts[i]*[float(temperature_list[i])] for i in range(len(model_list))],[])
    model_list = sum([model_counts[i]*[model_list[i]] for i in range(len(model_list))],[])
    keynum_list = sum([model_counts[i]*[model_keys[i]] for i in range(len(model_keys))],[])

    logging.info(f"keynum list: {keynum_list}")
    lm = [sampler.LLM(conf.samples_per_prompt, models.LLMModel(model_name=model_list[i], top_p=conf.top_p,
        temperature=temperature_list[i], keynum=keynum_list[i],id = i,log_path=log_path,system_prompt=conf.system_prompt), log_path=log_path,api_call_timeout=conf.api_call_timeout,api_call_max_retries=conf.api_call_max_retries,ratelimit_backoff=conf.ratelimit_backoff) for i in range(len(model_list))]

    specification = spec_file.read()
    function_to_evolve, function_to_run = core._extract_function_names(specification)
    template = code_manipulation.text_to_program(specification)

    database = programs_database.ProgramsDatabase(
        conf.programs_database, template, function_to_evolve, identifier=timestamp)
    if load_backup:
        # If load_backup is a file that exists directly, use it
        if os.path.isfile(load_backup):
            database.load(load_backup)
        else:
            # If it's a string/number or partial path, search in backups folder
            backup_dir = os.path.join(output_path, "backups")
            matching_files = []
            for root, dirs, files in os.walk(backup_dir):
                for file in files:
                    if load_backup in file:
                        matching_files.append(os.path.join(root, file))
            
            if matching_files:
                # Get most recently modified matching file
                latest_file = max(matching_files, key=os.path.getmtime)
                logging.info(f"Found backup file: {latest_file}")
                database.load(latest_file)
            else:
                raise FileNotFoundError(f"Could not find backup file matching '{load_backup}' in {backup_dir}")

    parsed_inputs = parse_input(inputs)
    sandbox_class = next(c for c in SANDBOX_TYPES if c.__name__ == sandbox)

    portable_config = async_agents.PortableSystemConfig(log_path=log_path, output_path=output_path,sandbox_class=sandbox_class, parsed_inputs=parsed_inputs,
                                                    template=template, function_to_evolve=function_to_evolve, function_to_run=function_to_run, 
                                                    lm=lm,model_identifier=model_identifier,problem_name=problem_name,timestamp=timestamp,name_for_saving=name_for_saving,problem_identifier=problem_identifier,tag=tag)

    async def initiate_search():
        async_database = async_agents.AsyncProgramsDatabase(database)
        await async_agents.run_agents(conf, async_database, portable_config, team)

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
            # Wait up to 30s for tasks to cancel gracefully
            try:
                loop.run_until_complete(
                    asyncio.wait_for(
                        asyncio.gather(*asyncio.all_tasks(loop=loop), return_exceptions=True),
                        timeout=30
                    )
                )
            except asyncio.TimeoutError:
                logging.warning("Tasks did not cancel within 30s - forcing exit")
        except RuntimeError:
            # No running event loop
            pass
        # make plots
        plotscores(problem_identifier, output_path)
        # Ensure process termination
        logging.info("Shutting down gracefully...")
        time.sleep(3)  # Brief pause to allow final cleanup
        import sys
        sys.exit(0)  # Exit with error code 1 to indicate non-normal termination
@main.command()
@click.argument("db_file")
def ls(db_file):
    """List programs from a stored database.

    DB_FILE: Path to database file or partial name to search in data/backups/
    
    Lists the best program from each island.
    """
    conf = config.Config(num_evaluators=1)

    # Initialize the programs database
    database = programs_database.ProgramsDatabase(
        conf.programs_database, None, "", identifier="")

    # If it's a file that exists, load it directly
    if os.path.exists(db_file):
        database.load(db_file)
    else:
        # Search in backups folder for partial matches
        backup_dir = os.path.join("./data", "backups")
        matching_files = []
        for root, dirs, files in os.walk(backup_dir):
            for file in files:
                if db_file in file:
                    matching_files.append(os.path.join(root, file))
        
        if matching_files:
            # Get most recently modified matching file
            latest_file = max(matching_files, key=os.path.getmtime)
            logging.info(f"Found backup file: {latest_file}")
            database.load(latest_file)
        else:
            raise FileNotFoundError(f"Could not find backup file matching '{db_file}' in {backup_dir}")

    best_programs = database.get_best_programs_per_island()
    print(f"Found {len(best_programs)} programs")
    for i, (prog, score) in enumerate(best_programs):
        print(f"{i}: Program with score {score}")
        print(prog)
        print("\n")

@main.command()
@click.argument("a_number")
@click.argument("save_path", required=False)
@click.option('--max-terms', type=int, help='Maximum number of terms to fetch')
def oeis(a_number: str, save_path: Optional[str], max_terms: Optional[int]):
    """Fetch and save an OEIS sequence.
    
    A_NUMBER: The A-number of the sequence (e.g. 'A001011' or '001011')
    SAVE_PATH: Optional path to save the pickle file. Defaults to ./examples/oeis_data/<seqname>.pkl
    """
    try:
        file_path, json_path, sequence = oeis_util.save_oeis_sequence(a_number, save_path, max_terms)
        print(f"Successfully saved sequence to {file_path} (and also saved to {json_path}): first few elements are {sequence[:10]}")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise click.Abort()

@main.command()
@click.option('--duration', default=5, type=click.INT, help='Duration in seconds')
@click.option('--samplers', default=1, type=click.INT, help='Number of samplers')
@click.option('--evaluators', default=1, type=click.INT, help='Number of evaluators')
@click.option('--output_path', default="./data/tests/", type=click.Path(file_okay=False), help='Path for logs and data')
def mock_test(duration, samplers, evaluators, output_path):
    """Run a test with mock model for validation using cap_set_spec."""
    os.environ["WANDB_MODE"] = "disabled"
    
    spec_path = "examples/cap_set_spec.py"
    if not os.path.exists(spec_path):
        raise click.ClickException("cap_set_spec.py not found in examples directory")
    
    try:
        # Call runAsync's implementation directly
        runAsync.callback(
            spec_file=open(spec_path, 'r'),
            inputs="8",  # Standard test input for cap_set_spec
            model="mock_model",
            output_path=output_path,
            load_backup=None,
            iterations=-1,
            sandbox="ExternalProcessSandbox",
            samplers=samplers,
            evaluators=evaluators,
            islands=1,
            reset=60,
            duration=duration,
            temperature="0.8",
            team=None,
            envfile=None,
            name=None,
            tag="test"
        )
    except Exception as e:
        raise click.ClickException(str(e))

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

def plotscores(name, output_path = "./data"):
    """Generate a graph of best overall score and best scores per island over time."""
    #install dependencies
    # import subprocess
    #subprocess.check_call(["pip", "install", "pandas", "matplotlib"])
    import pandas as pd
    import matplotlib.pyplot as plt
    #from pathlib import Path

    # Read the CSV file
    timestamp = str(name)
    csv_filename = os.path.join(output_path, "scores", f"scores_log_{name}.csv")
    df = pd.read_csv(csv_filename)

    # Convert Time to seconds (assuming it's already in seconds)
    df['Time'] = pd.to_numeric(df['Time'])

    # Calculate the best overall score at each time point
    df['Best Overall'] = df.groupby('Time')['Best Score'].transform('max')

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
    
    os.makedirs(os.path.join(output_path, 'graphs'), exist_ok=True)
    plt.savefig(os.path.join(output_path, 'graphs', f'best_scores_over_time_{timestamp}.png'), bbox_inches='tight')
    plt.close()

    print(f"Best scores graph saved as best_scores_over_time_{timestamp}.png")

    # Create a new plot for average scores
    plt.figure(figsize=(12, 6))

    # Plot average overall score
    plt.plot(df['Time'].unique(), df.groupby('Time')['Average Score'].first(), 
             label='Average Overall', linewidth=2, color='black')

    # Plot average scores per island
    for island in df['Island'].unique():
        island_data = df[df['Island'] == island]
        plt.plot(island_data['Time'], island_data['Average Score'], 
                 label=f'Island {island}', alpha=0.7)

    plt.xlabel('Time (seconds)')
    plt.ylabel('Score')
    plt.title(f'Average Scores Over Time (Timestamp: {timestamp})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    # Set y-axis limits
    #plt.ylim(250, 410)

    # Adjust layout to prevent cutting off labels
    plt.tight_layout()

    # Save the average scores graph
    plt.savefig(os.path.join(output_path, 'graphs', f'average_scores_over_time_{timestamp}.png'), bbox_inches='tight')
    plt.close()

    print(f"Average scores graph saved as average_scores_over_time_{timestamp}.png")

@main.command()
@click.argument("timestamp")
def makegraphs(timestamp):
    plotscores(timestamp)


@main.command()
@click.argument("timestamps", nargs=-1)
def removetimestamp(timestamps):
    """Remove all data associated with the specified timestamps."""
    print("TEMPORARILY REMOVED")
    return 0
    import os
    import shutil

    if not timestamps:
        print("No timestamps provided. Usage: funsearch removetimestamp timestamp1 [timestamp2 ...]")
        return

    # Define paths
    data_dir = './data'
    graphs_dir = os.path.join(data_dir, 'graphs')

    for timestamp in timestamps:
        # Remove CSV file

        csv_file = os.path.join(data_dir, f'{timestamp}.csv')
        if os.path.exists(csv_file):
            os.remove(csv_file)
            print(f"Removed CSV file: {csv_file}")

        # Remove graph files
        best_scores_graph = os.path.join(graphs_dir, f'best_scores_over_time_{timestamp}.png')
        avg_scores_graph = os.path.join(graphs_dir, f'average_scores_over_time_{timestamp}.png')
        
        for graph_file in [best_scores_graph, avg_scores_graph]:
            if os.path.exists(graph_file):
                os.remove(graph_file)
                print(f"Removed graph file: {graph_file}")

        # Remove timestamp directory if it exists
        timestamp_dir = os.path.join(data_dir, timestamp)
        if os.path.exists(timestamp_dir):
            shutil.rmtree(timestamp_dir)
            print(f"Removed timestamp directory: {timestamp_dir}")

        print(f"All data associated with timestamp {timestamp} has been removed.")



