import asyncio
# from concurrent.futures import ThreadPoolExecutor
# from typing import List
import logging
import multiprocessing
from multiprocessing import Process, Queue
import time
import getpass
import sys

from funsearch import programs_database, evaluator, sampler, logging_stats, config as config_lib
# import pathlib

import csv
# from datetime import datetime
import os
import numpy as np

import wandb

import select as select_module
import dataclasses
from funsearch import sandbox

#config for multi-testing - not actually designed to be modified by users
@dataclasses.dataclass(frozen=False)
class PortableSystemConfig:
  """Portable configuration class"""
  log_path: str
  sandbox_class: type[sandbox.DummySandbox]
  parsed_inputs: list
  template: None
  function_to_evolve: None
  function_to_run: None
  lm: None
  timestamp: str
  model_identifier: str
  problem_name: str
  name_for_saving: str
  problem_identifier: str
  tag: str

class AsyncProgramsDatabase(programs_database.ProgramsDatabase):
    def __init__(self, database: programs_database.ProgramsDatabase):
        # Initialize with the same attributes as the original database
        for attr_name, attr_value in database.__dict__.items():
            setattr(self, attr_name, attr_value)
        self.orig_database = database

    async def get_prompt(self) -> programs_database.Prompt:
        #print("Getting prompt")
        return self.orig_database.get_prompt()

    async def register_program(self, program, scores_per_test,usage_stats):
        self.orig_database.register_program(program, scores_per_test, usage_stats)
    
    def test_nonzero_population(self):
        return self.orig_database.has_nonzero_population

# class AsyncEvaluator(evaluator.Evaluator):
#     async def async_analyse(self, sample: str, island_id: int | None, version_generated: int | None) -> None:
#         self.analyse(sample, island_id, version_generated)

# class AsyncSampler(sampler.Sampler):
#     async def async_sample(self, prompt, eval_queue, db_queue):
#         return self.sample(prompt, eval_queue, db_queue)

def evaluator_process(eval_queue: Queue, result_queue: Queue, config: config_lib.Config, portable_config: PortableSystemConfig, id: int):
    evaluator_instance = evaluator.Evaluator(
        AsyncProgramsDatabase(config.programs_database),
        portable_config.sandbox_class(base_path=portable_config.log_path, id=id),
        portable_config.template,
        portable_config.function_to_evolve,
        portable_config.function_to_run,
        portable_config.parsed_inputs,
        id=id,
        log_path=portable_config.log_path
    )
    #evaluator_process is synchronous, and initialised on alternate processes
    #time.sleep(id*0.1)
    while True:
        try:
            task = eval_queue.get(timeout=1)
            if task is None:
                break
            sample, usage_stats = task
            result = evaluator_instance.analyse(sample,  usage_stats)
            #logging.info(f"Evaluator {id}, island {usage_stats.island_id}, version generated {usage_stats.version_generated}, island version {usage_stats.island_version}: Eval Queue size: {eval_queue.qsize()}, Result Queue size: {result_queue.qsize()}, Result: {bool(result)}")
            if result:
                result_queue.put(result)
                #logging.info("increased result queue size to %d"%(result_queue.qsize()))
        except multiprocessing.queues.Empty:
            time.sleep(0.01)
            continue
    


    

async def database_worker(result_queue: multiprocessing.Queue, database: AsyncProgramsDatabase):
    logging.info("database worker start")
    while True:
        try:
            result = result_queue.get_nowait()
            if result is None:
                logging.info("Database worker received shutdown signal")
                break
            new_function_or_error, scores_per_test, usage_stats = result
            #model_name = usage_stats.model
            # else:
            #    logging.info("reduced result queue size to %d"%(result_queue.qsize()))
            await database.register_program(new_function_or_error,scores_per_test,usage_stats)
        except multiprocessing.queues.Empty:
            await asyncio.sleep(0.1)
    logging.info("Database worker end")

async def sampler_worker(sampler: sampler.Sampler, eval_queue: multiprocessing.Queue, database: AsyncProgramsDatabase, config: config_lib.Config):
    #wait a random amount of time to avoid synchronisation issues and API ratelimits
    await asyncio.sleep(np.random.rand()*config.num_samplers*0.5)
    while True:
        #logging.info('Sampling with sampler %d', sampler.label)
        prompt = await database.get_prompt()
        await sampler.sample(prompt, eval_queue)
        # Adaptive sleep based on queue size
        queue_size = eval_queue.qsize()
        sleep_time = min(0.1 * (queue_size), 60)  # Cap at 5 seconds
        if sleep_time > 0.5:
            logging.info('Slowed down sampling to %f seconds', sleep_time)
        await asyncio.sleep(sleep_time)

def countdown_timer(seconds,team=None):
    """Display a countdown timer."""
    for i in range(seconds, 0, -1):
        if team:
            sys.stdout.write(f"\rUsing team '{team}' in {i} seconds... Press Enter to select different entity/skip wait. ")
        else:
            sys.stdout.write(f"\rUsing wandb default team (typically personal account) in {i} seconds... Press Enter to select different entity/skip wait. ")
        sys.stdout.flush()
        # Check if user pressed Enter
        if sys.stdin in select_module.select([sys.stdin], [], [], 1)[0]:
            sys.stdin.readline()
            sys.stdout.write('\r' + ' ' * 70 + '\r')  # Clear the line
            return False
        time.sleep(1)
    sys.stdout.write('\r' + ' ' * 70 + '\r')  # Clear the line
    return True
def select_wandb_entity(team=None):
    """Select wandb entity, with optional team default."""
    try:
        # Use the existing countdown_timer function
        if countdown_timer(10,team):
            # Countdown completed without interruption - use default team
            return team
        else:
            # User interrupted - prompt for new entity
            entity = input("\nEnter entity name (leave empty for wandb default (typically personal account)): ").strip()
            return entity if entity else None
     
    except Exception as e:
        logging.warning(f"Error in entity selection: {e}")
        return None  # Default to no entity on error

async def validate_model(lm: sampler.LLM, timeout=30):
    """Test if the model is responding correctly with timeout.
    
    Args:
        lm: The language model to test (instance of sampler.LLM)
        timeout: Maximum time in seconds to wait for response (default: 30)
    """
    try:
        test_prompt = "Write a simple Python function that adds two numbers."
        
        # Create task with timeout
        try:
            async with asyncio.timeout(timeout):
                # Use LLM's draw_sample method directly
                response, usage_stats = await lm._draw_sample(test_prompt, label=None) #do not log this, so set label to None
                if not response or len(response) < 10:  # Basic check for a reasonable response
                    raise ValueError(f"Model {lm.model.model} returned an invalid response")
                logging.info(f"Successfully validated model {lm.model.model}")
                return True
        except asyncio.TimeoutError:
            logging.error(f"Model {lm.model.model} validation timed out after {timeout} seconds")
            return False
            
    except Exception as e:
        logging.error(f"Failed to validate model {lm.model.model}: {str(e)}")
        return False

async def validate_all_models(lm_list):
    """Validate all models before starting the main loop."""
    print("\nValidating models...")
    valid_models = []
    validated_model_names = set()
    
    for lm in lm_list:
        model_name = lm.model.model
        if model_name in validated_model_names:
            # Skip validation but add to valid models if we already validated this model name
            valid_models.append(lm)
            continue
            
        print(f"Testing {model_name}...")  # Add progress indicator
        if await validate_model(lm):
            valid_models.append(lm)
            validated_model_names.add(model_name)
            print(f"✓ {model_name} passed validation")
        else:
            print(f"✗ {model_name} failed validation - skipping")
    
    if not valid_models:
        raise RuntimeError("No valid models available. Please check your API keys and model configurations.")
    
    if len(valid_models) < len(lm_list):
        print(f"\nWarning: Only {len(valid_models)}/{len(lm_list)} models passed validation")
        response = input("Continue with valid models? (Y/n): ").strip().lower()
        if response == 'n':
            raise RuntimeError("User chose to abort due to model validation failures")
    
    return valid_models

async def run_agents(config: config_lib.Config, database: AsyncProgramsDatabase, portable_config: PortableSystemConfig, team=None):
    #num_cores = min(multiprocessing.cpu_count(), config.num_evaluators,2)
    problem_identifier = portable_config.problem_name + "_" + portable_config.timestamp
    name_for_saving_to_wandb = portable_config.name_for_saving 
    num_cores = min(multiprocessing.cpu_count()-1, config.num_evaluators)
    logging.info("Number of cores/evaluators to be used: %d", num_cores)
    eval_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()

    # Validate models before starting
    valid_lm = await validate_all_models(portable_config.lm)
    if len(valid_lm) != len(portable_config.lm):
        config.num_samplers = len(valid_lm)
        logging.info(f"Adjusted number of samplers to {config.num_samplers} based on valid models")
    portable_config.lm = valid_lm

    # Start evaluator processes
    evaluator_processes = []
    for id in range(num_cores):
        p = Process(target=evaluator_process, args=(eval_queue, result_queue, config, portable_config, id))
        p.start()
        evaluator_processes.append(p)

    run_start_time = time.time()
    # Create samplers with validated models
    samplers = [sampler.Sampler(database, None, portable_config.lm[i], i) for i in range(config.num_samplers)]

    # Create and start tasks
    db_worker = asyncio.create_task(database_worker(result_queue, database))

    # Check if WANDB_API_KEY is set in environment variables
    wandb_api_key = os.environ.get('WANDB_API_KEY')
    if wandb_api_key:
        logging.info("Logging into wandb with env API key")
        wandb.login(key=wandb_api_key)
    # Get wandb entity from user
    if team==None:
        entity = select_wandb_entity(team)
    else:
        entity = team
    if entity==None:
        logging.info("Logging to wandb to default entity (typically personal account)")
    else:
        logging.info(f"Logging to wandb to entity: {entity}")
    
    logging.info(f"Initialising wandb with name: {name_for_saving_to_wandb}, tagged as: {portable_config.tag}")
    wandb.init(
        entity=entity,
        project="funsearch",
        name=name_for_saving_to_wandb,
        tags=[portable_config.tag],
        config={
            "model_names": [lm.model.model for lm in portable_config.lm],
            "num_cores": num_cores,
            "num_samplers": config.num_samplers,
            "run_duration": config.run_duration,
            "num_islands": len(database._islands),
            "problem_name": portable_config.problem_name,
            "temperatures": [lm.model.temperature for lm in portable_config.lm]

        }
    )

    os.makedirs("./data/scores", exist_ok=True)
    csv_filename = f"./data/scores/scores_log_{problem_identifier}.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Time', 'Island', 'Best Score', 'Average Score'])

   # Initial evaluation
    initial = portable_config.template.get_function(portable_config.function_to_evolve).body
    eval_queue.put((initial, logging_stats.UsageStats(id=None, model=None, prompt=None, provider=None, response=None, eval_state=None, sandbox_current_call_count=None, prompt_count=None, sampler_id=None)))

    logging.info("Waiting for initial program registration...")
    while not database.test_nonzero_population():
        await asyncio.sleep(0.1)

    logging.info("Initialising %d samplers"%(len(samplers)))
    sampler_tasks = [asyncio.create_task(sampler_worker(s, eval_queue, database,config)) for s in samplers]
    

    try:
        start_time = time.time()
        logging_info_interval = config.logging_info_interval
        while time.time() - start_time < config.run_duration:
            # Add wandb run state check
            if not wandb.run:
                logging.info("Wandb run stopped externally. Initiating graceful shutdown.")
                break

            current_time = time.time() - start_time
            if current_time >= logging_info_interval * (current_time // logging_info_interval):
                eval_queue_size = eval_queue.qsize()
                result_queue_size = result_queue.qsize()
                best_scores_per_island = database._best_score_per_island
                
                # Calculate average scores per island
                avg_scores_per_island = []
                for island in database._islands:
                    island_scores = [cluster.score for cluster in island._clusters.values()]
                    avg_score = sum(island_scores) / len(island_scores) if island_scores else 0
                    avg_scores_per_island.append(avg_score)

                # Calculate best score overall and average score overall
                best_score_overall = max(best_scores_per_island)
                avg_score_overall = sum(avg_scores_per_island) / len(avg_scores_per_island) if avg_scores_per_island else 0

                logging.info(f"Time: {current_time:.2f}s, best score: {best_score_overall:.2f}, average score: {avg_score_overall:.2f}, Eval Queue size: {eval_queue_size}, Result Queue size: {result_queue_size}")

                # Log best test scores across all islands
                # Only log per-input scores if there are multiple inputs
                if len(database._best_scores_per_test_per_island[0]) > 1:
                    best_scores_by_test = {}
                    for scores_per_test in database._best_scores_per_test_per_island:
                        if scores_per_test is not None:
                            for test_key, test_score in scores_per_test.items():
                                if test_key not in best_scores_by_test or test_score > best_scores_by_test[test_key]:
                                    best_scores_by_test[test_key] = test_score
                    
                    # Log all best test scores at once
                    wandb.log({
                        f'Best scores by input/Input {test_key}': score 
                        for test_key, score in best_scores_by_test.items()
                    } | {'time': current_time})

                # Log scores to CSV
                with open(csv_filename, 'a', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    for island, best_score, avg_score in zip(range(len(best_scores_per_island)), best_scores_per_island, avg_scores_per_island):
                        csv_writer.writerow([f"{current_time:.2f}", island, best_score, avg_score])
                        
                        # Log to wandb
                        wandb.log({
                            f'Best Score/Island {island}': best_score,
                            f'Average Score/Island {island}': avg_score,
                            'time': current_time
                        })
                stats_per_model = database.get_stats_per_model()
                if len(stats_per_model["success_rates"])==1:
                    # For single model case, log rates without model name in path
                    model = next(iter(stats_per_model["success_rates"]))
                    wandb.log({
                        'Rate/Success': stats_per_model["success_rates"][model],
                        'Rate/Parse Failed': stats_per_model["parse_failed_rates"][model], 
                        'Rate/Did Not Run': stats_per_model["did_not_run_rates"][model],
                        'Rate/Unsafe': stats_per_model["unsafe_rates"][model],
                        'time': current_time
                    })
                elif len(stats_per_model["success_rates"])>1:
                    for model in stats_per_model["success_rates"]:
                        wandb.log({
                            f'Rate/Success/{model}': stats_per_model["success_rates"][model],
                            f'Rate/Parse Failed/{model}': stats_per_model["parse_failed_rates"][model],
                            f'Rate/Did Not Run/{model}': stats_per_model["did_not_run_rates"][model],
                            f'Unsafe Rate/{model}': stats_per_model["unsafe_rates"][model],
                            'time': current_time
                        })

                    
                # total_prompt_tokens=
                # Log overall metrics to wandb
                wandb.log({
                    'Overall/Best Score': best_score_overall,
                    'Overall/Average Score': avg_score_overall,
                    'Queue Sizes/Eval Queue': eval_queue_size,
                    'Queue Sizes/Result Queue': result_queue_size,
                    'Programs Processed': database.orig_database._program_counter,# non-mutable type, refer to original database
                    'API Responses': sum(sampler.api_responses for sampler in samplers),
                    'time': current_time
                })

            await asyncio.sleep(logging_info_interval)
            if eval_queue.qsize() > 500:
                logging.warning("Eval queue size exceeded 500. Initiating shutdown.")
                break
            if result_queue.qsize() > 50:
                logging.warning("Result queue size exceeded 50. Initiating shutdown.")
                break
    except asyncio.CancelledError:
        logging.info("Cancellation requested. Shutting down gracefully.")
    finally:
        for task in sampler_tasks:
            task.cancel()
        await asyncio.gather(*sampler_tasks, return_exceptions=True)

        logging.info(f"Length of result_queue upon termination: {result_queue.qsize()}")
        logging.info(f"Length of eval_queue upon termination: {eval_queue.qsize()}")
        # Signal processes to shut down
        for _ in evaluator_processes:
            eval_queue.put(None)
        evaluator_shutdown_timeout = 60
        database_shutdown_timeout = 30
        logging.info("All evaluator workers requested to shut down - waiting %d seconds for them to finish", evaluator_shutdown_timeout)
        # Wait up to 60 seconds total for all evaluator processes to finish
        start_time = time.time()
        remaining_processes = list(evaluator_processes)
        
        while remaining_processes and (time.time() - start_time < evaluator_shutdown_timeout):
            for p in remaining_processes[:]:  # Iterate over copy to allow removal
                if not p.is_alive():
                    logging.info(f"Evaluator process {p.pid} terminated successfully")
                    remaining_processes.remove(p)
                else:
                    p.join(timeout=0.1)  # Small timeout to check frequently
                    
        # Kill any remaining processes
        for p in remaining_processes:
            logging.warning(f"Evaluator process {p.pid} did not terminate within timeout, killing it")
            p.kill()
        logging.info("All evaluator processes have terminated; waiting %d seconds for result queue to drain (length: %d)", database_shutdown_timeout, result_queue.qsize())
        result_queue.put(None)
            
        try:
            await asyncio.wait_for(db_worker, timeout=database_shutdown_timeout)
            logging.info("Database worker finished")
        except asyncio.TimeoutError:
            logging.warning("Database worker timed out during shutdown, final queue length: %d", result_queue.qsize())
            db_worker.cancel()
        db_worker.cancel()

        logging.info(f"Total programs processed: {database.orig_database._program_counter}") ## non-mutable type, refer to original database
        logging.info(f"Best scores per island: {database._best_score_per_island}")

        print_usage_summary(database, run_start_time)
        
        try:
            if wandb.run is not None:
                wandb.finish()
        except Exception as e:
            logging.warning(f"Error while closing wandb: {e}")

        logging.info("Shutdown complete.")
                # Add additional cleanup for wandb

    return database.get_best_programs_per_island()

def print_usage_summary(database,start_time):
    # Log total usage from all samplers grouped by model type
    usage_by_model = {}
    
    # Collect data
    for model_name, stats in database.orig_database.database_worker_counter_dict.items():
        total_data = stats
        
        if model_name not in usage_by_model:
            usage_by_model[model_name] = {
                'sampler_ids': total_data['sampler_ids'],
                'count': len(total_data['sampler_ids']),
                'responses': total_data['eval_success']+total_data['eval_parse_failed']+total_data['eval_did_not_run'],
                'prompt_tokens': total_data['tokens_prompt'],
                'completion_tokens': total_data['tokens_completion'], 
                'total_tokens': total_data['total_tokens'],
                'eval_success': total_data['eval_success'],
                'eval_parse_failed': total_data['eval_parse_failed'],
                'eval_did_not_run': total_data['eval_did_not_run'],
                'eval_unsafe': total_data['eval_unsafe'],
                'counts_each': total_data['counts_each']
            }
    # Print table header
    print("\n\n")
    # Print per-model statistics
    total_samplers = 0
    total_responses = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_success = 0
    total_parse_failed = 0
    total_did_not_run = 0
    total_unsafe = 0
    total_time = time.time() - start_time
    amount_to_pad = max(max(len(key) for key in usage_by_model.keys()), 20)+2
    logging.info(f"Total time: {total_time:.2f} seconds")
    
    # First table - Request statistics
    logging.info("Response Statistics:")
    logging.info("-" * 100)
    logging.info(f"{'Model':<{amount_to_pad}} {'#':<6} {'responses':<8} {'Success':<8} {'No parse':<8} {'Unsafe':<8} {'No Exec':<8} {'Req/s':<6}")
    logging.info("-" * 100)

    for model, stats in usage_by_model.items():
        responses_per_sec = stats['responses'] / total_time
        success_rate = (stats['eval_success'] / stats['responses'] * 100) if stats['responses'] > 0 else 0
        parse_failed = (stats['eval_parse_failed'] / stats['responses'] * 100) if stats['responses'] > 0 else 0
        unsafe_rate = (stats['eval_unsafe'] / stats['responses'] * 100) if stats['responses'] > 0 else 0
        no_exec = (stats['eval_did_not_run'] / stats['responses'] * 100) if stats['responses'] > 0 else 0
        
        logging.info(f"{model:<{amount_to_pad}} {stats['count']:<6} {stats['responses']:<8} {success_rate:>6.1f}% {parse_failed:>6.1f}% {unsafe_rate:>7.1f}% {no_exec:>7.1f}% {responses_per_sec:>5.1f}/s")
        
        total_samplers += stats['count']
        total_responses += stats['responses']
        total_prompt_tokens += stats['prompt_tokens']
        total_completion_tokens += stats['completion_tokens']
        total_success += stats['eval_success']
        total_parse_failed += stats['eval_parse_failed']
        total_did_not_run += stats['eval_did_not_run']
        total_unsafe += stats['eval_unsafe']
    # Print totals for first table
    logging.info("-" * 100)
    total_success_rate = (total_success / total_responses * 100) if total_responses > 0 else 0
    total_parse_failed_rate = (total_parse_failed / total_responses * 100) if total_responses > 0 else 0
    total_no_exec_rate = (total_did_not_run / total_responses * 100) if total_responses > 0 else 0
    total_unsafe_rate = (total_unsafe / total_responses * 100) if total_responses > 0 else 0
    total_responses_per_sec = total_responses / total_time
    logging.info(f"{'TOTAL':<{amount_to_pad}} {total_samplers:<6} {total_responses:<8} {total_success_rate:>6.1f}% {total_parse_failed_rate:>6.1f}% {total_unsafe_rate:>7.1f}% {total_no_exec_rate:>7.1f}% {total_responses_per_sec:>5.1f}/s")

    # Second table - Token statistics
    print("\n")
    logging.info("Token Statistics:")
    logging.info("-" * 120)
    logging.info(f"{'Model':<{amount_to_pad}} {'#':<6} {'Prompt':<10} {'Response':<10} {'Prompt/q':<10} {'Resp/q':<10} {'Tok/s':<8}")
    logging.info("-" * 120)

    for model, stats in usage_by_model.items():
        total_tokens = stats['prompt_tokens'] + stats['completion_tokens']
        tokens_per_sec = total_tokens / total_time
        prompt_per_query = stats['prompt_tokens'] / stats['responses'] if stats['responses'] > 0 else 0
        resp_per_query = stats['completion_tokens'] / stats['responses'] if stats['responses'] > 0 else 0
        
        logging.info(f"{model:<{amount_to_pad}} {stats['count']:<6} {stats['prompt_tokens']:<10} {stats['completion_tokens']:<10} {prompt_per_query:<10.1f} {resp_per_query:<10.1f} {tokens_per_sec:<8.1f}")

    # Print totals for second table
    logging.info("-" * 120)
    total_all_tokens = total_prompt_tokens + total_completion_tokens
    total_tokens_per_sec = total_all_tokens / total_time
    total_prompt_per_query = total_prompt_tokens / total_responses if total_responses > 0 else 0
    total_resp_per_query = total_completion_tokens / total_responses if total_responses > 0 else 0
    logging.info(f"{'TOTAL':<{amount_to_pad}} {total_samplers:<6} {total_prompt_tokens:<10} {total_completion_tokens:<10} {total_prompt_per_query:<10.1f} {total_resp_per_query:<10.1f} {total_tokens_per_sec:<8.1f}")

    # Print per-sampler statistics
    print("\n")
    logging.info("Per-Sampler Response Counts:")
    logging.info("-" * 80)
    for model, stats in usage_by_model.items():
        sampler_ids = sorted(stats['counts_each'].keys())
        counts = [stats['counts_each'][id] for id in sampler_ids]
        logging.info(f"{model} samplers: {sampler_ids}, counts: {counts}")
    logging.info("-" * 80)
