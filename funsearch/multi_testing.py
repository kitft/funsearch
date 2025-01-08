import asyncio
# from concurrent.futures import ThreadPoolExecutor
# from typing import List
import logging
import multiprocessing
from multiprocessing import Process, Queue
import time

from funsearch import programs_database, evaluator, sampler, config as config_lib
# import pathlib

import csv
# from datetime import datetime
import os
import numpy as np

import wandb

class AsyncProgramsDatabase(programs_database.ProgramsDatabase):
    def __init__(self, database: programs_database.ProgramsDatabase):
        # Initialize with the same attributes as the original database
        for attr_name, attr_value in database.__dict__.items():
            setattr(self, attr_name, attr_value)
        self.orig_database = database

    async def get_prompt(self) -> programs_database.Prompt:
        #print("Getting prompt")
        return self.orig_database.get_prompt()

    async def register_program(self, program, island_id, scores_per_test, island_version=None, model=None):
        self.orig_database.register_program(program, island_id, scores_per_test, island_version, model)

# class AsyncEvaluator(evaluator.Evaluator):
#     async def async_analyse(self, sample: str, island_id: int | None, version_generated: int | None) -> None:
#         self.analyse(sample, island_id, version_generated)

# class AsyncSampler(sampler.Sampler):
#     async def async_sample(self, prompt, eval_queue, db_queue):
#         return self.sample(prompt, eval_queue, db_queue)

def evaluator_process(eval_queue: Queue, result_queue: Queue, config: config_lib.Config, multitestingconfig: config_lib.MultiTestingConfig, id: int):
    evaluator_instance = evaluator.Evaluator(
        AsyncProgramsDatabase(config.programs_database),
        multitestingconfig.sandbox_class(base_path=multitestingconfig.log_path, id=id),
        multitestingconfig.template,
        multitestingconfig.function_to_evolve,
        multitestingconfig.function_to_run,
        multitestingconfig.parsed_inputs,
        id=id
    )
    #evaluator_process is synchronous, and initialised on alternate processes
    #time.sleep(id*0.1)
    while True:
        try:
            task = eval_queue.get(timeout=1)
            if task is None:
                break
            sample, island_id, version_generated, island_version, model = task
            result = evaluator_instance.analyse(sample, island_id, version_generated, island_version, model)
            logging.info(f"Evaluator {id}, island {island_id}, version generated {version_generated}, island version {island_version}: Eval Queue size: {eval_queue.qsize()}, Result Queue size: {result_queue.qsize()}, Result: {bool(result)}")
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
                logging.info("database worker exiting loop")
                break
            # else:
            #    logging.info("reduced result queue size to %d"%(result_queue.qsize()))
            new_function, island_id, scores_per_test, island_version, model = result
            await database.register_program(new_function, island_id, scores_per_test, island_version, model)
        except multiprocessing.queues.Empty:
            await asyncio.sleep(0.1)
    logging.info("database worker end")

async def sampler_worker(sampler: sampler.Sampler, eval_queue: multiprocessing.Queue, database: AsyncProgramsDatabase, config: config_lib.Config):
    #wait a random amount of time to avoid synchronisation issues and API ratelimits
    await asyncio.sleep(np.random.rand()*config.num_samplers*0.5)
    while True:
        #logging.info('Sampling with sampler %d', sampler.label)
        prompt = await database.get_prompt()
        await sampler.sample(prompt, eval_queue)
        # Adaptive sleep based on queue size
        queue_size = eval_queue.qsize()
        sleep_time = min(0.1 * (queue_size), 5)  # Cap at 5 seconds
        if sleep_time > 0.5:
            logging.info('Slowed down sampling to %f seconds', sleep_time)
        await asyncio.sleep(sleep_time)

async def runAsync(config: config_lib.Config, database: AsyncProgramsDatabase, multitestingconfig: config_lib.MultiTestingConfig):
    #num_cores = min(multiprocessing.cpu_count(), config.num_evaluators,2)

    num_cores = min(multiprocessing.cpu_count()-1, config.num_evaluators)
    logging.info("Number of cores/evaluators to be used: %d", num_cores)
    eval_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()

    # Start evaluator processes
    evaluator_processes = []
    for id in range(num_cores):
        p = Process(target=evaluator_process, args=(eval_queue, result_queue, config, multitestingconfig, id))
        p.start()
        evaluator_processes.append(p)

    # Create samplers
    samplers = [sampler.Sampler(database, None, multitestingconfig.lm[i], i) for i in range(config.num_samplers)]

    # Create and start tasks
    db_worker = asyncio.create_task(database_worker(result_queue, database))

    # Initial evaluation
    initial = multitestingconfig.template.get_function(multitestingconfig.function_to_evolve).body
    eval_queue.put((initial, None, None, None, None))
    time.sleep(3)
    logging.info("Initialising %d samplers"%(len(samplers)))
    sampler_tasks = [asyncio.create_task(sampler_worker(s, eval_queue, database,config)) for s in samplers]

    timestamp = multitestingconfig.timestamp
    os.makedirs("./data/scores", exist_ok=True)
    csv_filename = f"./data/scores/scores_log_{timestamp}.csv"
    
    # Initialize wandb
    wandb.init(
        project="funsearch",
        name=f"run_{timestamp}",
        config={
            "num_cores": num_cores,
            "num_samplers": config.num_samplers,
            "run_duration": config.run_duration,
            "num_islands": len(database._islands)
        }
    )

    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Time', 'Island', 'Best Score', 'Average Score'])

    try:
        start_time = time.time()
        while time.time() - start_time < config.run_duration:
            current_time = time.time() - start_time
            if current_time >= 5 * (current_time // 5):
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

                logging.info(f"Time: {current_time:.2f}s, Eval Queue size: {eval_queue_size}, Result Queue size: {result_queue_size}")

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
                
                # Log overall metrics to wandb
                wandb.log({
                    'Overall/Best Score': best_score_overall,
                    'Overall/Average Score': avg_score_overall,
                    'Queue Sizes/Eval Queue': eval_queue_size,
                    'Queue Sizes/Result Queue': result_queue_size,
                    'API Calls': sum(sampler.api_calls for sampler in samplers),
                    'time': current_time
                })

            await asyncio.sleep(5)
            if eval_queue.qsize() > 500:
                logging.warning("Eval queue size exceeded 500. Initiating shutdown.")
                break
            if result_queue.qsize() > 50:
                logging.warning("Result queue size exceeded 50. Initiating shutdown.")
                break
    except asyncio.CancelledError:
        logging.info("Cancellation requested. Shutting down gracefully.")
    finally:
        # Cancel all tasks
        for task in sampler_tasks:
            task.cancel()
        logging.info(f"Length of result_queue upon termination: {result_queue.qsize()}")
        await asyncio.sleep(1)
        db_worker.cancel()
        logging.info(f"Length of result_queue after shutting down: {result_queue.qsize()}")
        # Signal processes to shut down
        for _ in evaluator_processes:
            eval_queue.put(None)
        result_queue.put(None)
        logging.info("All tasks cancelled, workers signaled to shut down, sleeping 1 second")
        await asyncio.sleep(1)

        logging.info(f"Total programs processed: {database._program_counter}")
        logging.info(f"Best scores per island: {database._best_score_per_island}")
        
        # Close wandb
        wandb.finish()

        logging.info("Shutdown complete.")

    return database.get_best_programs_per_island()
