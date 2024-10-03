import threading
import queue
import time
from typing import Any, Callable
import jax.random as random

#generic multi-threading code for testing- implemented three worker types: sampler, evaluator and a database for storing programs
#THIS IS NOT FUNCTIONAL YET

class ProgramDatabase:
    def __init__(self):
        self.programs = {}
        self.lock = threading.Lock()

    def add_program(self, name: str, program: Callable):
        with self.lock:
            self.programs[name] = program
            print(f"Program '{name}' added to the database.")

    def get_program(self, name: str) -> Callable:
        with self.lock:
            program = self.programs.get(name)
            if program:
                print(f"Program '{name}' retrieved from the database.")
            else:
                print(f"Program '{name}' not found in the database.")
            return program

class Sampler(threading.Thread):
    def __init__(self, sampler_id: int, program_queue: queue.Queue, proposal_queue: queue.Queue, rng_key):
        super().__init__()
        self.sampler_id = sampler_id
        self.program_queue = program_queue
        self.proposal_queue = proposal_queue
        self.rng_key = rng_key
        self.running = True

    def run(self):
        while self.running:
            try:
                program_name = self.program_queue.get(timeout=1)
                print(f"Sampler {self.sampler_id} is sampling program '{program_name}'.")
                # Simulate sampling new implementation
                new_program = self.sample_program(program_name)
                self.proposal_queue.put((program_name, new_program))
                self.program_queue.task_done()
            except queue.Empty:
                continue

    def sample_program(self, program_name: str) -> Callable:
        # Placeholder for actual sampling logic using JAX
        def new_implementation(*args, **kwargs):
            return f"New implementation of {program_name}"
        return new_implementation

    def stop(self):
        self.running = False

class Evaluator(threading.Thread):
    def __init__(self, evaluator_id: int, proposal_queue: queue.Queue, evaluation_results: queue.Queue, rng_key):
        super().__init__()
        self.evaluator_id = evaluator_id
        self.proposal_queue = proposal_queue
        self.evaluation_results = evaluation_results  # Fixed typo: evaluation_queue -> evaluation_results
        self.rng_key = rng_key
        self.running = True
        # This code initializes an Evaluator thread with:
        # - A unique ID
        # - A queue for receiving program proposals to evaluate
        # - A queue for sending back evaluation results
        # - A random number generator key for reproducibility
        # - A flag to control the thread's execution

    def run(self):
        while self.running:
            try:
                program_name, program = self.proposal_queue.get(timeout=1)
                print(f"Evaluator {self.evaluator_id} is evaluating program '{program_name}'.")
                score = self.evaluate_program(program)
                self.evaluation_results.put((program_name, score))
                self.proposal_queue.task_done()
            except queue.Empty:
                continue

    def evaluate_program(self, program: Callable) -> float:
        # Placeholder for actual evaluation logic
        time.sleep(0.5)  # Simulate evaluation time
        return random.uniform(self.rng_key, (1,)).item()

    def stop(self):
        self.running = False

class FunSearchSystem:
    def __init__(self, num_samplers: int, num_evaluators: int):
        self.program_db = ProgramDatabase()
        self.program_queue = queue.Queue()
        self.proposal_queue = queue.Queue()
        self.evaluation_results = queue.Queue()
        self.samplers = []
        self.evaluators = []
        key = random.PRNGKey(0)

        # Initialize samplers
        for i in range(num_samplers):
            sampler = Sampler(
                sampler_id=i,
                program_queue=self.program_queue,
                proposal_queue=self.proposal_queue,
                rng_key=random.fold_in(key, i)
            )
            self.samplers.append(sampler)

        # Initialize evaluators
        for i in range(num_evaluators):
            evaluator = Evaluator(
                evaluator_id=i,
                proposal_queue=self.proposal_queue,
                evaluation_results=self.evaluation_results,
                rng_key=random.fold_in(key, i + num_samplers)
            )
            self.evaluators.append(evaluator)

    def start(self):
        for sampler in self.samplers:
            sampler.start()
        for evaluator in self.evaluators:
            evaluator.start()

    def add_program(self, name: str, program: Callable):
        self.program_db.add_program(name, program)
        self.program_queue.put(name)

    def get_evaluation_results(self):
        results = []
        while not self.evaluation_results.empty():
            results.append(self.evaluation_results.get())
        return results

    def stop(self):
        for sampler in self.samplers:
            sampler.stop()
        for evaluator in self.evaluators:
            evaluator.stop()
        for sampler in self.samplers:
            sampler.join()
        for evaluator in self.evaluators:
            evaluator.join()

if __name__ == "__main__":
    # Initialize the FunSearch system with 2 samplers and 3 evaluators
    system = FunSearchSystem(num_samplers=2, num_evaluators=3)
    system.start()

    # Add initial programs to the database
    system.add_program("priority", lambda x: x)

    # Let the system run for a short period
    time.sleep(5)

    # Retrieve and print evaluation results
    results = system.get_evaluation_results()
    for program_name, score in results:
        print(f"Program '{program_name}' scored {score}.")

    # Stop the system
    system.stop()