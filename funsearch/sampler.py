# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Class for sampling new programs."""
from collections.abc import Collection, Sequence

import llm
import numpy as np
import time

from funsearch import evaluator
from funsearch import programs_database
import asyncio

import os
from mistralai import Mistral
class MistralModel:
    def __init__(self, model_name="mistral-small-latest", top_p=0.9, temperature=0.7):
        self.key = os.environ.get("MISTRAL_API_KEY")
        if not self.key:
            raise ValueError("MISTRAL_API_KEY environment variable is not set")
        self.client = Mistral(api_key=self.key)
        self.model = model_name
        self.top_p = top_p
        self.temperature = temperature

    async def prompt(self, prompt_text):
        try:
            chat_response = await self.client.chat.complete_async(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt_text
                    }
                ],
                top_p=self.top_p,
                temperature=self.temperature
            )
            if chat_response is not None:
                return chat_response.choices[0].message.content
            else:
                print("Error: No response received from Mistral API")
                return None
        except Exception as e:
            print(f"Error in MistralModel.prompt: {e}")
            return None
class LLM:
  """Language model that predicts continuation of provided source code."""

  def __init__(self, samples_per_prompt: int, model, log_path=None) -> None:
    self._samples_per_prompt = samples_per_prompt
    self.model = model
    self.prompt_count = 0
    self.log_path = log_path

  async def _draw_sample(self, prompt: str, label: int) -> str:
    """Returns a predicted continuation of `prompt`."""
    #start_time = time.time()
    #response = await asyncio.to_thread(self.model.prompt, prompt)
    response = await self.model.prompt(prompt)
    #end_time = time.time()
    #print(f"Awaited model.prompt for {end_time - start_time:.2f} seconds")
    #print(f"Prompt count: {self.prompt_count}, Label: {label}")
    self._log(prompt, response, self.prompt_count, label)
    self.prompt_count += 1
    return response

  async def draw_samples(self, prompt: str, label: int) -> Collection[str]:
    """Returns multiple predicted continuations of `prompt`."""
    return [await self._draw_sample(prompt, label) for _ in range(self._samples_per_prompt)]

  def _log(self, prompt: str, response: str, index: int, label: int):
    if self.log_path is not None:
      with open(self.log_path / f"prompt_{label}_{index}.log", "a") as f:
        f.write(prompt)
      with open(self.log_path / f"response_{label}_{index}.log", "a") as f:
        f.write(str(response))


class Sampler:
  """Node that samples program continuations and sends them for analysis."""

  def __init__(
      self,
      database: programs_database.ProgramsDatabase or multi_testing.AsyncProgramsDatabase,
      evaluators: Sequence[evaluator.Evaluator],
      model: LLM,
      label = 0
  ) -> None:
    self._database = database
    self._evaluators = evaluators
    self._llm = model
    self.label = label
    self.api_calls = 0

  async def sample(self, prompt, eval_queue):
    """Continuously gets prompts, samples programs, sends them for analysis."""
    #prompt = await self._database.get_prompt()
    samples = await self._llm.draw_samples(prompt.code, self.label)
    #print('done drawing samples')
    # This loop can be executed in parallel on remote evaluator machines.
    self.api_calls += len(samples)
    for sample in samples:
      #chosen_evaluator = np.random.choice(self._evaluators)
      eval_queue.put((sample, prompt.island_id, prompt.version_generated, prompt.island_version))
      #chosen_evaluator.analyse(
      #    sample, prompt.island_id, prompt.version_generated, self.label)
