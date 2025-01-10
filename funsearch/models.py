from mistralai import Mistral
import anthropic
import openai
import os
import asyncio
import google.generativeai as genai
import logging
import time
import shortuuid
from typing import Optional
import json
from datetime import datetime
from pathlib import Path
import atexit
import httpx
import aiohttp
from funsearch import logging_stats

system_prompt="""You are a state-of-the-art python code completion system that will be used as part of a genetic algorithm.
You will be given a list of functions, and you should improve the incomplete last function in the list.
1. Make only small changes but be sure to make some change.
2. Try to keep the code short and any comments concise.
3. Your response should be an implementation of the function priority_v# (where # is the current iteration number); do not include any examples or extraneous functions.
4. You may use numpy.
The code you generate will be appended to the user prompt and run as a python program.
"""

def get_model_provider(model_name):
    if "/" in model_name.lower():
        return "openrouter"
    elif "codestral" in model_name.lower() or "mistral" in model_name.lower():
        return "mistral"
    elif "gpt" in model_name.lower() or "o1" in model_name.lower():
        return "openai"
    elif "claude" in model_name.lower():
        return "anthropic"
    elif "gemini" in model_name.lower():
        return "google"
    elif "llama" in model_name.lower() or "deepseek" in model_name.lower() or "qwen" in model_name.lower() or "mixtral" in model_name.lower():
        return "deepinfra"
    else:
        raise ValueError(f"Unsupported model name: {model_name}. Have a look in __main__.py and models.py to add support for this model.")

class LLMModel:
    def __init__(
        self, 
        model_name="codestral-latest", 
        top_p=0.9, 
        temperature=0.7, 
        keynum=0, 
        timeout=10, 
        retries=10, 
        id=None,
        #config=None  # Add config parameter
        log_path=None
    ):
        self.id = str(shortuuid.uuid()) if id is None else str(id)
        
        # Get problem identifier from config
        #problem_identifier = config.problem_name + "_" + config.timestamp if config else "default"
        
        # Initialize usage logger with problem identifier and model ID
  
        
        if '%' in model_name:
            s = model_name.split('%')
            assert len(s)==2
            self.provider = s[0]
            model_name = s[1]
        else:
            self.provider = get_model_provider(model_name)
        keyname = self.provider.upper() + "_API_KEY"
        if keynum > 0:
            keyname += str(keynum)
        self.key = os.environ.get(keyname)
        if not self.key:
            raise ValueError(f"{keyname} environment variable is not set")
        if self.provider == "mistral":
            self.client = Mistral(api_key=self.key)
        elif self.provider == "anthropic":
            self.client = anthropic.AsyncAnthropic(api_key=self.key)
        elif self.provider == "openai":
            self.client = openai.AsyncOpenAI(api_key=self.key)
        elif self.provider == "google":
            genai.configure(api_key=self.key)
            self.client = genai.GenerativeModel(model_name,system_instruction=system_prompt)
        elif self.provider == "deepinfra":
            self.client = openai.AsyncOpenAI(api_key=self.key,base_url="https://api.deepinfra.com/v1/openai/")
        elif self.provider == "openrouter":
            self.client = openai.AsyncOpenAI(api_key=self.key,base_url="https://openrouter.ai/api/v1")
            self.data_request = openai.AsyncOpenAI(api_key=self.key,base_url="https://openrouter.ai/api/v1/generation")
        self.model = model_name
        self.top_p = top_p
        self.temperature = temperature
        self.counter = 0
        self.timeout = timeout
        self.retries = retries
        self.log_stats = True
        self.log_detailed_stats = False
        logging.info(f"Created {self.provider} {self.model} sampler {self.id} using {keyname}")
        #logging.info(f"Ensure temperature defaults are correct for this model??")

    # async def log_usage(self, usage_stats: UsageStats):
    #     """Append usage statistics to the log file"""
    #     with open(self.usage_log_file, 'a') as f:
    #         f.write(usage_stats.to_json_line() + '\n')


    async def get_openrouter_stats(self, generation_id: str) -> dict:
        """Fetch detailed stats for an OpenRouter generation including token counts and costs
        Does not work until some time after the generation is complete.
        We therefore do not use this for now."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://openrouter.ai/api/v1/generation?id={generation_id}",
                    headers={
                        "Authorization": f"Bearer {self.key}"
                    }
                ) as response:
                    data = (await response.json())["data"]

            return {
                "id": data["id"],
                "model": data["model"], 
                "tokens_prompt": data["tokens_prompt"],
                "tokens_completion": data["tokens_completion"],
                "native_tokens_prompt": data["native_tokens_prompt"], 
                "native_tokens_completion": data["native_tokens_completion"],
                "total_cost": data["total_cost"],
                "generation_time": data["generation_time"]
            }
        except Exception as e:
            logging.warning(f"Failed to fetch OpenRouter stats for generation {generation_id}: {e}")
            return None

    async def complete(self, prompt_text):
        usage_stats = None
        if self.provider == "mistral":
            response = await self.client.chat.complete_async(
                model=self.model,
                messages=[
                    { "role": "system", "content": system_prompt },
                    { "role": "user", "content": prompt_text },
                ],
                top_p=self.top_p,
                temperature=self.temperature
            )
            if hasattr(response, 'status_code') and response.status_code == 429:
                raise httpx.HTTPStatusError("Rate limit exceeded: 429", request=None, response=response)
            chat_response = None if response is None else response.choices[0].message.content
            if self.log_stats:
                usage_stats = logging_stats.UsageStats(
                    id=response.id,
                    model=response.model,
                    provider=self.provider,
                    total_tokens=response.usage.total_tokens if hasattr(response, 'usage') else None,
                    tokens_prompt=response.usage.prompt_tokens if hasattr(response, 'usage') else None,
                    tokens_completion=response.usage.completion_tokens if hasattr(response, 'usage') else None,
                    instance_id=self.id,
                    prompt=prompt_text,
                    response=chat_response
                )

        elif self.provider == "anthropic":
            response = await self.client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=[{ "role": "user", "content": prompt_text }],
                max_tokens=4096,
                top_p=self.top_p,
                temperature=self.temperature
            )
            chat_response = None if response is None else response.content[0].text
            if hasattr(response, 'status_code') and response.status_code == 429:
                raise httpx.HTTPStatusError("Rate limit exceeded: 429", request=None, response=response)
            if self.log_stats:
                usage_stats = logging_stats.UsageStats(
                    id=response.id,
                    model=response.model,
                    provider=self.provider,
                    total_tokens=response.usage.total_tokens if hasattr(response, 'usage') else None,
                    tokens_prompt=response.usage.prompt_tokens if hasattr(response, 'usage') else None,
                    tokens_completion=response.usage.completion_tokens if hasattr(response, 'usage') else None,
                    instance_id=self.id,
                    prompt=prompt_text,
                    response=chat_response
                )
        elif self.provider in ["openai", "deepinfra", "openrouter"]:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    { "role": "system", "content": system_prompt },
                    { "role": "user", "content": prompt_text },
                ],
                max_tokens=4096,
                top_p=self.top_p,
                temperature=self.temperature
            )
            chat_response = None if response is None else response.choices[0].message.content
            if hasattr(response, 'status_code') and response.status_code == 429:
                raise httpx.HTTPStatusError("Rate limit exceeded: 429", request=None, response=response)
            if self.log_stats:
                id = response.id
                ## TODO - log cost of response and number of tokens used
                if response:
                    # Log basic usage stats immediately
                    usage_stats = logging_stats.UsageStats(
                        id=response.id,
                        model=response.model,
                        provider=self.provider,
                        total_tokens=response.usage.total_tokens if hasattr(response, 'usage') else None,
                        tokens_prompt=response.usage.prompt_tokens if hasattr(response, 'usage') else None,
                        tokens_completion=response.usage.completion_tokens if hasattr(response, 'usage') else None,
                        instance_id=self.id,
                        prompt=prompt_text,
                        response=chat_response
                    )
                    
                    # Fetch detailed stats asynchronously
                    detailed_stats = False#await self.get_openrouter_stats(response.id)
                    if self.log_detailed_stats and detailed_stats:# and 'data' in detailed_stats:
                        data = detailed_stats['data']
                        usage_stats.total_cost = data.get('total_cost')
                        usage_stats.generation_time = data.get('generation_time')
                    
                    #await self.usage_logger.log_usage(usage_stats)
        
        elif self.provider == "google":
            response = await self.client.generate_content_async(
                prompt_text,
                generation_config={
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "max_output_tokens": 4096,
                }
            )
            chat_response = None if response is None else response.text
            if hasattr(response, 'status_code') and response.status_code == 429:
                raise httpx.HTTPStatusError("Rate limit exceeded: 429", request=None, response=response)
        else:
            return None

        return chat_response, usage_stats

    async def prompt(self, prompt_text):
        max_retries = 10
        base_timeout = 30  # Initial timeout for exponential backoff
        base_of_exponential_backoff = 1.1
        begin = time.time()
        for attempt in range(max_retries):
            try:
                start = time.time()
                logging.debug(f"prompt:start:{self.model}:{self.id}:{self.counter}:{attempt}")
                task = self.complete(prompt_text)
                chat_response, usage_stats = await asyncio.wait_for(task, timeout=base_timeout * (base_of_exponential_backoff ** attempt))
                end = time.time()
                logging.debug(f"prompt:end:{self.model}:{self.id}:{self.counter}:{attempt}:{end-start:.3f}")
                if chat_response is not None:
                    logging.debug(f"prompt:success:{self.model}:{self.id}:{self.counter}:{attempt}:{end-begin:.3f}",)
                    self.counter += 1
                    return chat_response,usage_stats
                logging.warning(f"prompt:error-empty:{self.model}:{self.id}:{self.counter}:{attempt}:{end-start:.3f}")
            except Exception as e:
                end = time.time()
                if '429' in str(e):#if it's a rate limit error, not a big issue
                    logging.debug(f"prompt:error:exception_ratelimit:{e}:{self.model}:{self.id}:{self.counter}:{attempt}:{end-start:.3f}")
                else:
                    logging.warning(f"prompt:error:exception_other:{e}:{self.model}:{self.id}:{self.counter}:{attempt}:{end-start:.3f}")
                if attempt < max_retries - 1:
                    sleep_time = base_timeout * (base_of_exponential_backoff ** attempt)
                    logging.debug(f"prompt:sleep:{self.model}:{self.id}:{self.counter}:{attempt}:{end-start:.3f}:{sleep_time:.3f}")
                    await asyncio.sleep(sleep_time)
                    end = time.time()
                    logging.debug(f"prompt:awoke:{self.model}:{self.id}:{self.counter}:{attempt}:{end-start:.3f}")
        logging.error(f"prompt:error-fatal:{self.model}:{self.id}:{self.counter}:{attempt}:{max_retries} attempts failed")
        return None, None  # If we've exhausted all retries

