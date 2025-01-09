from mistralai import Mistral
import anthropic
import openai
import os
import asyncio
import google.generativeai as genai
import logging
import time
import shortuuid

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
    def __init__(self, model_name="codestral-latest", top_p=0.9, temperature=0.7, keynum=0, timeout=10, retries=10, id=None):
        self.id = str(shortuuid.uuid()) if id is None else str(id)
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
        self.model = model_name
        self.top_p = top_p
        self.temperature = temperature
        self.counter = 0
        self.timeout = timeout
        self.retries = retries
        logging.info(f"Created {self.provider} {self.model} sampler {self.id} using {keyname}")
        #logging.info(f"Ensure temperature defaults are correct for this model??")

    async def complete(self, prompt_text):
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

        return chat_response

    async def prompt(self, prompt_text):
        max_retries = 10
        base_timeout = 10  # Initial timeout for exponential backoff
        base_of_exponential_backoff = 1.1
        begin = time.time()
        for attempt in range(max_retries):
            try:
                start = time.time()
                logging.debug(f"prompt:start:{self.model}:{self.id}:{self.counter}:{attempt}")
                task = self.complete(prompt_text)
                chat_response = await asyncio.wait_for(task, timeout=base_timeout * (base_of_exponential_backoff ** attempt))
                end = time.time()
                logging.debug(f"prompt:end:{self.model}:{self.id}:{self.counter}:{attempt}:{end-start:.3f}")
                if chat_response is not None:
                    logging.debug(f"prompt:success:{self.model}:{self.id}:{self.counter}:{attempt}:{end-begin:.3f}",)
                    self.counter += 1
                    return chat_response
                logging.warning(f"prompt:error-empty:{self.model}:{self.id}:{self.counter}:{attempt}:{end-start:.3f}")
            except Exception as e:
                end = time.time()
                if '429' in str(e):#if it's a rate limit error, not a big issue
                    logging.debug(f"prompt:error:exception:{e}:{self.model}:{self.id}:{self.counter}:{attempt}:{end-start:.3f}")
                else:
                    logging.warning(f"prompt:error:exception:{e}:{self.model}:{self.id}:{self.counter}:{attempt}:{end-start:.3f}")
                if attempt < max_retries - 1:
                    sleep_time = base_timeout * (base_of_exponential_backoff ** attempt)
                    logging.debug(f"prompt:sleep:{self.model}:{self.id}:{self.counter}:{attempt}:{end-start:.3f}:{sleep_time:.3f}")
                    await asyncio.sleep(sleep_time)
                    end = time.time()
                    logging.debug(f"prompt:awoke:{self.model}:{self.id}:{self.counter}:{attempt}:{end-start:.3f}")
        logging.error(f"prompt:error-fatal:{self.model}:{self.id}:{self.counter}:{attempt}:{max_retries} attempts failed")
        return None  # If we've exhausted all retries


