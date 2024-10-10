from mistralai import Mistral
import anthropic
import openai
import os
import google.generativeai as genai
import asyncio
def get_model(model_name):
    if "codestral" in model_name.lower() or "mistral" in model_name.lower():
        return MistralModel
    elif "gpt" in model_name.lower() or "o1" in model_name.lower():
        return OpenAIModel
    elif "claude" in model_name.lower():
        return AnthropicModel
    elif "gemini" in model_name.lower():
        return GeminiModel
    else:
        raise ValueError(f"Unsupported model name: {model_name}. Have a look in __main__.py and models.py to add support for this model.")

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
        max_retries = 5
        base_delay = 1  # Start with a 1-second delay
        for attempt in range(max_retries):
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
                if isinstance(e, Mistral.APIError) and e.status_code == 429:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        print(f"Rate limit exceeded. Retrying in {delay} seconds...")
                        await asyncio.sleep(delay)
                    else:
                        print("Max retries reached. Unable to get a response.")
                        return None
                else:
                    print(f"Error in MistralModel.prompt: {e}")
                    return None
            
            # Check for INFO log about 429 status
            if "HTTP/1.1 429 Too Many Requests" in str(chat_response):
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    print(f"Rate limit (429) detected. Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    print("Max retries reached. Unable to get a response.")
                    return None
            
        return None  # If we've exhausted all retries

class AnthropicModel:
    def __init__(self, model_name="claude-3-sonnet-20240620", top_p=0.9, temperature=0.7):
        self.key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
        self.client = anthropic.AsyncAnthropic(api_key=self.key)
        self.model = model_name
        self.top_p = top_p
        self.temperature = temperature

    async def prompt(self, prompt_text):
        max_retries = 5
        base_delay = 1  # Start with a 1-second delay
        for attempt in range(max_retries):
            try:
                chat_response = await self.client.messages.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt_text}
                    ],
                    max_tokens=4096,
                    top_p=self.top_p,
                    temperature=self.temperature
                )
                if chat_response is not None:
                    return chat_response.content[0].text
                else:
                    print("Error: No response received from Anthropic API")
                    return None
            except Exception as e:
                if isinstance(e, anthropic.RateLimitError):
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        print(f"Rate limit exceeded. Retrying in {delay} seconds...")
                        await asyncio.sleep(delay)
                    else:
                        print("Max retries reached. Unable to get a response.")
                        return None
                else:
                    print(f"Error in AnthropicModel.prompt: {e}")
                    return None
            
            # Check for INFO log about 429 status
            if "HTTP/1.1 429 Too Many Requests" in str(chat_response):
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    print(f"Rate limit (429) detected. Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    print("Max retries reached. Unable to get a response.")
                    return None
            
        return None  # If we've exhausted all retries

class OpenAIModel:
    def __init__(self, model_name="gpt-4o", top_p=0.9, temperature=0.7):
        self.key = os.environ.get("OPENAI_API_KEY")
        if not self.key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.client = openai.AsyncOpenAI(api_key=self.key)  # Changed to AsyncOpenAI
        self.model = model_name
        self.top_p = top_p
        self.temperature = temperature

    async def prompt(self, prompt_text):
        max_retries = 5
        base_delay = 1  # Start with a 1-second delay
        for attempt in range(max_retries):
            try:
                chat_response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt_text
                        }
                    ],
                    max_tokens=4096,
                    top_p=self.top_p,
                    temperature=self.temperature
                )
                if chat_response is not None:
                    return chat_response.choices[0].message.content
                else:
                    print("Error: No response received from OpenAI API")
                    return None
            except Exception as e:
                if isinstance(e, openai.RateLimitError):
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        print(f"Rate limit exceeded. Retrying in {delay} seconds...")
                        await asyncio.sleep(delay)
                    else:
                        print("Max retries reached. Unable to get a response.")
                        return None
                else:
                    print(f"Error in OpenAIModel.prompt: {e}")
                    return None
            #OPENAI seems to not have a 429 status code, so no need to check for that - it retries on its own
        return None  # If we've exhausted all retries


class GeminiModel:
    def __init__(self, model_name="gemini-1.5-flash", top_p=0.9, temperature=0.7):
        self.key = os.environ.get("GOOGLE_API_KEY")
        if not self.key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        genai.configure(api_key=self.key)
        self.model = genai.GenerativeModel(model_name)
        self.top_p = top_p
        self.temperature = temperature

    async def prompt(self, prompt_text):
        max_retries = 5
        base_delay = 1  # Start with a 1-second delay
        for attempt in range(max_retries):
            try:
                response = await self.model.generate_content_async(
                    prompt_text,
                    generation_config={
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "max_output_tokens": 4096,
                    }
                )
                if response is not None:
                    return response.text
                else:
                    print("Error: No response received from Gemini API")
                    return None
            except Exception as e:
                if isinstance(e, genai.RateLimitError):
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        print(f"Rate limit exceeded. Retrying in {delay} seconds...")
                        await asyncio.sleep(delay)
                    else:
                        print("Max retries reached. Unable to get a response.")
                        return None
                else:
                    print(f"Error in GeminiModel.prompt: {e}")
                    return None
        return None  # If we've exhausted all retries

