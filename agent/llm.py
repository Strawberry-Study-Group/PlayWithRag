from openai import OpenAI
import requests
import anthropic

class LLM:
    """
    Class for handling text generation requests to different language models and APIs.
    """
    
    def __init__(self, config):
        """
        Initialize the LLM class with the provided configuration.
        
        Args:
            config (dict): Configuration dictionary containing API settings.
        """
        self.llm_provider = config.get("llm_provider", "openai")
        if self.llm_provider == "openai":
            self.client = OpenAI(api_key=config.get("api_key"))


        self.api_key = config.get("api_key")
        self.model = config.get("model", "text-davinci-003")
        self.max_tokens = config.get("max_tokens", 100)
        self.temperature = config.get("temperature", 0.7)
        self.n = config.get("n", 1)
        self.stop = config.get("stop", None)
    
    def completion(self, prompt):
        """
        Generate text completion based on the provided prompt.
        
        Args:
            prompt (str): The input prompt for text generation.
        
        Returns:
            str: The generated text completion.
        """
        if self.llm_provider == "openai":
            return self._openai_completion(prompt)
        elif self.llm_provider == "huggingface":
            return self._huggingface_completion(prompt)
        elif self.llm_provider == "anthropic":
            return self._anthropic_completion(prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
    def _openai_completion(self, prompt):
        """
        Generate text completion using the OpenAI API.
        
        Args:
            prompt (str): The input prompt for text generation.
        
        Returns:
            str: The generated text completion.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": prompt},
            ]
            )
        return response.choices[0].message.content
    
    def _huggingface_completion(self, prompt):
        """
        Generate text completion using the Hugging Face API.
        
        Args:
            prompt (str): The input prompt for text generation.
        
        Returns:
            str: The generated text completion.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "n": self.n,
            "stop": self.stop,
        }
        response = requests.post(
            "https://api-inference.huggingface.co/models/text-generation",
            headers=headers,
            json=data,
        )
        response.raise_for_status()
        return response.json()[0]["generated_text"].strip()
    
    def _anthropic_completion(self, prompt):
        """
        Generate text completion using the Anthropic API (Claude).
        
        Args:
            prompt (str): The input prompt for text generation.
        
        Returns:
            str: The generated text completion.
        """
        client = anthropic.Anthropic(api_key=self.api_key)
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=messages
        )
        
        return response.content[0].text.strip()