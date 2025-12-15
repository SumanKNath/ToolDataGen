import os
import json
from pydoc import text
import re
import time
from dotenv import load_dotenv
from azure.identity import get_bearer_token_provider, AzureCliCredential
from openai import AzureOpenAI, APITimeoutError, APIConnectionError, RateLimitError

class LLMCLient:
    def __init__(self, timeout: float = 60.0, max_retries: int = 3):
        """
        Initialize the LLM client.
        
        Args:
            timeout: Request timeout in seconds (default: 60)
            max_retries: Maximum number of retry attempts (default: 3)
        """
        load_dotenv()  # Load environment variables from a .env file if present

        self.timeout = timeout
        self.max_retries = max_retries
        
        self.llm_client = AzureOpenAI(
            api_version=os.getenv("OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
            azure_ad_token_provider=get_bearer_token_provider(
                AzureCliCredential(),
                os.getenv("OPENAI_SCOPE", "https://cognitiveservices.azure.com/.default")
            ),
            timeout=timeout
        )

    def get_response_text(self, user_prompt: str, system_prompt: str = None) -> str:
        """
        Get text response from the LLM with timeout and retry logic.
        
        Retries up to max_retries times on:
        - Timeout errors
        - Connection errors
        - Rate limit errors (with exponential backoff)
        
        Args:
            user_prompt: The user's prompt/question
            system_prompt: Optional system prompt to set context
            
        Returns:
            The LLM's response text
            
        Raises:
            Exception: If all retry attempts fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                response = self.llm_client.chat.completions.create(
                    model=os.getenv("OPENAI_TEXT_MODEL"),
                    messages=[
                        {"role": "system", "content": system_prompt if system_prompt else "You are a helpful assistant."},
                        {"role": "user", "content": user_prompt}
                    ],
                    timeout=self.timeout
                )
                return response.choices[0].message.content.strip()
                
            except APITimeoutError as e:
                last_exception = e
                wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                print(f"  Timeout on attempt {attempt + 1}/{self.max_retries}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                
            except APIConnectionError as e:
                last_exception = e
                wait_time = 2 ** attempt
                print(f"  Connection error on attempt {attempt + 1}/{self.max_retries}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                
            except RateLimitError as e:
                last_exception = e
                # For rate limits, use longer backoff
                wait_time = (2 ** attempt) * 5  # 5, 10, 20 seconds
                print(f"  Rate limited on attempt {attempt + 1}/{self.max_retries}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                
            except Exception as e:
                # For other exceptions, don't retry
                raise e
        
        # All retries exhausted
        raise Exception(f"Failed after {self.max_retries} attempts. Last error: {last_exception}")
    
    def get_response_json(self, user_prompt: str, system_prompt: str = None) -> list:
        # Get text response from the LLM
        text_response = self.get_response_text(user_prompt, system_prompt)
        
        # First try to extract JSON from code blocks
        json_match = re.search(r'```json\s*(.*?)\s*```', text_response, re.DOTALL)
        json_text = json_match.group(1).strip() if json_match else text_response
        
        # Parse the JSON
        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError:
            parsed = None
       
        return parsed
    
if __name__ == "__main__":
    client = LLMCLient()
    prompt = "Generate a JSON array of two objects, each with 'name' and 'age' and 'GPT-version' fields. Be creative. "
    prompt = "what is the name and version of your model? Respond in JSON format with fields 'model_name' and 'model_version'."
    json_output = client.get_response_json(prompt)
    print(json_output)
