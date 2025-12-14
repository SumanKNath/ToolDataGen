import os
import json
from pydoc import text
import re
from dotenv import load_dotenv
from azure.identity import get_bearer_token_provider, AzureCliCredential
from openai import AzureOpenAI

class LLMCLient:
    def __init__(self):
        load_dotenv()  # Load environment variables from a .env file if present

        self.llm_client = AzureOpenAI(
            api_version=os.getenv("OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
            azure_ad_token_provider=get_bearer_token_provider(
                AzureCliCredential(),
                os.getenv("OPENAI_SCOPE", "https://cognitiveservices.azure.com/.default")
            )
        )

    def get_response_text(self, user_prompt:str, system_prompt:str =None) -> str:
        response = self.llm_client.chat.completions.create(
            model=os.getenv("OPENAI_TEXT_MODEL"),
            messages=[
                {"role": "system", "content": system_prompt if system_prompt else "You are a helpful assistant."},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    
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
    prompt = "Generate a JSON array of two objects, each with 'name' and 'age' fields."
    json_output = client.get_response_json(prompt)
    print(json_output)
