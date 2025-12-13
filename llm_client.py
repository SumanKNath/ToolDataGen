from nt import environ
import sys
from dotenv import load_dotenv
from azure.identity import get_bearer_token_provider, AzureCliCredential
from openai import AzureOpenAI


class LLMClient:
    def __init__(self):
        load_dotenv()
        self.client = AzureOpenAI(
            api_version= sys.environ["OPENAI_API_VERSION"],
            azure_endpoint=  sys.environ["OPENAI_ENDPOINT"],
            azure_ad_token_provider=get_bearer_token_provider(
                AzureCliCredential(),
                sys.environ["OPENAI_SCOPE"]
            )
        )

    def get_response_text(self, user_prompt:str, system_prompt:str=None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        response = self.client.chat.completions.create(
            model=sys.environ["OPENAI_TEXT_MODEL"],
            messages=messages
        )
        return response.choices[0].message.content