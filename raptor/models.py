import os
from abc import ABC, abstractmethod
import requests
from sentence_transformers import SentenceTransformer
from tenacity import retry, wait_random_exponential, stop_after_attempt
import numpy as np

class BaseEmbeddingModel(ABC):
    def __init__(self, name: str):
        self.name = name
        self.dims = None
    @abstractmethod
    def create_embedding(self, text):
        pass

class JinaEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_id="jinaai/jina-embeddings-v2-base-en") -> None:
        super().__init__(name="jina")
        self.model = SentenceTransformer(model_id, trust_remote_code=True) # trust_remote_code is needed to use the encode method
        self.dims = 768

    def create_embedding(self, text, as_list=True) -> np.ndarray:
        out = self.model.encode(text)
        if as_list:
            return out.tolist()
        return out



class BaseSummarizationModel(ABC):
    @abstractmethod
    def summarize(self, context, max_tokens=150):
        pass

class AzureLlamaSummarizationModel(BaseSummarizationModel):
    def __init__(self, model_name='llama-2-70b-chat') -> None:
        self.endpoint = "https://Llama-2-70b-chat-suraj-demo-serverless.eastus2.inference.ai.azure.com/v1/completions"
        self.key = os.environ['KEY_70B']


    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=1000, stop_sequence=None):
        message = f"Rewrite the following into a concise paragraph. Do not write anything that is not stated here. Do not miss including anything interesting, surprising or otherwise relevant to the reader.\n\nTEXT: {context}\n\n"

        payload = {
            "prompt": message,
            "max_tokens": max_tokens,
            "temperature": 0.6,
            "stop": stop_sequence,
        }
        headers = {'Content-Type':'application/json', 'Authorization':(self.key)}
        response = requests.post(self.endpoint, json=payload, headers=headers)

        if response.status_code == 200:
            return response.json()['choices'][0]['text']
        else:
            raise Exception(f"Request failed with status code {response.status_code}")
