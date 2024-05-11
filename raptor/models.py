import os
import warnings
import requests
import numpy as np
from text import clean
from openai import OpenAI
from omegaconf import OmegaConf
from abc import ABC, abstractmethod
from typing import Dict, Union, List
from sentence_transformers import SentenceTransformer
from tenacity import retry, wait_random_exponential, stop_after_attempt


config = OmegaConf.load('config.yaml')

class EmbeddingModel:
    def __init__(self, model_id: str, dims: int):
        self.model_id = model_id
        self.dims = dims
        self.model = SentenceTransformer(model_id, trust_remote_code=True) # trust_remote_code is needed to use the encode method

    def get_text_embedding(self, text: str, to_numpy: bool = True) -> Union[Dict[str, np.ndarray], Dict[str, List[float]]]:
        """
        Create an embedding for the given text.

        Args:
            text (str): The text to create an embedding for.
            to_numpy (bool, optional): Whether to convert the embedding to a numpy array
        """
        text = clean(text)
        out = self.model.encode(text)
        if not to_numpy:
            out = out.tolist()
        return out


class RerankerModel:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model = SentenceTransformer(model_id, trust_remote_code=True)

    def rerank(self, query: str, candidates: List[str]) -> List[str]:
        """
        Rerank the given candidates based on the query.

        Args:
            query (str): The query to use for reranking.
            candidates (List[str]): The candidates to rerank.
        """
        query_embedding = self.model.encode(query)
        candidate_embeddings = self.model.encode(candidates)
        distances = np.dot(query_embedding, candidate_embeddings.T)
        sorted_indices = np.argsort(distances)[::-1]
        return sorted_indices
        # return [candidates[i] for i in sorted_indices]


class LanguageModel:
    def __init__(self, endpoint, key, model_id="my_llm"):
        self.model_id = model_id
        self.client = OpenAI(api_base=endpoint, api_key=key)

    def _chat_format(self, messages: Dict[str, str]) -> str:
        raise NotImplementedError

    def generate(self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        stop: Optional[List[str]] = [],
        vllm_kwargs: Optional[Dict[str, Any]] = None
        ) -> str:
        """
        Generate text based on the given prompt.

        Args:
            prompt (str): The prompt to use for generation.
            max_tokens (int): The maximum number of tokens to generate.
            temperature (float): The temperature to use for generation.
            stop (List[str], optional): The stop tokens to use for generation. Defaults to None.
            repetition_penalty (float, optional): The repetition penalty to use for generation. Defaults to 1.2.
            vllm_kwargs (Dict[str, Any], optional): Additional keyword arguments to pass to the VLLM API. Defaults to None.
        """

        if vllm_kwargs is None:
            vllm_kwargs = {}
        return self.client.completions.create(
            model=self.model_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            **vllm_kwargs)["choices"][0]["text"]

    def extract_facts(self, text: str):
        text = clean(text)
        msg = {
            "system": "You are a sharp analyst who can extract all the salient facts from the given text.",
            "user": f"List down all the important facts contained in the following text:\n\n{text}",
            "assistant": "Here are the most salient facts in the provided passage:\n\n- "
        }
        prompt = self._chat_format(msg)
        response = "- " + self.generate(prompt, max_tokens=config.tree_builder.parent_text_tokens, temperature=0.6)
        return response

    def extract_questions(self, text: str):
        text = clean(text)
        msg = {
            "system": "You are a sharp analyst who can extract all the salient facts from the given text.",
            "user": f"Write a list of questions that can be answered by the following text:\n\n{text}",
            "assistant": "Here are the questions whose answers are in the provided passage:\n\n- "
        }
        prompt = self._chat_format(msg)
        response = "- " + self.generate(prompt, max_tokens=config.tree_builder.parent_text_tokens, temperature=0.6)
        return response

    def write_passage(self, facts: str):
        msg = {
            "system": "You are a helpful assistant who can write a passage based on the given facts.",
            "user": f"Write a short passage based on the following facts:\n\n{facts}",
            "assistant": "Here is the passage:\n\n"
        }
        prompt = self._chat_format(msg)
        response = self.generate(prompt, max_tokens=config.tree_builder.parent_text_tokens, temperature=0.6)
        return response




class Llama3(LanguageModel):
    def __init__(self, endpoint, key, model_id="llama3"):
        super().__init__(endpoint, key, model_id)

    def _chat_format(self, messages: Dict[str, str]) -> str:
        """
        Format the given message for chat generation.

        Args:
            message (Dict[str, str]): The message to format.
        """
        if not messages["user"]:
            raise ValueError("User message cannot be empty")
        if not messages["system"]:
            messages["system"] = "You are a helpful assistant"
        if not messages["assistant"]:
            messages["assistant"] = ""

        def encode(role, content):
            return f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}"

        return "".join(encode(role, content) for role, content in messages.items())
