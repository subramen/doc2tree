import os
import torch
import warnings
import requests
import numpy as np
from text import clean
from openai import OpenAI
from omegaconf import OmegaConf
from abc import ABC, abstractmethod
from typing import Dict, Union, List, Optional, Any
from sentence_transformers import SentenceTransformer
from tenacity import retry, wait_random_exponential, stop_after_attempt


config = OmegaConf.load('raptor/config.yaml')

class EmbeddingModel:
    def __init__(self, model_id: str, dims: int, batch_size: int):
        self.model_id = model_id
        self.dims = dims
        self.model = SentenceTransformer(model_id, trust_remote_code=True) # trust_remote_code is needed to use the encode method
        self.batch_size = batch_size

    def get_text_embedding(self, text: Union[List[str], str]) -> Union[Dict[str, np.ndarray], Dict[str, List[float]]]:
        """
        Create an embedding for the given text.

        Args:
            text (str): The text to create an embedding for.
            to_numpy (bool, optional): Whether to convert the embedding to a numpy array
        """
        if isinstance(text, str):
            text = [text]
        text = [clean(t) for t in text]
        batches = (text[i:i+self.batch_size] for i in range(0, len(text), self.batch_size))
        result = []
        with torch.inference_mode():
            for b in batches:
                result.extend(self.model.encode(b))
        return result


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
        sorted_indices = np.argsort(distances[0])[::-1]
        return sorted_indices


class LanguageModel:
    def __init__(self, endpoint, key, model_id, batch_size: int):
        self.model_id = model_id
        self.client = OpenAI(base_url=endpoint, api_key=key)
        self.batch_size = batch_size

    def _prompt_format(self, messages: Dict[str, str]) -> str:
        raise NotImplementedError

    def generate(self,
        prompt: Union[List[str], str],
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

        if isinstance(prompt, str):
            prompt = [prompt]

        batches = (prompt[i:i+self.batch_size] for i in range(0, len(prompt), self.batch_size))
        result = []
        for b in batches:
            response = self.client.completions.create(
                model=self.model_id,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                **vllm_kwargs).choices
            result.extend([r.text for r in response])
        return result

    def extract_facts(self, text: Union[List[str], str]):
        def _get_message(text):
            return {
                "system": "You are a sharp analyst who can extract all the salient information from the given text.",
                "user": f"Paraphrase the following text in a neutral third-person tone. Don't add anything that is not mentioned in the given text.\n\n{text}",
                "assistant": "Here is a concise version of the text:\n\n"
            }

        if isinstance(text, str):
            text = [text]
        text = [clean(t) for t in text]
        messages = [_get_message(t) for t in text]
        prompts = [self._prompt_format(m) for m in messages]
        response = self.generate(prompts, max_tokens=config.tree_builder.parent_text_tokens, temperature=0.6)
        return response


    def extract_questions(self, text: Union[List[str], str]):
        def _get_message(text):
            return {
                "system": "You are a sharp analyst who can extract all the salient information from the given text.",
                "user": f"Write a list of questions that can be answered by the following text:\n\n{text}",
                "assistant": "Here are the questions whose answers are in the provided passage:\n\n- "
            }

        if isinstance(text, str):
            text = [text]
        text = [clean(t) for t in text]
        messages = [_get_message(t) for t in text]
        prompts = [self._prompt_format(m) for m in messages]
        response = self.generate(prompts, max_tokens=config.tree_builder.parent_text_tokens, temperature=0.6)
        return response

    def write_passage(self, facts: Union[List[str], str]):
        def _get_message(text):
            return {
                "system": "You are a fluent author who can craft well-written essays from a list of facts.",
                "user": f"Write a short passage based on the following facts:\n\n{text}",
                "assistant": "Here is the passage:\n\n"
            }
        if isinstance(facts, str):
            facts = [facts]
        messages = [_get_message(t) for t in facts]
        prompts = [self._prompt_format(m) for m in messages]
        response = self.generate(prompts, max_tokens=config.tree_builder.parent_text_tokens, temperature=0.6)


    def write_response(self, question: str, context: str):
        msg = {
            "system": "You are provided with a question and various references. Your task is to answer the question succinctly, using the fewest words possible. If the references do not contain the necessary information to answer the question, respond with 'I don't know'. There is no need to explain the reasoning behind your answer",
            # You are an attentive analyst and fluent communicator. Given a question and some relevant context, write a response. Ensure your response is only based on the provided context. ",
            "user": f"Context: {context}\n\nQuestion: {question}",
            "assistant": "Here is the response:\n\n"
        }
        prompt = self._prompt_format(msg)
        response = self.generate(prompt, max_tokens=4096, temperature=0.4)
        return response


class Llama3(LanguageModel):

    def _prompt_format(self, messages: Dict[str, str]) -> str:
        """
        Format the given message for chat generation.

        Args:
            message (Dict[str, str]): The message to format.
        """
        if "user" not in messages.keys():
            raise ValueError("User message cannot be empty")
        messages["system"] = messages.get("system", "You are a helpful assistant")
        messages["assistant"] =  messages.get("assistant", "")

        def encode(role, content):
            return f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}"

        return "".join(encode(role, messages[role]) for role in ["system", "user", "assistant"])
