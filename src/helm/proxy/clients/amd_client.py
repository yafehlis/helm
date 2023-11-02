from typing import List, Dict
from transformers import LlamaForCausalLM, LlamaTokenizer

import torch 

from helm.common.cache import Cache, CacheConfig
from helm.common.request import Request, RequestResult, Sequence, Token
from helm.common.tokenization_request import (
    DecodeRequest,
    DecodeRequestResult,
    TokenizationRequest,
    TokenizationRequestResult,
    TokenizationToken,
)
from .client import Client, wrap_request_time
from helm.yao_models.dummy_llama import sample_model

class AMDClient(Client):
    """Implements some "models" that just generate silly things quickly just to debug the infrastructure."""

    def __init__(self, cache_config: CacheConfig):
        self.cache = Cache(cache_config)

        self.batch_size = 1
        self.llama7b_name_path = "/workspace/.cache/huggingface/hub/model-7B"

        self.my_model = LlamaForCausalLM.from_pretrained(self.llama7b_name_path, device_map='balanced')
        self.tokenizer = LlamaTokenizer.from_pretrained(self.llama7b_name_path)

    def make_request(self, request: Request) -> RequestResult:
        tokenizer = self.tokenizer
        
        encoded = tokenizer(request.prompt, return_tensors="pt").input_ids
        input_ids = torch.stack([encoded[0]] * self.batch_size).to(self.my_model.device)

        prompt_length = len(input_ids[0])
        tokens = sample_model(self.my_model, input_ids)
        
        if request.echo_prompt is False:
            output = tokenizer.decode(tokens[0][prompt_length:], skip_special_tokens=True)
        else:
            output = tokenizer.decode(tokens[0], skip_special_tokens=True)

        generated_tokens = []
        for token in tokens[0][prompt_length:]: # This is different than lit_gpt_client.py
            generated_tokens.append(Token(text=tokenizer.decode(token), logprob=0, top_logprobs={}))
        
        completions = [Sequence(text=output, logprob=0, tokens=generated_tokens)]


        return RequestResult(
            success=True,
            cached=False,
            request_time=0,
            completions=completions,
            embedding=[],
        )

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        raise NotImplementedError("Use the HuggingFaceClient to tokenize.")

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        raise NotImplementedError
