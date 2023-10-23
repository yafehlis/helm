from typing import List, Dict
from transformers import LlamaForCausalLM, LlamaTokenizer

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
from helm.yao-models.dummy_llama import sample_model

class AMDClient(Client):
    """Implements some "models" that just generate silly things quickly just to debug the infrastructure."""

    def __init__(self, cache_config: CacheConfig):
        self.cache = Cache(cache_config)
        

        self.llama7b_name_path = "/workspace/.cache/huggingface/hub/model-7B"

        self.my_model = LlamaForCausalLM.from_pretrained(self.llama7b_name_path)
        self.tokenizer = LlamaTokenizer.from_pretrained(llama7b_name_path)

    def make_request(self, request: Request) -> RequestResult:
        raw_request = {
            "engine": request.model_engine,
            "prompt": request.prompt,
            "n": request.num_completions,
        }

        
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids
        input_ids = torch.stack([input_ids[0]] * batch_size).to(my_model.device)

        generated_ids = sample_model(self.my_model, input_ids)
        
        generated_ids = self.tokenizer.decode(generated_ids, skip_special_tokens=True)


            completions = [
                Sequence(
                    text=generated_ids,
                    logprob=0.0,
                    tokens=[],
                )
                for text, logprob in response["completions"].items()
            ]


        return RequestResult(
            success=True,
            cached=False,
            request_time=0,
            request_datetime=response.get("request_datetime"),
            completions=completions,
            embedding=[],
        )

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        raise NotImplementedError("Use the HuggingFaceClient to tokenize.")

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        raise NotImplementedError

 
