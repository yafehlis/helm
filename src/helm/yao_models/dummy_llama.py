import sys
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

llama7b_name = 'decapoda-research/llama-7b-hf'
llama7b_name_path = "/home/yafehlis/.cache/huggingface/hub/model-7B"

my_model = LlamaForCausalLM.from_pretrained(llama7b_name)
my_model.save_pretrained(llama7b_name_path)

tokenizer = LlamaTokenizer.from_pretrained(llama7b_name)
tokenizer.save_pretrained(llama7b_name_path)


#my_model = LlamaForCausalLM.from_pretrained(llama7b_name_path)
#tokenizer = LlamaTokenizer.from_pretrained(llama7b_name_path)


texts = [
    'In which country is Hamburg?',
    'How are you doing today?',
    'It was a dark and stormy night.',]


TEMPERATURE = 0.5
MAX_NEW_TOKENS = 10
batch_size = 1

def __get_temperature_distribution(logits, temperature=TEMPERATURE):
    return torch.softmax(logits / temperature, dim=-1)


def _sample_fn(logits, temperature=TEMPERATURE):
    probs = __get_temperature_distribution(logits, temperature)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def sample_model(model,
                 input_ids,
                 nb_tokens=MAX_NEW_TOKENS,
                 temperature=TEMPERATURE):
    for _ in range(nb_tokens):
        outputs = model(input_ids)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = _sample_fn(next_token_logits, temperature)
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=-1)
    return input_ids

if __name__ == "__main__":

    nb_tokens = 0
    for text in texts:
        input_ids = tokenizer(text, return_tensors="pt").input_ids

        input_ids = torch.stack([input_ids[0]] * batch_size).to(my_model.device)

        generated_ids = sample_model(my_model, input_ids)
        nb_tokens += generated_ids.shape[1] - input_ids.shape[1]
        print("Completion: ", tokenizer.decode(
            generated_ids[0], skip_special_tokens=True))
