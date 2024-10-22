import os
import sys
import gin
import tqdm
from collections.abc import Callable
from collections import defaultdict

# Set cache dir
os.environ['HF_HOME'] = "~/project/IT-Copying/models"
cache_dir = "~/project/IT-Copying/data"
os.environ['HF_DATASETS_CACHE'] = cache_dir

import transformers
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from utils import load_gin_configs, set_seed

# Default prompt
PROMPT = "In my younger and more vulnerable years my father gave me some advice that I've been turning over in my mind ever since"

@gin.configurable
def read_prompts_bbc(date : str = "2024-05", sampled_idx : str = "configs/bbc_sampled_idx.txt") -> dict[int, str]:
    """Reads the BBC prompts from the dataset.

    Args:
        date (str, optional): month split. Defaults to "2024-05".
        sampled_idx (str, optional): path for sampled indices to select from. Defaults to "data/bbc_sampled_idx.txt".

    Returns:
        dict[int, str]: list of prompts
    """
    bbc_idx = np.loadtxt(sampled_idx)
    bbc_dataset = load_dataset("RealTimeData/bbc_news_alltime", date, split="train")
    bbc_dataset = bbc_dataset.select(indices=bbc_idx)
    bbc_prompts = bbc_dataset["content"]
    bbc_prompts = {bbc_idx[i]: prompt for i, prompt in enumerate(bbc_prompts)}

    return bbc_prompts

@gin.configurable
def read_prompts_arxiv(date : str = "2024-05", sampled_idx : str = "configs/arxiv_sampled_idx.txt") -> dict[int, str]:
    """Reads the arXiv prompts from the dataset.

    Args:
        date (str, optional): month split. Defaults to "2024-05".
        sampled_idx (str, optional): path for sampled indices to select from. Defaults to "data/arxiv_sampled_idx.txt".

    Returns:
        dict[int, str]: list of prompts
    """
    arxiv_idx = np.loadtxt(sampled_idx)
    arxiv_dataset = load_dataset("RealTimeData/arxiv_alltime", date, split="train")
    arxiv_dataset = arxiv_dataset.select(indices=arxiv_idx)
    arxiv_prompts = arxiv_dataset["text"]
    arxiv_prompts = {arxiv_idx[i]: prompt for i, prompt in enumerate(arxiv_prompts)}

    return arxiv_prompts

@gin.configurable
def split_prompts(prompts : dict[int, str], first_n : int = 250) -> dict[int, str]:
    return {key: " ".join(prompt.split()[:first_n]) for key, prompt in prompts.items()}

@gin.configurable
def load_datasets(data_sources : dict[str, Callable[[str, str], dict[int, str]]] = 
                  {"bbc": read_prompts_bbc, "arxiv": read_prompts_arxiv}) -> dict[str, dict[int, str]]:
    datasets = {}
    for key, value in data_sources.items():
        datasets[key] = split_prompts(value())
    return datasets

@gin.configurable
def run_generation(prompt : str = PROMPT, 
                   model_name : str = "allenai/OLMo-7B-hf",
                   tokenizer_name : str = None,
                   generate_kwargs : dict[str, str] = {"min_length": 250, "max_length": 250}):

    model = AutoModelForCausalLM.from_pretrained(model_name)
    if tokenizer_name is None: 
        tokenizer_name = model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model.to("cuda")

    inputs = tokenizer.encode(prompt, return_tensors="pt")
    inputs = inputs.to("cuda")
    outputs = model.generate(inputs, **generate_kwargs)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text

@gin.configurable
def save_generated_data(outputs : dict[str, dict[int, str]], 
                        path_constructor = lambda key: f"../data/{key}_generated"):
    for key, value in outputs.items():
        dataset = load_dataset("text", data_files={"train": value})
        dataset.save_to_disk(path_constructor(key))

def main():

    print("Transformers version:", transformers.__version__)
    print("PyTorch version:", torch.__version__)
    assert torch.cuda.is_available(), "CUDA available: False"

    # load configs
    load_gin_configs(["configs/generate_default.gin"] + sys.argv[1:])
    set_seed()

    prompts = load_datasets()
    outputs = defaultdict(dict)
    for key, value in prompts.items():
        with tqdm.tqdm(value, desc="Generating for dataset:{key}") as pbar:
            for idx, prompt in value.items():
                outputs[key][idx] = run_generation(prompt)
                pbar.update(1)

    save_generated_data(outputs)

if __name__ == "__main__":
    main()
