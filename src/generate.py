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
from datasets import load_dataset,Dataset

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
    """Split the prompts and keep the first n words.

    Args:
        prompts (dict[int, str]): A dictionary of prompts where the keys are integers and the values are strings.
        first_n (int, optional): The number of words to split each prompt. Defaults to 250.

    Returns:
        dict[int, str]: A new dictionary of prompts where each prompt is split into a specified number of words.
    """
    return {key: " ".join(prompt.split()[:first_n]) for key, prompt in prompts.items()}

@gin.configurable
def load_datasets(data_sources : dict[str, Callable[[str, str], dict[int, str]]] = 
                  {"bbc": read_prompts_bbc, "arxiv": read_prompts_arxiv}) -> dict[str, dict[int, str]]:
    """
    Load datasets from the specified data sources.

    Args:
        data_sources (dict[str, Callable[[str, str], dict[int, str]]], optional): 
        A dictionary mapping data source names to functions that read and process 
        the data. Defaults to {"bbc": read_prompts_bbc, "arxiv": read_prompts_arxiv}.

    Returns:
        dict[str, dict[int, str]]: A dictionary containing the loaded datasets, where
        the keys are the data source names and the values are dictionaries mapping 
        prompt IDs to prompt strings.
    """
    datasets = {}
    for key, value in data_sources.items():
        datasets[key] = split_prompts(value())
    return datasets

@gin.configurable
def load_model_and_tokenizer(model_kwargs : dict[str, str], tokenizer_kwargs : dict[str, str] = {}):
    """Load the model and tokenizer from Hugging Face.

    Args:
        model_kwargs (dict[str, str]): A dictionary of keyword arguments to pass to the model.
        tokenizer_kwargs (dict[str, str]): A dictionary of keyword arguments to pass to the tokenizer.

    Returns:
        AutoModelForCausalLM, AutoTokenizer: The model and tokenizer objects.
    """
    if "pretrained_model_name_or_path" not in tokenizer_kwargs:
        tokenizer_kwargs["pretrained_model_name_or_path"] = model_kwargs["pretrained_model_name_or_path"]

    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(**tokenizer_kwargs)
    return model, tokenizer

@gin.configurable
def run_generation(model : AutoModelForCausalLM, 
                   tokenizer : AutoTokenizer,
                   prompt : str = PROMPT, 
                   generate_kwargs : dict[str, str] = {}) -> str:

    inputs = tokenizer.encode(prompt, return_tensors="pt")
    inputs = inputs.to("cuda")
    with torch.no_grad():
        outputs = model.generate(inputs, **generate_kwargs)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text

@gin.configurable
def save_generated_data(outputs : dict[str, dict[int, str]], 
                        path_constructor = lambda key: f"~/project/IT-Copying/data/OLMo_7b_{key}_generated"):
    """Save the generated data to disk.

    Args:
        outputs (dict[str, dict[int, str]]): A dictionary containing the generated data.
            The keys represent the data categories, and the values are dictionaries
            where the keys are the original indices and the values are the generated texts.
        path_constructor (function, optional): A function that constructs the path to save the data.
            It takes a key as input and returns the path. Defaults to a lambda function that constructs
            the path based on the key.

    Returns:
        None
    """
    for key, value in outputs.items():
        constructor_dict = {"original_idx": list(value.keys()), "generated_text": list(value.values())}
        dataset = Dataset.from_dict(constructor_dict)
        dataset.save_to_disk(path_constructor(key))

def main():

    print("Transformers version:", transformers.__version__)
    print("PyTorch version:", torch.__version__)
    assert torch.cuda.is_available(), "CUDA available: False"
    # print all available devices
    print("Available GPUs:", torch.cuda.device_count())

    # load configs
    load_gin_configs(["configs/generate_default.gin"] + sys.argv[1:])
    set_seed()

    # load model
    model, tokenizer = load_model_and_tokenizer()

    prompts = load_datasets()
    outputs = defaultdict(dict)
    for key, value in prompts.items():
        with tqdm.tqdm(value, desc = f"Generating for dataset {key}") as pbar:
            for idx, prompt in value.items():
                outputs[key][idx] = run_generation(model, tokenizer, prompt)
                pbar.update(1)

    save_generated_data(outputs)

if __name__ == "__main__":
    main()
