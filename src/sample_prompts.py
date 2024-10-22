import os
import random
import numpy as np
from datasets import load_dataset

cache_dir = "../data"
os.environ['HF_DATASETS_CACHE'] = cache_dir
N = 500

def main():
    datasets = {"bbc": load_dataset("RealTimeData/bbc_news_alltime", "2024-05", cache_dir=cache_dir, split="train"),
                "arxiv": load_dataset("RealTimeData/arxiv_alltime", "2024-05", cache_dir=cache_dir, split="train")}

    for dataset_name, dataset in datasets.items():
        # Filter out those less than 500 words
        idx_to_remove = []
        if dataset_name == "bbc":
            idx_to_remove = list(filter(lambda x: len(dataset[x]['content'].split()) < 500, range(len(dataset))))
        elif dataset_name == "arxiv":
            idx_to_remove = list(filter(lambda x: len(dataset[x]['text'].split()) < 500, range(len(dataset))))
        
        # Print how much is left
        print(f"Dataset: {dataset_name}; entries longer than 500 words: {len(dataset) - len(idx_to_remove)}, original size: {len(dataset)}")

        # Randomly sample N rows
        remaining_idx = list(filter(lambda x: x not in idx_to_remove, range(len(dataset))))
        samples = np.random.choice(list(range(len(dataset) - len(idx_to_remove))), size = N, replace=False)
        sampled_idx = np.array([remaining_idx[i] for i in samples], dtype=int)

        # Save the sampled idx 
        np.savetxt(f"../configs/{dataset_name}_sampled_idx.txt", sampled_idx, fmt="%4d")

if __name__ == "__main__":
    main()