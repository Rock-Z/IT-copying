import os
import random
from datasets import load_dataset

cache_dir = "../data"
os.environ['HF_DATASETS_CACHE'] = cache_dir

#dataset = load_dataset("allenai/dolma", name="v1_6-sample", cache_dir=cache_dir, split="train")
dataset = load_dataset("RealTimeData/bbc_news_alltime", "2024-05", cache_dir=cache_dir, split="train")

# samples some random rows and print them
sampled_indices = random.choices(list(range(len(dataset))), k = 5)
for index in sampled_indices:
    print(dataset[index])
    # print number of words
    print(len(dataset[index]['content'].split()))

# print the number of rows in the dataset
print(len(dataset))
