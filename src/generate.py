import os
import transformers
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set cache dir
os.environ['HF_HOME'] = "../models"

# Set random seeds
torch.manual_seed(0)
np.random.seed(0)

# Default prompt
PROMPT = "In my younger and more vulnerable years my father gave me some advice that I've been turning over in my mind ever since"

def run_generation(prompt : str = PROMPT):
    model_name = "allenai/OLMo-7B-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to("cuda")

    inputs = tokenizer.encode(prompt, return_tensors="pt")
    inputs = inputs.to("cuda")
    outputs = model.generate(inputs, max_length=100)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Generated text:", generated_text)

def main():
    print("Transformers version:", transformers.__version__)
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    run_generation()

if __name__ == "__main__":
    main()
