import torch

def main():
    print("Hello from it-copying!")
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

if __name__ == "__main__":
    main()