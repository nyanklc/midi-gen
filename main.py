import torch

def main():
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"cuda device: {torch.cuda.get_device_name(torch.cuda.device)}")

if __name__ == "__main__":
    main()
