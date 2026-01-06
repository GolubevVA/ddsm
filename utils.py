import os

import torch

def get_best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def load_dotenv(path = ".env"):
    try:
        with open(path) as f:
            for line in f:
                key, value = line.strip().split('=', 1)
                os.environ[key] = value
    except FileNotFoundError:
        pass
