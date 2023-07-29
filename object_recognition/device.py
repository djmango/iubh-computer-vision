import torch
import torch.backends.mps
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Check that MPS is available
device: torch.device = torch.device("cpu")

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
else:
    print("CUDA is not available, using CPU instead")
