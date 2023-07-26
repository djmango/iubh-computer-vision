import torch
import torch.backends.mps
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Check that MPS is available
device: torch.device = torch.device("cpu")

if torch.backends.mps.is_available() and False: # Disabling because for some reason the memory usage is crazy
    device = torch.device("mps")
    print("MPS is available")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("MPS is not available, using CUDA instead")
else:
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

