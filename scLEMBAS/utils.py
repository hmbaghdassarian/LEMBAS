import torch
import numpy as np

def np_to_torch(arr: np.array, dtype: torch.float32, device: str = 'cpu'):
    """Convert a numpy array to a torch.tensor

    Parameters
    ----------
    arr : np.array
        
    dtype : torch.dtype, optional
        datatype to store values in torch, by default torch.float32
    device : str
        whether to use gpu ("cuda") or cpu ("cpu"), by default "cpu"
    """
    return torch.tensor(arr, dtype=dtype, device = device)

def set_seeds(seed: int=888):
    """Sets random seeds for torch operations.

    Parameters
    ----------
    seed : int, optional
        seed value, by default 888
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
