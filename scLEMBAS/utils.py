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