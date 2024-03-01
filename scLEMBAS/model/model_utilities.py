"""
Helper functions for building the model.
"""

import numpy as np
import pandas as pd
import scipy
from scipy.sparse.linalg import eigs
from torch import nn

def format_network(net: pd.DataFrame, 
                   weight_label: str = 'mode_of_action', 
                   stimulation_label: str = 'stimulation', 
                   inhibition_label: str = 'inhibition') -> pd.DataFrame:
    """Formats the standard network file format to that needed by `SignalingModel.parse_network`

    Parameters
    ----------
    net : pd.DataFrame
        signaling network adjacency list with the following columns:
            - `weight_label`: whether the interaction is stimulating (1) or inhibiting (-1) or unknown (0.1). Exclude non-interacting (0) nodes. 
            - `stimulation_label`: binary whether an interaction is stimulating (1) or [not stimultaing or unknown] (0)
            - `inhibition_label`: binary whether an interaction is inhibiting (1) or [not inhibiting or unknown] (0)
    weight_label : str, optional
        converts `stimulation_label` and `inhibition_label` to a single column of stimulating (1), inhibiting (-1), or
        unknown (0.1), by default 'mode_of_action'
    stimulation_label : str, optional
        column name of stimulating interactions, see `net`, by default 'stimulation'
    inhibition_label : str, optional
        column name of inhibitory interactions, see `net`, by default 'inhibition'

    Returns
    -------
    formatted_net : pd.DataFrame
        the same dataframe with the additional `weight_label` column
    """
    if net[(net[stimulation_label] == 1) & (net[inhibition_label] == 1)].shape[0] > 0:
        raise ValueError('An interaction can either be stimulating (1,0), inhibition (0,1) or unknown (0,0)')
    
    formatted_net = net.copy()
    formatted_net[weight_label] = np.zeros(net.shape[0])
    formatted_net.loc[formatted_net[stimulation_label] == 1, weight_label] = 1
    formatted_net.loc[formatted_net[inhibition_label] == 1, weight_label] = -1
    
    #ensuring that lack of known MOA does not imply lack of representation in scipy.sparse.find(A)
    formatted_net[weight_label] = formatted_net[weight_label].replace(0, 0.1)
    formatted_net[weight_label] = formatted_net[weight_label].replace(np.nan, 0.1)

    return formatted_net

def get_spectral_radius(weights: nn.parameter.Parameter):
    """_summary_

    Parameters
    ----------
    weights : nn.parameter.Parameter
        the interaction weights

    Returns
    -------
    spectral_radius : np.ndarray
        a single element numpy array representing the denominator of the scaling factor for weights 
    """
    A = scipy.sparse.csr_matrix(weights.detach().numpy())
    eigen_value, _ = eigs(A, k = 1) # first eigen value
    spectral_radius = np.abs(eigen_value)
    return spectral_radius