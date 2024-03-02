"""
Helper functions for running and training the model. 
"""

import torch
import numpy as np
import time

def set_seeds(seed: int=888):
    """Sets random seeds for torch operations.

    Parameters
    ----------
    seed : int, optional
        seed value, by default 888
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_lr(e: int, max_iter: int, max_height: float = 1e-3, 
             start_height: float=1e-5, end_height: float=1e-5, 
             peak: int = 1000):
    """Calculates learning rate for a given iteration during training.

    Parameters
    ----------
    e : int
        the current iteration
    max_iter : int
        the maximum number of training iterations
    max_height : float, optional
        tuning parameters for learning for the first 95% of iterations, by default 1e-3
    start_height : float, optional
        tuning parameter for learning rate before peak iterations, by default 1e-5
    end_height : float, optional
        tuning parameter for learning rate afer peak iterations, by default 1e-5
    peak : int, optional
        the first # of iterations to calculate lr on (should be less than 95% 
        of max_iter), by default 1000

    Returns
    -------
    lr : float
        the learning rate
    """

    phase_length = 0.95 * max_iter
    if e<=peak:
        effective_e = e/peak
        lr = (max_height-start_height) * 0.5 * (np.cos(np.pi*(effective_e+1))+1) + start_height
    elif e<=phase_length:
        effective_e = (e-peak)/(phase_length-peak)
        lr = (max_height-end_height) * 0.5 * (np.cos(np.pi*(effective_e+2))+1) + end_height
    else:
        lr = end_height
    return lr

def initialize_progress(max_iter: int):
    """Track various stats of the progress of training the model.

    Parameters
    ----------
    max_iter : int
        the maximum number of training iterations

    Returns
    -------
    stats : dict
        a dictionary of progress statistics
    """
    stats = {}
    stats['start_time'] = time.time()
    stats['end_time'] = 0
    stats['iter_time'] = np.nan*np.ones(max_iter)
    
    stats['loss'] = np.nan*np.ones(max_iter)
    stats['lossSTD'] = np.nan*np.ones(max_iter)
    stats['eig'] = np.nan*np.ones(max_iter)
    stats['eig_sigma'] = np.nan*np.ones(max_iter)

    stats['test'] = np.nan*np.ones(max_iter)
    stats['rate'] = np.nan*np.ones(max_iter)
    stats['violations'] = np.nan*np.ones(max_iter)

    return stats
