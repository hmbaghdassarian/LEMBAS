"""
Train the signaling model.
"""
from typing import Dict, List, Union

import numpy as np
import torch

sclembas_path = '/home/hmbaghda/Projects/scLEMBAS'
import sys
import os
sys.path.insert(1, os.path.join(sclembas_path))
import scLEMBAS.utilities as utils

LR_PARAMS = {'max_iter': 5000, 'learning_rate': 2e-3}
OTHER_PARAMS = {'batch_size': 10, 'noise_level': 10, 'gradient_noise_level': 1e-9}
REGULARIZATION_PARAMS = {'param_lambda_L2': 1e-6, 'moa_lambda_L1': 0.1, 'ligand_lambda_L2': 1e-5, 'uniform_lambda_L2': 1e-4, 
                   'uniform_max': (1/1.2), 'spectral_loss_factor': 1e-5}
SPECTRAL_RADIUS_PARAMS = {'n_probes_spectral': 5, 'power_steps_spectral': 50, 'subset_n_spectral': 10}
HYPER_PARAMS = {**LR_PARAMS, **OTHER_PARAMS, **REGULARIZATION_PARAMS, **SPECTRAL_RADIUS_PARAMS}
                   
def train_signaling_model(mod,  
                          optimizer: torch.optim, 
                          loss_fn: torch.nn.modules.loss,
                          reset_epoch : int = 200,
                          hyper_params: Dict[str, Union[int, float]] = None, 
                         seed: int = 888, 
                         verbose: bool = True):
    """Trains the signaling model

    Parameters
    ----------
    mod : SignalingModel
        initialized signaling model. Suggested to also run `mod.signaling_network.prescale_weights` prior to training
    optimizer : torch.optim.adam.Adam
        optimizer to use during training
    loss_fn : torch.nn.modules.loss.MSELoss
        loss function to use during training
    reset_epoch : int, optional
        number of epochs upon which to reset the optimizer state, by default 200
    hyper_params : Dict[str, Union[int, float]], optional
        various hyper parameter inputs for training
            - 'max_iter' : the number of epochs, by default 5000
            - 'learning_rate' : the starting learning rate, by default 2e-3
            - 'batch_size' : batch size to split samples into, by default 10
            - 'noise_level' : noise added to signaling network input, by default 10. Set to 0 for no noise. Makes model more robust. 
            - 'gradient_noise_level' : noise added to gradient after backward pass. Makes model more robust. 
            - 'param_lambda_L2' : L2 regularization penalty term for most of the model weights and biases
            - 'moa_lambda_L1' : L1 regularization penalty term for incorrect interaction mechanism of action (inhibiting/stimulating)
            - 'ligand_lambda_L2' : L2 regularization penalty term for ligand biases
            - 'uniform_lambda_L2' : L2 regularization penalty term for 
            - 'uniform_max' : 
            - 'spectral_loss_factor' : regularization penalty term for 
            - 'n_probes_spectral' : 
            - 'power_steps_spectral' : 
            - 'subset_n_spectral' : 
    seed : int, optional
        seed value, by default 888
    verbose : bool, optional
        whether to print various progress stats across training epochs


    Returns
    -------
    mod : SignalingModel
        the trained signaling model
    cur_loss : List[float], optional
        a list of the loss (excluding regularizations) across training iterations
    cur_eig : List[float], optional
        a list of the spectral_radius across training iterations
    mean_loss : torch.Tensor
        mean TF activity loss across samples (independent of training)
    """
    if not hyper_params:
        hyper_params = HYPER_PARAMS.copy()
    else:
        hyper_params = {k: v for k,v in {**HYPER_PARAMS, **hyper_params}.items() if k in HYPER_PARAMS} # give user input priority
    
    stats = utils.initialize_progress(hyper_params['max_iter'])
    reset_state = optimizer.state.copy()

    X_in = mod.df_to_tensor(mod.X_in)
    y_out = mod.df_to_tensor(mod.y_out)
    n_samples = X_in.shape[0]
    mean_loss = loss_fn(torch.mean(y_out, dim=0) * torch.ones(y_out.shape, device = y_out.device), y_out) # mean TF (across samples) loss

    for e in range(hyper_params['max_iter']):
        # set learning rate
        cur_lr = utils.get_lr(e, hyper_params['max_iter'], max_height = hyper_params['learning_rate'],
                              start_height=hyper_params['learning_rate']/10, end_height=1e-6, peak = 1000)
        optimizer.param_groups[0]['lr'] = cur_lr
        
        cur_loss = []
        cur_eig = []
        
        np.random.seed(seed + e)
        train_loader = np.array_split(np.random.permutation(n_samples), np.ceil(n_samples/hyper_params['batch_size']).astype(int))
        
        # iterate through batches
        for data_index in train_loader:
            mod.train()
            optimizer.zero_grad()
            
            # get batch I/O
            batch_size_iter = len(data_index)
            X_in_ = X_in[data_index, :].view(batch_size_iter, X_in.shape[1])
            y_out_ = y_out[data_index, :].view(batch_size_iter, y_out.shape[1])
            
            # forward pass
            X_full = mod.input_layer(X_in_) # transform to full network with ligand input concentrations
            utils.set_seeds(mod.seed + mod._gradient_seed_counter)
            network_noise = torch.randn(X_full.shape, device = X_full.device)
            X_full = X_full + (hyper_params['noise_level'] * cur_lr * network_noise) # randomly add noise to signaling network input, makes model more robust
            Y_full = mod.signaling_network(X_full) # train signaling network weights
            Y_hat = mod.output_layer(Y_full)
            
            # get prediction loss
            fit_loss = loss_fn(y_out_, Y_hat)
            
            # get regularization losses
            sign_reg = mod.signaling_network.sign_regularization(lambda_L1 = hyper_params['moa_lambda_L1']) # incorrect MoA
            ligand_reg = mod.ligand_regularization(lambda_L2 = hyper_params['ligand_lambda_L2']) # ligand biases
            stability_loss, spectral_radius = mod.signaling_network.get_SS_loss(Y_full = Y_full.detach(), spectral_loss_factor = hyper_params['spectral_loss_factor'],
                                                                                subset_n = hyper_params['subset_n_spectral'], n_probes = hyper_params['n_probes_spectral'], 
                                                                                power_steps = hyper_params['power_steps_spectral'])
            uniform_reg = mod.uniform_regularization(lambda_L2 = hyper_params['uniform_lambda_L2']*cur_lr, Y_full = Y_full, 
                                                     target_min = 0, target_max = hyper_params['uniform_max']) # uniform distribution
            param_reg = mod.L2_reg(hyper_params['param_lambda_L2']) # all model weights and signaling network biases
            
            total_loss = fit_loss + sign_reg + ligand_reg + param_reg + stability_loss + uniform_reg
    
            # gradient
            total_loss.backward()
            mod.add_gradient_noise(noise_level = hyper_params['gradient_noise_level'])
            optimizer.step()
    
            # store
            cur_eig.append(spectral_radius)
            cur_loss.append(fit_loss.item())
    
        stats = utils.update_progress(stats, iter = e, loss = cur_loss, eig = cur_eig, learning_rate = cur_lr, 
                                     n_sign_mismatches = mod.signaling_network.count_sign_mismatch())
        
        if verbose and e % 250 == 0:
            utils.print_stats(stats, iter = e)
        
        if np.logical_and(e % reset_epoch == 0, e>0):
            optimizer.state = reset_state.copy()

    return mod, cur_loss, cur_eig, mean_loss, stats