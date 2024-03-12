"""
Defines the various layers in the SignalingModel RNN.
"""

from typing import Dict, List, Union, Annotated
from annotated_types import Ge

import pandas as pd
import numpy as np 
import scipy
from scipy.sparse.linalg import eigs

import torch
import torch.nn as nn

import sys
sys.path.insert(1, "/home/hmbaghda/Projects/scLEMBAS/")
from scLEMBAS.model.model_utilities import np_to_torch, format_network
from scLEMBAS.model.activation_functions import activation_function_map
from scLEMBAS.utilities import set_seeds

class ProjectInput(nn.Module):
    """Generate all nodes for the signaling network and linearly scale input ligand values by NN parameters."""
    def __init__(self, node_idx_map: Dict[str, int], input_labels: np.array, projection_amplitude: Union[int, float] = 1, dtype: torch.dtype=torch.float32, device: str = 'cpu'):
        """Initialization method.

        Parameters
        ----------
        node_idx_map : Dict[str, int]
            a dictionary mapping node labels (str) to the node index (float)
            generated by `SignalingModel.parse_network`
        input_labels : np.array
            names of the input nodes (ligands) from net
        projection_amplitude : Union[int, float]
            value with which to initialize learned linear scaling parameters, by default 1. (if turn require_grad = False for this layer, this is still applied 
            simply as a constant linear scalar in each forward pass)
        dtype : torch.dtype, optional
            datatype to store values in torch, by default torch.float32
        device : str
            whether to use gpu ("cuda") or cpu ("cpu"), by default "cpu"
        """
        super().__init__()

        self.device = device
        self.dtype = dtype
        self.projection_amplitude = projection_amplitude
        self.size_out = len(node_idx_map) # number of nodes total in prior knowledge network
        self.input_node_order = torch.tensor([node_idx_map[x] for x in input_labels], device = self.device) # idx representation of ligand inputs
        weights = self.projection_amplitude * torch.ones(len(input_labels), dtype=self.dtype, device = self.device) # scaled input weights
        self.weights = nn.Parameter(weights)
        
    def forward(self, X_in: torch.Tensor):
        """Learn the weights for the input ligands to the signaling network (if grad_fn set to False, 
        simply scales by projection amplitude).
        Transform from ligand input (samples x ligands) to full signaling network (samples x network nodes).

        Parameters
        ----------
        X_in : torch.Tensor
            the ligand concentration inputs. Shape is (samples x ligands). 

        Returns
        -------
        X_full :  torch.Tensor
            the linearly scaled ligand inputs. Shape is (samples x network nodes)
        """
        X_full = torch.zeros([X_in.shape[0],  self.size_out], dtype=self.dtype, device=self.device) # shape of (samples x total nodes in network)
        X_full[:, self.input_node_order] = self.weights * X_in # only modify those nodes that are part of the input (ligands)
        return X_full
    
    def L2_reg(self, lambda_L2: Annotated[float, Ge(0)] = 0):
        """Get the L2 regularization term for the neural network parameters.
        Here, this pushes learned parameters towards `projection_amplitude` 
        
        Parameters
        ----------
        lambda_2 : Annotated[float, Ge(0)]
            the regularization parameter, by default 0 (no penalty) 
        
        Returns
        -------
        projection_L2 : torch.Tensor
            the regularization term
        """
        # if removed the `- self.projection_amplitude` part, would force weights to 0, thus shrinking ligand inputs
        projection_L2 = lambda_L2 * torch.sum(torch.square(self.weights - self.projection_amplitude))  
        return projection_L2

class BioNet(nn.Module):
    """Builds the RNN on the signaling network topology."""
    def __init__(self, edge_list: np.array, 
                 edge_MOA: np.array, 
                 n_network_nodes: int, 
                 bionet_params: Dict[str, float], 
                 activation_function: str = 'MML', 
                 dtype: torch.dtype=torch.float32, 
                device: str = 'cpu', 
                seed: int = 888):
        """Initialization method.

        Parameters
        ----------
        edge_list : np.array
            a (2, net.shape[0]) array where the first row represents the indices for the target node and the 
            second row represents the indices for the source node. net.shape[0] is the total # of interactions
            output from  `SignalingModel.parse_network` 
        edge_MOA : np.array
            a (2, net.shape[0]) array where the first row is a boolean of whether the interactions are stimulating and the 
            second row is a boolean of whether the interactions are inhibiting
            output from  `SignalingModel.parse_network`
        n_network_nodes : int
            the number of nodes in the network
        bionet_params : Dict[str, float]
            training parameters for the model
            see `SignalingModel.set_training_parameters`
        activation_function : str, optional
            RNN activation function, by default 'MML'
            options include:
                - 'MML': Michaelis-Menten-like
                - 'leaky_relu': Leaky ReLU
                - 'sigmoid': sigmoid 
        dtype : torch.dtype, optional
           datatype to store values in torch, by default torch.float32
        device : str
            whether to use gpu ("cuda") or cpu ("cpu"), by default "cpu"
        seed : int
            random seed for torch and numpy operations, by default 888
        """
        super().__init__()
        self.training_params = bionet_params
        self.dtype = dtype
        self.device = device
        self.seed = seed
        self._ss_seed_counter = 0

        self.n_network_nodes = n_network_nodes
        # TODO: delete these _in _out?
        self.n_network_nodes_in = n_network_nodes
        self.n_network_nodes_out = n_network_nodes

        self.edge_list = (np_to_torch(edge_list[0,:], dtype = torch.int32, device = 'cpu'), 
                          np_to_torch(edge_list[1,:], dtype = torch.int32, device = 'cpu'))
        self.edge_MOA = np_to_torch(edge_MOA, dtype=torch.bool, device = self.device)

        # initialize weights and biases
        weights, bias = self.initialize_weights()
        self.weights = nn.Parameter(weights)
        self.bias = nn.Parameter(bias)

        self.weights_MOA, self.mask_MOA = self.make_mask_MOA() # mechanism of action 

        # activation function
        self.activation = activation_function_map[activation_function]['activation']
        self.delta = activation_function_map[activation_function]['delta']
        self.onestepdelta_activation_factor = activation_function_map[activation_function]['onestepdelta']

    def initialize_weight_values(self):
        """Initialize the RNN weight_values for all interactions in the signaling network.

        Returns
        -------
        weight_values : torch.Tensor
            a torch.Tensor with randomly initialized values for each signaling network interaction
        bias : torch.Tensor
            a torch.Tensor with randomly initialized values for each signaling network node
        """
        
        network_targets = self.edge_list[0].numpy() # the target nodes receiving an edge
        n_interactions = len(network_targets)

        set_seeds(self.seed)
        weight_values = 0.1 + 0.1*torch.rand(n_interactions, dtype=self.dtype, device = self.device)
        weight_values[self.edge_MOA[1,:]] = -weight_values[self.edge_MOA[1,:]] # make those that are inhibiting negative
        
        bias = 1e-3*torch.ones((self.n_network_nodes_in, 1), dtype = self.dtype, device = self.device)
        
        for nt_idx in np.unique(network_targets):
            if torch.all(weight_values[network_targets == nt_idx]<0):
                bias.data[nt_idx] = 1
    
        return weight_values, bias

    def make_mask(self):
        """Generates a mask for adjacency matrix for non-interacting nodes.

        Returns
        -------
        weights_mask : torch.Tensor
            a boolean adjacency matrix of all nodes in the signaling network, masking (True) interactions that are not present
        """

        weights_mask = torch.zeros(self.n_network_nodes, self.n_network_nodes, dtype=bool, device = self.device) # adjacency list format (targets (rows)--> sources (columns))
        weights_mask[self.edge_list] = True # if interaction is present, do not mask
        weights_mask = torch.logical_not(weights_mask) # make non-interacting edges False and vice-vesa
        return weights_mask

    def initialize_weights(self):
        """Initializes weights and masks for interacting nodes and mechanism of action.

        Returns
        -------
        weights : torch.Tensor
            a torch.Tensor adjacency matrix with randomly initialized values for each signaling network interaction
        bias : torch.Tensor
            a torch.Tensor with randomly initialized values for each signaling network node
        """

        weight_values, bias = self.initialize_weight_values()
        self.mask = self.make_mask()
        weights = torch.zeros(self.mask.shape, dtype = self.dtype, device = self.device) # adjacency matrix
        weights[self.edge_list] = weight_values
        
        return weights, bias

    def make_mask_MOA(self):
        """Generates mask (and weights) for adjacency matrix for non-interacting nodes AND nodes were mode of action (stimulating/inhibiting) 
        is unknown.

        Returns
        -------
        weights_MOA : torch.Tensor
            an adjacency matrix of all nodes in the signaling network, with activating interactions set to 1, inhibiting interactions set 
            to -1, and interactions that do not exist or have an unknown mechanism of action (stimulating/inhibiting) set to 0
        mask_MOA : torch.Tensor
            a boolean adjacency matrix of all nodes in the signaling network, with interactions that do not exist or have an unknown 
            mechanism of action masked (True)
        """
    
        signed_MOA = self.edge_MOA[0, :].type(torch.long) - self.edge_MOA[1, :].type(torch.long) #1=activation -1=inhibition, 0=unknown
        weights_MOA = torch.zeros(self.n_network_nodes_out, self.n_network_nodes_in, dtype=torch.long, device = self.device) # adjacency matrix
        weights_MOA[self.edge_list] = signed_MOA
        mask_MOA = weights_MOA == 0

        return weights_MOA, mask_MOA

    def prescale_weights(self, target_radius: float = 0.8):
        """Scale weights according to spectral radius
    
        Parameters
        ----------
        target_radius : float, optional
            _description_, by default 0.8
        """

        A = scipy.sparse.csr_matrix(self.weights.detach().cpu().numpy())
        np.random.seed(self.seed)
        eigen_value, _ = eigs(A, k = 1, v0 = np.random.rand(A.shape[0])) # first eigen value
        spectral_radius = np.abs(eigen_value)
        
        factor = target_radius/spectral_radius.item()
        self.weights.data = self.weights.data * factor

    def forward(self, X_full: torch.Tensor):
        """Learn the edeg weights within the signaling network topology.

        Parameters
        ----------
        X_full : torch.Tensor
            the linearly scaled ligand inputs. Shape is (samples x network nodes). Output of ProjectInput.

        Returns
        -------
        Y_full :  torch.Tensor
            the signaling network scaled by learned interaction weights. Shape is (samples x network nodes).
        """
        self.weights.data.masked_fill_(mask = self.mask, value = 0.0) # fill non-interacting edges with 0
        
        X_bias = X_full.T + self.bias # this is the bias with the projection_amplitude included
        X_new = torch.zeros_like(X_bias) #initialize all values at 0
        
        for t in range(self.training_params['max_steps']): # like an RNN, updating from previous time step
            X_old = X_new
            X_new = torch.mm(self.weights, X_new) # scale matrix by edge weights
            X_new = X_new + X_bias  # add original values and bias       
            X_new = self.activation(X_new, self.training_params['leak'])
            
            if (t % 10 == 0) and (t > 20):
                diff = torch.max(torch.abs(X_new - X_old))    
                if diff.lt(self.training_params['tolerance']):
                    break

        Y_full = X_new.T
        return Y_full

    def L2_reg(self, lambda_L2: Annotated[float, Ge(0)] = 0):
        """Get the L2 regularization term for the neural network parameters.
        
        Parameters
        ----------
        lambda_2 : Annotated[float, Ge(0)]
            the regularization parameter, by default 0 (no penalty) 
        
        Returns
        -------
        bionet_L2 : torch.Tensor
            the regularization term
        """
        bias_loss = lambda_L2 * torch.sum(torch.square(self.bias))
        weight_loss = lambda_L2 * torch.sum(torch.square(self.weights))

        bionet_L2 = bias_loss + weight_loss
        return bionet_L2

    def get_sign_mistmatch(self):
        """Identifies edge weights in network that have a sign that does not agree
        with the known mode of action.
    
        Mode of action: stimulating interactions are expected to have positive weights and inhibiting interactions
        are expected to have negative weights.
        
        Returns
        -------
        sign_mismatch : torch.Tensor
            a binary adjacency matrix of all nodes in the signaling network, where values are 1 if they do not 
            match the mode of action and 0 if they match the mode of action or have an unknown mode of action
        """
        sign_mismatch = torch.ne(torch.sign(self.weights), self.weights_MOA).type(self.dtype) 
        sign_mismatch = sign_mismatch.masked_fill(self.mask_MOA, 0) # do not penalize sign mismatches of unknown interactions
    
        return sign_mismatch

    def count_sign_mismatch(self):
        """Counts total sign mismatches identified in `get_sign_mistmatch`
        
        Returns
        -------
        n_sign_mismatches : float
            the total number of sign mismatches at `iter`
        """
        n_sign_mismatches = torch.sum(self.get_sign_mistmatch()).item()
        return n_sign_mismatches

    def sign_regularization(self, lambda_L1: Annotated[float, Ge(0)] = 0):
        """Get the L1 regularization term for the neural network parameters that 
        do not fit the mechanism of action (i.e., negative weights for stimulating interactions or positive weights for inhibiting interactions).
        Only penalizes sign mismatches of known MOA.
    
        Parameters
        ----------
        lambda_L1 : Annotated[float, Ge(0)]
            the regularization parameter, by default 0 (no penalty) 
    
        Returns
        -------
        loss : torch.Tensor
            the regularization term
        """
        lambda_L1 = torch.tensor(lambda_L1, dtype = self.dtype, device = self.device)
        sign_mismatch = self.get_sign_mistmatch() # will not penalize sign mismatches of unknown interactions

        loss = lambda_L1 * torch.sum(torch.abs(self.weights * sign_mismatch))
        return loss

    # def get_sign_mistmatch_edge_list(self):
    #     """Same as `get_sign_mistmatch`, but converts to coordinates corresponding to `edge_list`
        
    #     Returns
    #     -------
    #     sign_mismatch : torch.Tensor
    #         a binary vector corresponding to coordinates in `edge_list`, where values are 1 if they do not 
    #         match the mode of action and 0 if they match the mode of action or have an unknown mode of action
    #     """
    #     sign_mismatch = self.get_sign_mistmatch()
        
    #     # violations = sign_mismatch[self.edge_list] # 1 for interactions in edge list that mismatch, 0 otherwise
    #     # activation_mismatch = torch.logical_and(violations, self.edge_MOA[0])
    #     # inhibition_mismatch = torch.logical_and(violations, self.edge_MOA[1])
    #     # all_mismatch = torch.logical_or(activation_mismatch, inhibition_mismatch)
        
    #     sign_mismatch_edge = sign_mismatch[self.edge_list] # 1 for interactions in edge list that mismatch, 0 otherwise
        
    #     return sign_mismatch_edge

    def get_SS_loss(self, Y_full: torch.Tensor, spectral_loss_factor: float, subset_n: int = 10, **kwargs):
        """_summary_
    
        Parameters
        ----------
        Y_full : torch.Tensor
            output of the forward pass
            ensure to run `torch.Tensor.detach` method prior to inputting so that gradient calculations are not effected
        spectral_loss_factor : float
            _description_
        subset_n : int, optional
            _description_, by default 10
    
        Returns
        -------
        _type_
            _description_
        """
        spectral_loss_factor = torch.tensor(spectral_loss_factor, dtype=Y_full.dtype, device=Y_full.device)
        exp_factor = torch.tensor(self.training_params['exp_factor'], dtype=Y_full.dtype, device=Y_full.device)
    
        if self.seed:
            np.random.seed(self.seed + self._ss_seed_counter)
        selected_values = np.random.permutation(Y_full.shape[0])[:subset_n]
    
        SS_deviation, aprox_spectral_radius = self._get_SS_deviation(Y_full[selected_values,:], **kwargs)        
        spectral_radius_factor = torch.exp(exp_factor*(aprox_spectral_radius-self.training_params['spectral_target']))
        
        loss = spectral_radius_factor * SS_deviation/torch.sum(SS_deviation.detach())
        loss = spectral_loss_factor * torch.sum(loss)
        aprox_spectral_radius = torch.mean(aprox_spectral_radius).item()
    
        self._ss_seed_counter += 1 # new seed each time this (and _get_SS_deviation) is called
    
        return loss, aprox_spectral_radius
    
    def _get_SS_deviation(self, Y_full_sub, n_probes: int = 5, power_steps: int = 50):
        x_prime = self.onestepdelta_activation_factor(Y_full_sub, self.training_params['leak'])     
        x_prime = x_prime.unsqueeze(2)
        
        T = x_prime * self.weights
        if self.seed:
            set_seeds(self.seed + self._ss_seed_counter)
        delta = torch.randn((Y_full_sub.shape[0], Y_full_sub.shape[1], n_probes), dtype=Y_full_sub.dtype, device=Y_full_sub.device)
        for i in range(power_steps):
            new = delta
            delta = torch.matmul(T, new)
    
        SS_deviation = torch.max(torch.abs(delta), axis=1)[0]
        aprox_spectral_radius = torch.mean(torch.exp(torch.log(SS_deviation)/power_steps), axis=1)
        
        SS_deviation = torch.sum(torch.abs(delta), axis=1)
        SS_deviation = torch.mean(torch.exp(torch.log(SS_deviation)/power_steps), axis=1)
    
        return SS_deviation, aprox_spectral_radius

class ProjectOutput(nn.Module):
    """Transforms signaling network to TF activity."""
    def __init__(self, node_idx_map, output_labels, projection_amplitude, dtype, device):
        super().__init__()

        self.size_in = len(node_idx_map)
        self.size_out = len(output_labels)
        self.projection_amplitude = projection_amplitude

        self.output_node_order = torch.tensor([node_idx_map[x] for x in output_labels], device = device) # idx representation of TF outputs

        weights = self.projection_amplitude * torch.ones(len(output_labels), dtype=dtype, device = device)
        self.weights = nn.Parameter(weights)

    def forward(self, Y_full):
        """Learn the weights for the output TFs of the signaling network (if grad_fn set to False, 
        simply scales by projection amplitude).
        Transforms full signaling network  (samples x network nodes) to only the space of the TFs.

        Parameters
        ----------
        Y_full : torch.Tensor
            the signaling network scaled by learned interaction weights. Shape is (samples x network nodes). 
            Output of BioNet.

        Returns
        -------
        Y_hat :  torch.Tensor
            the linearly scaled TF outputs. Shape is (samples x TFs)
        """
        Y_hat = self.weights * Y_full[:, self.output_node_order]
        return Y_hat

    def L2_reg(self, lambda_L2: Annotated[float, Ge(0)] = 0):
        """Get the L2 regularization term for the neural network parameters.
        Here, this pushes learned parameters towards `projection_amplitude` 
        
        Parameters
        ----------
        lambda_2 : Annotated[float, Ge(0)]
            the regularization parameter, by default 0 (no penalty) 
        
        Returns
        -------
        projection_L2 : torch.Tensor
            the regularization term
        """
        projection_L2 = lambda_L2 * torch.sum(torch.square(self.weights - self.projection_amplitude))  
        return projection_L2
    
    # def set_device(self, device: str):
    #     """Sets torch.tensor objects to the device
        
    #     Parameters
    #     ----------
    #     device : str
    #         set to use gpu ("cuda") or cpu ("cpu")
    #     """
    #     self.output_node_order = self.output_node_order.to(device)

class SignalingModel(torch.nn.Module):
    """Constructs the signaling network based RNN."""
    DEFAULT_TRAINING_PARAMETERS = {'target_steps': 100, 'max_steps': 300, 'exp_factor': 20, 'leak': 0.01, 'tolerance': 1e-5}
    
    def __init__(self, net: pd.DataFrame, X_in: pd.DataFrame, y_out: pd.DataFrame,
                 projection_amplitude_in: Union[int, float] = 1, projection_amplitude_out: float = 1,
                 ban_list: List[str] = None, weight_label: str = 'mode_of_action', 
                 source_label: str = 'source', target_label: str = 'target', 
                bionet_params: Dict[str, float] = None , 
                 activation_function: str='MML', dtype: torch.dtype=torch.float32, device: str = 'cpu', seed: int = 888):
        """Parse the signaling network and build the model layers.

        Parameters
        ----------
        net: pd.DataFrame
            signaling network adjacency list with the following columns:
                - `weight_label`: whether the interaction is stimulating (1) or inhibiting (-1). Exclude non-interacting (0) nodes. 
                - `source_label`: source node column name
                - `target_label`: target node column name
        X_in : pd.DataFrame
            input ligand concentrations. Index represents samples and columns represent a ligand. Values represent amount of ligand introduced (e.g., concentration). 
        y_out : pd.DataFrame
            output TF activities. Index represents samples and columns represent TFs. Values represent activity of the TF. 
        ban_list : List[str], optional
            a list of signaling network nodes to disregard, by default None
        projection_amplitude_in : Union[int, float]
            value with which to scale ligand inputs by, by default 1 (see `ProjectInput` for details, can also be tuned as a learned parameter in the model)
        projection_amplitude_out : float
             value with which to scale TF activity outputs by, by default 1 (see `ProjectOutput` for details, can also be tuned as a learned parameter in the model)
        bionet_params : Dict[str, float], optional
            training parameters for the model, by default None
            Key values include:
                - 'max_steps': maximum number of time steps of the RNN, by default 300
                - 'tolerance': threshold at which to break RNN; based on magnitude of change of updated edge weight values, by default 1e-5
                - 'leak': parameter to tune extent of leaking, analogous to leaky ReLU, by default 0.01
                - 'spectral_target': _description_, by default np.exp(np.log(params['tolerance'])/params['target_steps'])
                - 'exp_factor': _description_, by default 20
        activation_function : str, optional
            RNN activation function, by default 'MML'
            options include:
                - 'MML': Michaelis-Menten-like
                - 'leaky_relu': Leaky ReLU
                - 'sigmoid': sigmoid 
        dtype : torch.dtype, optional
            datatype to store values in torch, by default torch.float32
        device : str
            whether to use gpu ("cuda") or cpu ("cpu"), by default "cpu"
        seed : int
            random seed for torch and numpy operations, by default 888
        """
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.seed = seed
        self._gradient_seed_counter = 0
        self.projection_amplitude_out = projection_amplitude_out

        edge_list, node_labels, edge_MOA = self.parse_network(net, ban_list, weight_label, source_label, target_label)
        if not bionet_params:
            bionet_params = self.DEFAULT_TRAINING_PARAMETERS.copy()
        else:
            bionet_params = self.set_training_parameters(**bionet_params)

        # filter for nodes in the network, sorting by node_labels order
        self.X_in = X_in.loc[:, np.intersect1d(X_in.columns.values, node_labels)]
        self.y_out = y_out.loc[:, np.intersect1d(y_out.columns.values, node_labels)]

        # define model layers
        self.input_layer = ProjectInput(node_idx_map = self.node_idx_map, 
                                        input_labels = self.X_in.columns.values, 
                                        projection_amplitude = projection_amplitude_in, 
                                        dtype = self.dtype, 
                                        device = self.device)
        self.signaling_network = BioNet(edge_list = edge_list, 
                                        edge_MOA = edge_MOA, 
                                        n_network_nodes = len(node_labels), 
                                        bionet_params = bionet_params, 
                                        activation_function = activation_function, 
                                        dtype = self.dtype, device = self.device, seed = self.seed)
        self.output_layer = ProjectOutput(node_idx_map = self.node_idx_map, 
                                          output_labels = self.y_out.columns.values, 
                                          projection_amplitude = self.projection_amplitude_out, 
                                          dtype = self.dtype, device = device)

    def parse_network(self, net: pd.DataFrame, ban_list: List[str] = None, 
                 weight_label: str = 'mode_of_action', source_label: str = 'source', target_label: str = 'target'):
        """Parse adjacency network.
    
        Parameters
        ----------
        net: pd.DataFrame
            signaling network adjacency list with the following columns:
                - `weight_label`: whether the interaction is stimulating (1) or inhibiting (-1) or unknown (0). Exclude non-interacting (0)
                nodes. 
                - `source_label`: source node column name
                - `target_label`: target node column name
        ban_list : List[str], optional
            a list of signaling network nodes to disregard, by default None
    
        Returns
        -------
        edge_list : np.array
            a (2, net.shape[0]) array where the first row represents the indices for the target node and the 
            second row represents the indices for the source node. net.shape[0] is the total # of interactions
        node_labels : list
            a list of the network nodes in the same order as the indices
        edge_MOA : np.array
            a (2, net.shape[0]) array where the first row is a boolean of whether the interactions are stimulating and the 
            second row is a boolean of whether the interactions are inhibiting. 
            
            Note: Edge_list includes interactions that are not delineated as activating OR inhibiting, s.t. edge_MOA records this 
            as [False, False].
        """
        if not ban_list:
            ban_list = []
        if sorted(net[weight_label].unique()) != [-1, 0.1, 1]:
            raise ValueError(weight_label + ' values must be 1 or -1')
        
        net = net[~ net[source_label].isin(ban_list)]
        net = net[~ net[target_label].isin(ban_list)]
    
        # create an edge list with node incides
        node_labels = sorted(pd.concat([net[source_label], net[target_label]]).unique())
        self.node_idx_map = {idx: node_name for node_name, idx in enumerate(node_labels)}
        
        source_indices = net[source_label].map(self.node_idx_map).values
        target_indices = net[target_label].map(self.node_idx_map).values

        # # get edge list
        # edge_list = np.array((target_indices, source_indices))
        # edge_MOA = net[weight_label].values
        # get edge list *ordered by source-target node index*
        n_nodes = len(node_labels)
        A = scipy.sparse.csr_matrix((net[weight_label].values, (source_indices, target_indices)), shape=(n_nodes, n_nodes)) # calculate adjacency matrix
        source_indices, target_indices, edge_MOA = scipy.sparse.find(A) # re-orders adjacency list by index
        edge_list = np.array((target_indices, source_indices)) 
        edge_MOA = np.array([[edge_MOA==1],[edge_MOA==-1]]).squeeze() # convert to boolean

        return edge_list, node_labels, edge_MOA

    def df_to_tensor(self, df: pd.DataFrame):
        """Converts a pandas dataframe to the appropriate torch.tensor"""
        return torch.tensor(df.values.copy(), dtype=self.dtype, device = self.device)

    def set_training_parameters(self, **attributes):
        """Set the parameters for training the model. Overrides default parameters with attributes if specified.
        Adapted from LEMBAS `trainingParameters`
    
        Parameters
        ----------
        attributes : dict
            keys are parameter names and values are parameter value
        """
        #set defaults
        default_parameters = self.DEFAULT_TRAINING_PARAMETERS.copy()
        allowed_params = list(default_parameters.keys()) + ['spectral_target']
    
        params = {**default_parameters, **attributes}
        if 'spectral_target' not in params.keys():
            params['spectral_target'] = np.exp(np.log(params['tolerance'])/params['target_steps'])
    
        params = {k: v for k,v in params.items() if k in allowed_params}
    
        return params

    def forward(self, X_in):
        """Linearly scales ligand inputs, learns weights for signaling network interactions, and transforms this to TF activity. See
        `forward` methods of each layer for details."""
        X_full = self.input_layer(X_in) # input ligands to signaling network
        Y_full = self.signaling_network(X_full) # RNN of full signaling network
        Y_hat = self.output_layer(Y_full) # TF outputs of signaling network
        return Y_hat, Y_full

    def L2_reg(self, lambda_L2: Annotated[float, Ge(0)] = 0):
        """Get the L2 regularization term for the neural network parameters.
        
        Parameters
        ----------
        lambda_L2 : Annotated[float, Ge(0)]
            the regularization parameter, by default 0 (no penalty) 
        
        Returns
        -------
         : torch.Tensor
            the regularization term (as the sum of the regularization terms for each layer)
        """
        return self.input_layer.L2_reg(lambda_L2) + self.signaling_network.L2_reg(lambda_L2) + self.output_layer.L2_reg(lambda_L2)

    def ligand_regularization(self, lambda_L2: Annotated[float, Ge(0)] = 0):
        """Get the L2 regularization term for the ligand biases. Intuitively, extracellular ligands should not contribute to 
        "baseline (i.e., unstimulated) activity" affecting intrecllular signaling nodes and thus TF outputs.
        
        Parameters
        ----------
        lambda_L2 : Annotated[float, Ge(0)]
            the regularization parameter, by default 0 (no penalty) 
        
        Returns
        -------
        loss : torch.Tensor
            the regularization term
        """
        loss = lambda_L2 * torch.sum(torch.square(self.signaling_network.bias[self.input_layer.input_node_order]))
        return loss

    def uniform_regularization(self, lambda_L2: float, Y_full: torch.Tensor, 
                     target_min: float = 0.0, target_max: float = None):
        """Get the L2 regularization term for deviations of the nodes in Y_full from that of a uniform distribution between 
        `target_min` and `target_max`. 
        Note, this penalizes both deviations from the uniform distribution AND values that are out of range (like a double penalty).
    
        Parameters
        ----------
        lambda_L2 : float
            scaling factor for state loss
        Y_full : torch.Tensor
            the signaling network scaled by learned interaction weights. Shape is (samples x network nodes). 
            Output of BioNet.
        target_min : float, optional
            minimum values for nodes in Y_full to take on, by default 0.0
        target_max : float, optional
            maximum values for nodes in Y_full to take on, by default 1/`self.projection_amplitude_out`
    
        Returns
        -------
        loss : torch.Tensor
            the regularization term
        """
        lambda_L2 = torch.tensor(lambda_L2, dtype = Y_full.dtype, device = Y_full.device)
        # loss = lambda_L2 * expected_uniform_distribution(Y_full, target_max = 1/self.projectionAmplitude)
        if not target_max:
            target_max = 1/self.projection_amplitude_out
        
        sorted_Y_full, _ = torch.sort(Y_full, axis=0) # sorts each column (signaling network node) in ascending order
        target_distribution = torch.linspace(target_min, target_max, Y_full.shape[0], dtype=Y_full.dtype, device=Y_full.device).reshape(-1, 1)
        
        dist_loss = torch.sum(torch.square(sorted_Y_full - target_distribution)) # difference in distribution
        below_loss = torch.sum(Y_full.lt(target_min) * torch.square(Y_full-target_min)) # those that are below the minimum value
        above_loss = torch.sum(Y_full.gt(target_max) * torch.square(Y_full-target_max)) # those that are above the maximum value
        loss = lambda_L2*(dist_loss + below_loss + above_loss)
        return loss

    def add_gradient_noise(self, noise_level: Union[float, int]):
        """Adds noise to backwards pass gradient calculations. Use during training to make model more robust. 
    
        Parameters
        ----------
        noise_level : Union[float, int]
            scaling factor for amount of noise to add 
        """
        all_params = list(self.parameters())
        if self.seed:
            set_seeds(self.seed + self._gradient_seed_counter)
        for i in range(len(all_params)):
            if all_params[i].requires_grad:
                all_noise = torch.randn(all_params[i].grad.shape, dtype=all_params[i].dtype, device=all_params[i].device)
                all_params[i].grad += (noise_level * all_noise)
    
        self._gradient_seed_counter += 1 # new random noise each time function is called