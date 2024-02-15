from typing import Dict, List, Union

import pandas as pd
import numpy as np 
import scipy
from scipy.sparse.linalg import eigs
#from scipy.linalg import norm
from scipy.linalg import eig

import torch
import torch.nn as nn

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

class ProjectInput(nn.Module):
    """Creates the signaling pathway network representation and sets up weights to learn for the inputs."""
    def __init__(self, node_idx_map: Dict[str, int], input_labels: np.array, projection_amplitude: Union[int, float], dtype: torch.dtype=torch.float32):
        """Initialization method for ProjectInput

        Parameters
        ----------
        node_idx_map : Dict[str, int]
            a dictionary mapping node labels (str) to the node index (float)
            generated by `SignalingModel.parse_network`
        input_labels : np.array
            names of the input nodes (ligands) from net
        projection_amplitude : Union[int, float]
            value with which to initialize projection layer weights
        dtype : torch.dtype, optional
            datatype to store values in torch, by default torch.float32
        """
        super().__init__()

        self.projection_amplitude = projection_amplitude
        self.size_out = len(node_idx_map) # number of nodes total in prior knowledge network
        self.input_node_order = torch.tensor([node_idx_map[x] for x in input_labels]) # idx representation of inputs
        weights = self.projection_amplitude * torch.ones(len(input_labels), dtype=dtype)
        self.weights = nn.Parameter(weights)
        
    def forward(self, X_in):
        """Learn the weights for the input ligands to the signaling network"""
        X_full = torch.zeros([X_in.shape[0],  self.size_out], dtype=X_in.dtype, device=X_in.device) # shape of (samples x total nodes in network)
        X_full[:, self.input_node_order] = self.weights * X_in # only modify those nodes that are part of the input (ligands)
        return X_full
    
    def L2Reg(self, L2):
        """Apply an L2 regularization to the neural network parameters"""
        projection_L2 = L2 * torch.sum(torch.square(self.weights - self.projection_amplitude))  
        return projection_L2    
    
    def set_device(self, device):
        self.input_node_order = self.input_node_order.to(device)

class SignalingModel(torch.nn.Module):
    DEFAULT_TRAINING_PARAMETERS = {'targetSteps': 100, 'maxSteps': 300, 'expFactor': 20, 'leak': 0.01, 'tolerance': 1e-5}
    
    def __init__(self, net: pd.DataFrame, X_in: pd.DataFrame, y_out: pd.DataFrame,
                 projection_amplitude: Union[int, float], projection_factor: float,
                 ban_list: List[str] = None, weight_label: str = 'mode_of_action', 
                 source_label: str = 'source', target_label: str = 'target', 
                bionet_params: Dict[str, float] = None , 
                 activation_function: str='MML', dtype: torch.dtype=torch.float32, device: str = 'cpu'):
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
        projection_amplitude : Union[int, float]
            value with which to initialize projection layer weights, passed to `ProjectInput`
        projection_factor : float
            _description_
        bionet_params : Dict[str, float], optional
            training parameters for the model, by default None
        activation_function : str, optional
            _description_, by default 'MML'
        dtype : torch.dtype, optional
            datatype to store values in torch, by default torch.float32
        device : str
            whether to use gpu ("cuda") or cpu ("cpu"), by default "cpu"
        """

        
        super().__init__()
        self.dtype = dtype
        self.device = device

        edge_list, node_labels, edge_weights = self.parse_network(net, ban_list, weight_label, source_label, target_label)
        if not bionet_params:
            bionet_params = self.DEFAULT_TRAINING_PARAMETERS.copy()
        else:
            bionet_params = self.set_training_parameters(**bionet_params)

        # filter for nodes in the network, sorting by node_labels order
        self.X_in = X_in.loc[:, np.intersect1d(X_in.columns.values, node_labels)]
        self.y_out = y_out.loc[:, np.intersect1d(y_out.columns.values, node_labels)]
        self.input_labels = self.X_in.columns.values
        self.output_labels = self.y_out.columns.values

        # define model layers 
        self.input_layer = ProjectInput(self.node_idx_map, self.input_labels, projection_amplitude, self.dtype)

    def parse_network(self, net: pd.DataFrame, ban_list: List[str] = None, 
                 weight_label: str = 'mode_of_action', source_label: str = 'source', target_label: str = 'target'):
        """Parse adjacency network . Adapted from LEMBAS `loadNetwork` and `makeNetworkList`.
    
        Parameters
        ----------
        net: pd.DataFrame
            signaling network adjacency list with the following columns:
                - `weight_label`: whether the interaction is stimulating (1) or inhibiting (-1) or unknown (0). Exclude non-interacting (0) nodes. 
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
        edge_weights : np.array
            a (1, net.shape[0]) array where the first row is a boolean of whether the interactions are stimulating and the 
            second row is a boolean of whether the interactions are inhibiting
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
        # edge_weights = net[weight_label].values
        # get edge list *ordered by source-target node index*
        n_nodes = len(node_labels)
        A = scipy.sparse.csr_matrix((net[weight_label].values, (source_indices, target_indices)), shape=(n_nodes, n_nodes)) # calculate adjacency matrix
        source_indices, target_indices, edge_weights = scipy.sparse.find(A) # re-orders adjacency list by index
        edge_list = np.array((target_indices, source_indices))
    
        return edge_list, node_labels, edge_weights

    def df_to_tensor(self, df: pd.DataFrame):
        """Converts a pandas dataframe to the appropriate torch.tensor"""
        return torch.tensor(df.values.copy(), dtype=self.dtype).to(self.device)

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
        allowed_params = list(default_parameters.keys()) + ['spectralTarget']
    
        params = {**default_parameters, **attributes}
        if 'spectralTarget' not in params.keys():
            params['spectralTarget'] = np.exp(np.log(params['tolerance'])/params['targetSteps'])
    
        params = {k: v for k,v in params.items() if k in allowed_params}
    
        return params

    # def forward(self, X_in):
    #     X_full = self.inputLayer(X_in) # input ligand weights
    #     fullY = self.network(X_full)
    #     Yhat = self.projectionLayer(fullY)
    #     return Yhat, fullY
