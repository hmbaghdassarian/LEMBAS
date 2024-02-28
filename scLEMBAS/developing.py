from torch import nn
from typing import Dict

import sys
sys.path.insert(1, "/home/hmbaghda/Projects/scLEMBAS/")
from scLEMBAS.utils import np_to_torch
from scLEMBAS.model.activation_functions import activation_function_map


class BioNet(nn.Module):
    def __init__(self, edge_list: np.array, 
                 edge_weights: np.array, 
                 n_network_nodes: int, 
                 bionet_params: Dict[str, float], 
                 activation_function: str = 'MML', 
                 dtype: torch.dtype=torch.float32):
        """Initialization method.

        Parameters
        ----------
        edge_list : np.array
            a (2, net.shape[0]) array where the first row represents the indices for the target node and the 
            second row represents the indices for the source node. net.shape[0] is the total # of interactions
            output from  `SignalingModel.parse_network` 
        edge_weights : np.array
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
        """
        super().__init__()
        self.training_params = bionet_params
        self.dtype = dtype
        self.device = device

        self.n_network_nodes = n_network_nodes
        # TODO: delete these _in _out?
        self.n_network_nodes_in = n_network_nodes
        self.n_network_nodes_out = n_network_nodes

        self.edge_list = (np_to_torch(edge_list[0,:], dtype = torch.int32, device = 'cpu'), 
                          np_to_torch(edge_list[1,:], dtype = torch.int32, device = 'cpu'))
        self.edge_weights = np_to_torch(edge_weights, dtype=torch.bool, device = 'cpu')

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
        
        weight_values = 0.1 + 0.1*torch.rand(n_interactions, dtype=self.dtype)
        weight_values[self.edge_weights[1,:]] = -weight_values[self.edge_weights[1,:]] # make those that are inhibiting negative
        
        bias = 1e-3*torch.ones((self.n_network_nodes_in, 1), dtype = self.dtype)
        
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

        weights_mask = torch.zeros(self.n_network_nodes, self.n_network_nodes, dtype=bool) # adjacency list format (targets (rows)--> sources (columns))
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
        weights = torch.zeros(self.mask.shape, dtype = self.dtype) # adjacency matrix
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
            an boolean adjacency matrix of all nodes in the signaling network, with interactions that do not exist or have an unknown 
            mechanism of action masked (False)
        """
    
        signed_MOA = self.edge_weights[0, :].type(torch.long) - self.edge_weights[1, :].type(torch.long) #1=activation -1=inhibition, 0=unknown
        weights_MOA = torch.zeros(self.n_network_nodes_out, self.n_network_nodes_in, dtype=torch.long) # adjacency matrix
        weights_MOA[self.edge_list] = signed_MOA
        mask_MOA = weights_MOA == 0

        return weights_MOA, mask_MOA

    def set_device(self, device: str):
        """Sets torch.tensor objects to the device
        Here, this pushes learned parameters towards `projection_amplitude` 
        
        Parameters
        ----------
        device : str
            set to use gpu ("cuda") or cpu ("cpu")
        """
        self.edge_weights = self.edge_weights.to(device)
        self.mask = self.mask.to(device)
        self.weights_MOA = self.weights_MOA.to(device)
        self.mask_MOA = self.mask_MOA.to(device)

    def forward(self,X_full):
        """Learn the edeg weights within the signaling network topology."""
        self.weights.data.masked_fill_(mask = self.mask, value = 0.0) # fill non-interacting edges with 0
        
        X_bias = X_full.T + self.bias # this is the bias with the projection_amplitude included
        X_new = torch.zeros_like(X_bias) #initialize all values at 0
        
        for t in range(self.training_params['maxSteps']): # like an RNN, updating from previous time step
            X_old = X_new
            X_new = torch.mm(self.weights, X_new) # scale matrix by edge weights
            X_new = X_new + X_bias  # add original values and bias       
            X_new = self.activation(X_new, self.training_params['leak'])
            
            if (t % 10 == 0) and (t > 20):
                diff = torch.max(torch.abs(X_new - X_old))    
                if diff.lt(self.training_params['tolerance']):
                    break

        steady_state = X_new.T
        return steady_state

    def L2Reg(self, L2):
        #L2 = torch.tensor(L2, dtype = self.weights.dtype, device = self.weights.device)
        biasLoss = L2 * torch.sum(torch.square(self.bias))
        weightLoss = L2 * torch.sum(torch.square(self.weights))     
        #biasLoss = 0.1 * torch.sum(torch.abs(self.bias))
        #weightLoss = 0.1 * torch.sum(torch.abs(self.weights))
        return L2 * (biasLoss + weightLoss)

    def getWeight(self, nodeNames, source, target):
        self.A.data = self.weights.detach().numpy()
        locationSource = numpy.argwhere(numpy.isin(nodeNames, source))[0]
        locationTarget = numpy.argwhere(numpy.isin(nodeNames, target))[0]
        weight = self.A[locationTarget, locationSource][0]
        return weight
    
    def steadyStateLoss(self, YhatFull, factor, topNvalues = 10):
        factor = torch.tensor(factor, dtype=YhatFull.dtype, device=YhatFull.device)
        expFactor = torch.tensor(self.param['expFactor'], dtype=YhatFull.dtype, device=YhatFull.device)
        
        selectedValues = numpy.random.permutation(YhatFull.shape[0])[:topNvalues]

        deviationFromSS, aproxSpectralRadius = self.deviationFromSS(YhatFull[selectedValues,:])        
        spectralRadiusFactor = torch.exp(expFactor*(aproxSpectralRadius-self.param['spectralTarget']))
        
        loss = spectralRadiusFactor * deviationFromSS/torch.sum(deviationFromSS.detach())
        loss = factor * torch.sum(loss)
        aproxSpectralRadius = torch.mean(aproxSpectralRadius).item()

        return loss, aproxSpectralRadius
    
    def deviationFromSS(self, yHatSS):
        nProbes = 5
        powerSteps = 50

        xPrime = self.onestepdelta_activation_factor(yHatSS, self.param['leak'])     
        xPrime = xPrime.unsqueeze(2)
        
        T = xPrime * self.weights
        delta = torch.randn((yHatSS.shape[0], yHatSS.shape[1], nProbes), dtype=yHatSS.dtype, device=yHatSS.device)
        for i in range(powerSteps):
            new = delta
            delta = torch.matmul(T, new)

        deviation = torch.max(torch.abs(delta), axis=1)[0]
        aproxSpectralRadius = torch.mean(torch.exp(torch.log(deviation)/powerSteps), axis=1)
        
        deviation = torch.sum(torch.abs(delta), axis=1)
        deviation = torch.mean(torch.exp(torch.log(deviation)/powerSteps), axis=1)

        return deviation, aproxSpectralRadius    
    
    def getWeights(self):
        values = self.weights[self.edge_list]
        return values    

    def getViolations(self):
        #dtype = self.weights.dtype
        signMissmatch = torch.ne(torch.sign(self.weights), self.self.weights_MOA) #.type(dtype)
        signMissmatch = signMissmatch.masked_fill(self.self.mask_MOA, False)
        violations = signMissmatch[self.edge_list]
        wrongSignActivation = torch.logical_and(violations, self.self.edge_weights[0])
        wrongSignInhibition = torch.logical_and(violations, self.self.edge_weights[1])#.type(torch.int)
        return torch.logical_or(wrongSignActivation, wrongSignInhibition)

    def signRegularization(self, MoAFactor):
        MoAFactor = torch.tensor(MoAFactor, dtype = self.weights.dtype, device = self.weights.device)
        signMissmatch = torch.ne(torch.sign(self.weights), self.self.weights_MOA).type(self.weights.dtype)
        signMissmatch = signMissmatch.masked_fill(self.self.mask_MOA, 0)
        loss = MoAFactor * torch.sum(torch.abs(self.weights * signMissmatch))
        return loss

    def balanceWeights(self):
        positiveWeights = self.weights.data>0
        negativeWeights = positiveWeights==False
        positiveSum = torch.sum(self.weights.data[positiveWeights])
        negativeSum = -torch.sum(self.weights.data[negativeWeights])
        factor = positiveSum/negativeSum
        self.weights.data[negativeWeights] = factor * self.weights.data[negativeWeights]



    def preScaleWeights(self, targetRadius = 0.8):
        spectralRadius = getSR(self.weights)
        factor = targetRadius/spectralRadius.item()
        self.weights.data = self.weights.data * factor
