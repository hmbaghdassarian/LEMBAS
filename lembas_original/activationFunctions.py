# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 08:16:52 2022

@author: Avlant-MIT
"""
import numpy
import torch

######## MML activation
def MMLactivation(x, leak):
    x = torch.nn.functional.leaky_relu(x, leak)
    shiftedX = 0.5 * (x - 0.5)
    mask = torch.lt(shiftedX, 0.0)
    gatedX = x + 10 * mask #prevents division by 0 issue on next line
    rightValues = 0.5 + torch.div(shiftedX, gatedX)
    x = mask * (x - rightValues) + rightValues #-x trick from relu
    return x

def MMLDeltaActivation(x, leak):   
    mask1 = x.lt(0)
    y = torch.ones(x.shape, dtype = x.dtype, device = x.device) #derivative = 1 if nothing else is stated
    
    mask2 = x.gt(0.5)
    rightValues = 0.25/torch.pow(x,2) - 1
    y = y + mask2 * rightValues
    y = y - (1-leak) * mask1
    return y

def MMLoneStepDeltaActivationFactor(yhatFull, leak):    #Note that this will only work for monoton functions
    y = torch.ones_like(yhatFull)
    piece1 = yhatFull.le(0)
    piece3 = yhatFull.gt(0.5)

    safeX = torch.clamp(1-yhatFull, max=0.9999)
    rightValues = 4 * torch.pow(safeX, 2) - 1
    y = y + piece3 * rightValues
    y = y - (1-leak) * piece1    
    return y

def MMLInvActivation(x, leak):
    safeX = torch.clamp(1-x, max=0.9999)
    factor = 1/leak
    if leak>0:
        mask = x.lt(0)
        x = x + mask * (factor * x -x)
    #Else if it is zero it will be multiplied with a zero later so no need to cover this case
    mask = x.gt(0.5)
    rightValues = 0.25/(safeX) - x    
    x = x + rightValues * mask
    return x



######## leaky relu activation
def leakyReLUActivation(x, leak):
    x = numpy.where(x < 0, x * leak, x)
    return x

def leakyReLUDeltaActivation(x, leak):
    y = numpy.ones(x.shape) #derivative = 1 if nothing else is stated
    y = numpy.where(x <= 0, leak, y)  #let derivative be 0.01 at x=0
    return y

def leakyReLUoneStepDeltaActivationFactor(yhatFull, leak):  #Note that this will only work for monoton functions
    y = torch.ones(yhatFull.shape, dtype=yhatFull.dtype)
    piece1 = yhatFull<=0
    y[piece1] = torch.tensor(leak, dtype=yhatFull.dtype) #there is a bug in torch that sets this to 0 if piece1 is all true, will probably never happen
    return y

def leakyReLUInvActivation(x, leak):
    if leak>0:
        x = numpy.where(x < 0, x/leak, x)
    else:
        x = numpy.where(x < 0, 0, x)
    return x



######## sigmoid activation
def sigmoidActivation(x, leak=0):
    #leak is not used for sigmoid
    x = 1/(1 + numpy.exp(-x))
    return x

def sigmoidDeltaActivation(x, leak):
    y = sigmoidActivation(x) * (1 - sigmoidActivation(x))
    return y

def sigmoidOneStepDeltaActivationFactor(yhatFull, leak):  #Note that this will only work for monoton functions
    y = yhatFull * (1- yhatFull)
    return y

def sigmoidInvActivation(x, leak):
    if leak>0:
        x = numpy.where(x < 0, x/leak, x)
    else:
        x = numpy.where(x < 0, 0, x)
    x = numpy.where(x > 0.5, -0.25/(x-1), x) #Pyhton will display division by zero warning since it evaluates both before selecting
    return x


# def oneStepActivationFactor(yhatFull, leak):
#     y = torch.ones(yhatFull.shape, dtype=yhatFull.dtype)
#     piece1 = yhatFull<=0
#     piece3 = yhatFull>0.5
#     y[piece1] = torch.tensor(leak, dtype=yhatFull.dtype) #there is a bug in torch that sets this to 0 if piece1 all true, will probably never hapen
#     y[piece3] = 4 * (yhatFull[piece3] - yhatFull[piece3]**2)
#     return y

# def activationFactor(x, leak):
#     #x[x<0] = leak*x[x<0]
#     #x[x>=0.5] = 0.5 * (1 + (1./(0.5/(x[x>=0.5]-0.5) + 1)))
#     #y = numpy.ones(x.shape)
#     y = numpy.where(x <= 0, leak, 1)
#     y = numpy.where(x > 0.5, 0.5 * (1 + (1/(0.5/(x-0.5) + 1)))/x, y) #Pyhton will display division by zero warning since it evaluates both before selecting
#     return y



# def activation(x, leak):
#     x = numpy.where(x <= 0, x * leak, x)
#     x = numpy.where(x > 0, 1/((1/x) + 1), x) #Pyhton will display division by zero warning since it evaluates both before selecting
#     return x

# def activationFactor(x, leak):
#     y = numpy.where(x <= 0, leak, 1)
#     y = numpy.where(x > 0, (1/((1/x) + 1))/x, y) #Pyhton will display division by zero warning since it evaluates both before selecting
#     return y

# def deltaActivation(x, leak):
#     y = numpy.ones(x.shape) #derivative = 1 if nothing else is stated
#     y = numpy.where(x <= 0, leak, y)  #let derivative be 0.01 at x=0
#     y = numpy.where(x > 0, 1/((x + 1)**2), y)
#     return y

# def invActivation(x, leak):
#     if leak>0:
#         x = numpy.where(x < 0, x/leak, x)
#     else:
#         x = numpy.where(x < 0, 0, x)
#     x = numpy.where(x > 0, -(1*x)/(x - 1), x)
#     return x
