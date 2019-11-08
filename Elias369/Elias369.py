#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:00:38 2019

@author: eliassmouni
"""

# =============================================================================
# LOAD PACKAGES
# =============================================================================
import os
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import pandas as pd
from sklearn.model_selection import train_test_split


%matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)


# =============================================================================
# SET WD
# =============================================================================

print("Current Working Directory " , os.getcwd())

try:
 #Change the current working Directory    
  os.chdir(os.path.expanduser("~/Documents/Ubiqum/malwaredatathon"))
  print("Directory changed")
except OSError:
  print("Can't change the Current Working Directory")        

# =============================================================================
# LOAD DATA
# =============================================================================

path = 'file:///Users/eliassmouni/Documents/Ubiqum/Elias369/train2.csv'
train_orig = pd.read_csv(path)

# =============================================================================
# EDA
# =============================================================================

train_orig.shape
colnames = train_orig.columns.values
print(colnames)
for i in colnames:
    print(train_orig[i].describe())
    
print(train_orig.Goal.value_counts())

badrow_indices = train_orig[train_orig.Goal == 'Goal'].index
len(badrow_indices)

train_orig = train_orig.drop(train_orig.index[badrow_indices])
len(train_orig)

print(train_orig.Goal.value_counts())

stats = []
for col in colnames:
    stats.append((col, train_orig[col].nunique(), train_orig[col].isnull().sum() * 100 / train_orig.shape[0], train_orig[col].value_counts(normalize=True, dropna=False).values[0] * 100, train_orig[col].dtype))
    
stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage of missing values', 'Percentage of values in the biggest category', 'type'])
stats_df.sort_values('Percentage of missing values', ascending=True)

print(stats_df)

keep_cols = list(colnames)

for col in colnames:
    rate = train_orig[col].isnull().sum() * 100 / train_orig.shape[0]
    if rate > 0.5:
        keep_cols.remove(col)

train = train_orig[keep_cols]
len(train.columns)

train = train.dropna()
train.shape

# =============================================================================
# PREPROCESS AND SPLIT
# =============================================================================

X = train.drop(["Goal"], axis = 1).copy()
y = train["Goal"].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

m_train = X_train.shape[0]
m_test = X_test.shape[0]

X_train_T = np.array(X_train.T)
X_test_T = np.array(X_test.T)

# =============================================================================
# INITIALIZE PARAMETERS FOR SHALLOW NN
# =============================================================================

def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    parameters -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(1)
    
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y,  n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters    


# TEST THAT THE THING WORKS AND THE DIMENSIONS ARE RIGHT
parameters = initialize_parameters(3,2,1)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


# =============================================================================
# INITIALIZE PARAMETERS FOR DEEP NN
# =============================================================================

def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters

# TEST THAT THE THING WORKS AND THAT THE DIMS ARE RIGHT
parameters = initialize_parameters_deep([5,4,3])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


# =============================================================================
# FORWARD PROPAGATION
# =============================================================================

"""
define these helper functions (in this order):

LINEAR
LINEAR -> ACTIVATION where ACTIVATION will be either ReLU or Sigmoid.
[LINEAR -> RELU]  ××  (L-1) -> LINEAR -> SIGMOID (whole model)

"""

# LINEAR FORWARD PROPAGATION

def linear_forward(A, W, b):
    """
    Implements the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass with less headache
    """
    
    Z = np.dot(W, A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1])) # make sure that dims are right
    cache = (A, W, b)
    
    return Z, cache

# DEFINE LINEAR -> ACTIVATION

"""
we will use the relu function in the hidden layers (n1 - nL-1)
and the sigmoid function as activation fuction in the output layer (nL)
first make a conditional helper function that will decide which activation to use
"""

# define activation functions
# sigmoid
def sigmoid(z):
    """
    Arguments:
        z -- a scalar or np array of any size
    
    Returns:
        s -- sigmoid of z
    """
    s = 1 / (1 + np.exp(-z))
    
    return s
   
# CHECK THAT IT WORKS    
sigmoid(np.array([0,2])) 

# relu
def relu(z):
    """
    Arguments:
        z -- a scalar or np array of any size
    
    Returns:
        r -- relu of z
    """
    r = np.maximum(0,z)
    
    return r

# check that the thing works
relu(np.array([[1,2,3],[4,5,6]])) # NICE

def linear_activation_forward(A_prev, W, b, activation):
    
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer: (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python tuple containing "linear_cache" and "activation_cache";
    """
    
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    if activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    
    return A, cache

# next, define the full forward propagation

def L_model_forward(X, parameters):
    
    """
    Implements forward propagation for [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID 
    
    Arguments:
    X -- data, numpy array of shape (input size, n examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value (interim yhat, prediction)
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """
    
    caches = []
    L = len(parameters) # n of layers
    A = X
    
    for l in range(1,L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], activation = "relu")
        caches.append(cache)
        
    A, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], activation = "sigmoid")
    caches.append(cache)
    
    assert (A.shape == (1, X.shape[1]))
    return AL, caches
    
# =============================================================================
# COST FUNCTION
# =============================================================================
    
def compute_cost(AL, Y):
    """
    Implements the cost function defined as - 1/m * sum of all YlogAL + (1-Y)log(1-AL).

    Arguments:
    AL -- probability vector of predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    cost = np.multiply(-1/m,(np.sum((Y* np.log(AL)) + ((1-Y)*np.log(1-AL)))))
    
    cost = np.squeeze(cost)      # To make sure shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost
    
# =============================================================================
#  BACK PROPAGATION   
# =============================================================================
    
# first, define a backward function for the linear part    

def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

# next, define backward function for linear_activation

# first calculate the derivatives of the activation functions
# sigmoid
def dsigmoid(z):
    ds = np.multiply(sigmoid(z), (1 - sigmoid(z)))
    
    return ds

def sigmoid_backward(dA, activation_cache):
    dZ = np.multiply(dA, dsigmoid(activation_cache["Z"]))
    
    return dZ

# relu

def drelu(z):
    if z <= 0:
        dr = 0
    elif z > 0:
        dr = 1
    return dr

def relu_backward(dA, activation_cache):
    dZ = np.multiply(dA, drelu(activation_cache["Z"]))
    
    return dZ

# linear_activation backward

def linear_activation_backward(dA, cache, activation):

    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

# finally, put it all together in L_model_backward

def L_model_backward(AL, Y, caches):
    """
    Implements backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1-Y, 1-AL))
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
    
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, activation = "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

# =============================================================================
# GRADIENT DESCENT FUNCTION
# =============================================================================

def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) 

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        parameters = parameters
        
    return parameters



