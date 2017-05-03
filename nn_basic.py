# neural_networks from scratch
# amritansh


import numpy as np
import scipy as sp
from sklearn import datasets
import matplotlib.pyplot as plt 


# generate a random distribution and plot it.

np.random.seed(0)
X, Y = datasets.make_moons(300, noise = 0.35)
plt.scatter(X[:,0], X[:,1], s=40 , c=Y, cmap = plt.cm.Spectral)
plt.show()

# Training dataset, input and output layer dimension
data_size = len(X)
nn_input_dim , nn_output_dim = 2, 2

# Gradient descent parameters
epsilon = 0.05
reg_lambda = 0.01

# Calculate loss of the neural model
def calculate_loss(model):
	W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2

    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Calculating the loss
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)

    # Add regulatization term to loss
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss

def predict(model, x):
	W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
	# forward propagation
	z1 = x1.dot(W1) + b1
	a1 = np.tanh(z1)
	z2 = a1.dot(W2) + b2
	output_scores = np.exp(z2)
	probs = exp_scores / np.sum(output_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)


