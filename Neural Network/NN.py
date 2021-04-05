"""
created by: bartdavids
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import os

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)  

"""
These helped a lot:
    https://www.youtube.com/watch?v=9RN2Wr8xvro&feature=share&ab_channel=BotAcademy
    https://www.youtube.com/watch?v=ILsA4nyG7I0&ab_channel=BotAcademyBotAcademy
    https://www.youtube.com/watch?v=aircAruvnKk&ab_channel=3Blue1Brown
    https://towardsdatascience.com/understanding-backpropagation-algorithm-7bb3aa2f95fd
"""

#%% Get MNIST dataset
with np.load("data/mnist.npz") as f:
    i, labels = f["x_train"], f["y_train"]

# scale to - to 1
i = i.astype("float64")
i /= 255
i[i<0.5] = 0
i[i>=0.5] = 1

# The images are now in columns and rows (shape = 60000, 28, 28), just 1 vector is needed per image (shape = 60000, 784)
i = np.reshape(i, (i.shape[0], i.shape[1] * i.shape[2]))
i.shape += (1,)

# Convert labels to categrories
labels = np.eye(10)[labels]
labels.shape += (1,)
#%% Initiate neural network
# Hidden layers in list: every number is the amnount neurons in that layer. 
# So [10, 10] means that there are going to be two layers with 10 neurons each
hidden = [20]

# neurons = n
n = [len(i[0])] + hidden + [len(labels[0])]

# initialize weights (w) between -0.5 and 0.5 and biases (b) as 0
w = [np.random.uniform(-0.5, 0.5, (n[h+1], n[h])) for h in range(len(n)-1)]
w = np.asarray(w, dtype=object)
b = np.asarray([np.zeros((n[h + 1], 1)) for h in range(len(n)-1)])
b = np.asarray(b, dtype=object)

# Set hyperparameters
learn_rate = 0.01

epochs = 3

#%% define some functions
# Activation functions and their derivatives
def sigmoid(r_pre):
    # sigmoid activation function
    return 1 / (1 + np.exp(-r_pre))

def relu(r_pre):
    # relu activation function
    # causes matmul overflow when used
    return np.max((np.zeros(r_pre.shape), r_pre), axis = 0)

def der_sigmoid(r_der):
    # sigmoid activation function derivative for backprop
    return (r_der * (1 - r_der))

def der_relu(r):
    # relu activation function derivative for backprop
    der_r = r.copy()
    der_r[der_r<=0] = 0
    der_r[der_r>0] = 1
    return der_r

def tanh(r):
    return ((np.exp(r) - np.exp(-r))/(np.exp(r) + np.exp(-r)))

def der_tanh(r):
    return 1 - ((np.exp(r) - np.exp(-r))**2) / ((np.exp(r) - np.exp(-r))**2)

# Cost functions and their derivatives
def mse(y, r):
    return 1 / (2 * len(r)) * np.sum((y - r) ** 2, axis=0)

def der_mse(y, r):
    return 1*(y - r)
    
def mae(y, r):
    return 1 / (2 * len(r)) * np.sum(abs(y - r) , axis=0)

def der_mae(y, r):
    e = y-r
    e[e<0] = -1
    e[e>0] = 1
    return e

# Backpropogation algorithms
def GradientDescent(batch_r, batch_l_, w, b, 
                    m1, m2, bm1, bm2, # I have no excuse
                    learn_rate = 0.01):
    # compute gradients for each batch and update weights and biases
    
    grad_w = [np.zeros(_.shape) for _ in w] # list of gradients for eacht activated neuron (so all except input)
    grad_b = [np.zeros(_.shape) for _ in b]
    
    grad = list(range(len(w)))
    grad_o = der_loss(batch_r[-1], batch_l_) # derrivative of the cost function for gradiënt descent
    
    #grad_o = grad_o.sum(axis=0) # accumulate gradients  
    grad[-1] = grad_o
    
    # Backpropagation output -> hidden (cost function derivative)
    grad_w[-1] = grad_o @ np.transpose(batch_r[-2], axes = [0,2,1]) 
    grad_b[-1] = grad_o
    
    for hli in list(range(len(w)-1))[::-1]: # for all the connections beteen the hidden layers, in reverse
        grad_h = w[hli + 1].T @ grad[hli + 1] * derivitive_activation(batch_r[hli + 1]) #derive gradients from previous gradient in backprop
        grad_w[hli] = grad_h @ np.transpose(batch_r[hli], axes = [0,2,1])
        grad_b[hli] = grad_h
        
        grad[hli] = grad_h
    
    # Average of the gradients per batch
    grad_w = [_.mean(axis = 0) for _ in grad_w]
    grad_b = [_.mean(axis = 0) for _ in grad_b]
    
    w += -learn_rate * np.asarray(grad_w)
    b += -learn_rate * np.asarray(grad_b)

    return w, b
#%%
def Adam(batch_r, batch_l_, w, b, 
         m1, m2, bm1, bm2,
         learn_rate = 0.01, beta1 = 0.8, beta2 = 0.999, eps = 1e-8):
    # Backpropagation output -> hidden (cost function derivative)
    
    grad_w = [np.zeros(_.shape) for _ in w] # list of gradients for eacht activated neuron (so all except input)
    grad_b = [np.zeros(_.shape) for _ in b]
    grad = list(range(len(w))) # list of gradients of the error for eacht activated neuron (so all except input)
    
    grad_o = der_loss(batch_r[-1], batch_l_) # derrivative of the cost function for gradiënt descent
    grad_o = grad_o.sum(axis=0) # accumulate gradients 
    grad[-1] = grad_o
    
    # update momentum
    m1[-1] = beta1 * m1[-1] + (1.0 - beta1) * grad_o @ np.transpose(batch_r[-2], axes = [0,2,1]) 
    m2[-1] = beta2 * m2[-1] + (1.0 - beta2) * (np.power(grad_o, 2) @ np.transpose(batch_r[-2], axes = [0,2,1])) 
    bm1[-1] = beta1 * bm1[-1] + (1.0 - beta1) * grad_o
    bm2[-1] = beta2 * bm2[-1] + (1.0 - beta2) * np.power(grad_o, 2)
    
    # Bias correction
    m1hat = m1[-1] / (1. - np.power(beta1, epoch + 1))
    m2hat = m2[-1] / (1. - np.power(beta2, epoch + 1))
    bm1hat = bm1[-1] / (1. - np.power(beta1, epoch + 1))
    bm2hat = bm2[-1] / (1. - np.power(beta2, epoch + 1))
    m1hat = m1hat.sum(axis=0)
    m2hat =m2hat.sum(axis=0)
    bm1hat = bm1hat.sum(axis=0)
    bm2hat = bm2hat.sum(axis=0)
    
    # Update variable value            
    grad_w[-1] = m1hat/(np.sqrt(m2hat) + eps) 
    grad_b[-1] = bm1hat/(np.sqrt(bm2hat) + eps)
    
     # Backpropagation hidden layers (activation function derivative)
    for hli in list(range(len(w)-1))[::-1]: # for all the connections beteen the hidden layers, in reverse
        
        grad_h = w[hli + 1].T @ grad[hli + 1] * derivitive_activation(batch_r[hli + 1])
        grad[hli] = grad_h
        
        # update momentum
        m1[hli] = beta1 * m1[hli] + (1.0 - beta1) * grad_h @ np.transpose(batch_r[hli], axes = [0,2,1])
        m2[hli] = beta2 * m2[hli] + (1.0 - beta2) * (np.power(grad_h, 2) @ np.transpose(batch_r[hli], axes = [0,2,1]))
        bm1[hli] = beta1 * bm1[hli] + (1.0 - beta1) * grad_h 
        bm2[hli] = beta2 * bm2[hli] + (1.0 - beta2) * np.power(grad_h, 2)
        
        # Bias correction
        m1hat = m1[hli] / (1. - np.power(beta1, epoch + 1))
        m2hat = m2[hli] / (1. - np.power(beta2, epoch + 1))
        bm1hat = bm1[hli] / (1. - np.power(beta1, epoch + 1))
        bm2hat = bm2[hli] / (1. - np.power(beta2, epoch + 1))
        m1hat = m1hat.sum(axis=0)
        m2hat =m2hat.sum(axis=0)
        bm1hat = bm1hat.sum(axis=0)
        bm2hat = bm2hat.sum(axis=0)
        
        #update weights
        grad_w[hli] += m1hat/(np.sqrt(m2hat) + eps)
        grad_b[hli] += bm1hat/(np.sqrt(bm2hat) + eps)
    
    w += -learn_rate * np.asarray(grad_w)
    b += -learn_rate * np.asarray(grad_b)
    return w, b

#%% assign the functions to the activation and loss function
activation = 'sigmoid'
loss = 'mean squared error'
optimizer = 'adam'

if activation == 'sigmoid':
    activation = sigmoid
    derivitive_activation = der_sigmoid
elif activation == 'relu':
    activation = relu
    derivitive_activation = der_relu
elif activation == 'tanh':
    activation = tanh
    derivitive_activation = der_tanh
else:
    raise(Exception(f"The activation function {activation} is not available"))
    
if loss == 'mean squared error' or loss == 'mse':
    loss = mse
    der_loss = der_mse  
elif loss == 'mean absolute error' or loss == 'mae':
    loss = mae
    der_loss = der_mae
else:
    raise(Exception(f"The loss function {loss} is not available"))
    
if optimizer == 'gradient descent':
    optimizer = GradientDescent
elif optimizer == 'adam':
    optimizer = Adam# Initialize mmomentii m1 and m2 for weights, mb1, mb2 for the biases
    m1 = [np.zeros(x.shape) for x in w]
    m2 = [np.zeros(x.shape) for x in w]
    bm1 = [np.zeros(x.shape) for x in b]
    bm2 = [np.zeros(x.shape) for x in b]
else:
    raise(Exception(f"The optimizer {optimizer} is not available"))
    
    

#%% Train
corr = [] # store the improvement over epochs

batch_amount = 1000

batch_i = np.array_split(i, batch_amount)
batch_l = np.array_split(labels, batch_amount)

for epoch in range(epochs):
    correct = 0

    for batch in range(batch_amount): 
        batch_r = [batch_i[batch]] # The results are stored in a list of nx1 matrices for each layer
        
        # Forward propagation
        for lay in range(len(w)): 
            # Forward propagation input -> hidden = step 0.
            # Forward propagation hiden -> ouput = step -1.
            r_pre = b[lay] + w[lay] @ batch_r[lay]
            batch_r.append(activation(r_pre))
        correct += np.sum(np.equal(np.argmax(batch_r[-1], axis = 1), np.argmax(batch_l[batch], axis = 1))) #The highest number is chosen, "correct" is only to determine the amount of correct outcomes per epoch
        
        # Backpropagation
        w, b = optimizer(batch_r, batch_l[batch], w, b, m1, m2, bm1, bm2, learn_rate = learn_rate)   

    # Show accuracy for this epoch
    # percentage correct:
    cor = round((correct / i.shape[0]) * 100, 2)
    corr.append(cor)
    print(f"Epoch {epoch+1}: Accuracy = {cor}%")
plt.plot(list(range(1, epochs+1)), corr)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')

#%% Compare

# Which one do you want to check? From 1 to the length of the dataset.
index = 7
img = i[index+1]

o = []
o.append(img)
for lay in range(len(w)): 
    # Forward propagation input -> hidden = step 0.
    # forward propagation hiden -> ouput = step -1.
    r_pre = b[lay] + w[lay] @ o[lay]  # pre = pre-activation, between hidden and input 
    o.append(activation(r_pre))  

network_result = o[-1].argmax()
print(f"The neural network says its a {network_result}")

actual_result = np.argmax(labels[index+1])
print(f"The label says its a {actual_result}")

if actual_result == network_result:
    print("That's correct!")
else:
    print("Need more pushups")


