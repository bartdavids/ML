# -*- coding: utf-8 -*-
"""
Created by: bartdavids: https://github.com/bartdavids/ML/

These helped a lot:
    https://satishgunjal.com/multivariate_lr/
"""
import numpy as np

class LogisticRegression():
    """
    Class for logistic regression.
    """  
    def __init__(self, i, labels, 
                 learning_rate = 0.01,
                 beta1 = 0.8, beta2 = 0.999, eps = 1e-9,
                 normalization = 'linear',
                 loss = 'mean squared error',
                 optimizer = 'gradient descent'):
        """
        Initialize the model

        Parameters
        ----------
        i : 2D numpy array containing numbers (int or float)
            A numpy array with the shape: m,n.
            Where m is the amount of samples and n is the amount 
            of variables.
        labels : 1D numpy array containing number (in or float) with a length of m (the amount of samples)
            The actual values. Can be put in as int/float, but also string
        learning_rate : float, optional
            The learning rate for the model. The default is 0.01.
        beta1 : float, optional
            Hyperparameter for the ADAM method. The default is 0.8.
        beta2 : float, optional
            Hyperparameter for the ADAM method. The default is 0.999.
        eps : float, optional
            Hyperparameter for the ADAM method. The default is 1e-9.
        normalization : string, optional
            Mode of normalization of the input parameters. The default is 'linear'.
            Other options are: 'z-score' and 'range'.
        loss : string, optional
            Type of loss function to be used. The default is 'mean squared error'.
            Other options are: 'cross entropy'
        optimizer : string, optional
            Type of optimizer. The default is 'gradient descent'.
            Other options are: 'adam'

        Returns
        -------
        None.

        """
        
        # Some checks
        if i.shape[0] != labels.shape[0]:
            print(f'The length of the input and its labels do not match up: {i.shape}, {labels.shape}')
            
        # Select the normalizer based on the input and normalize
        if normalization.lower() == 'linear':
            self.normalization = self.norm_0_to_1
        elif normalization.lower() == 'z-score':
            self.normalization = self.norm_variance
        elif normalization.lower() == 'range':
            self.normalization = self.norm_range
        else:
            raise(Exception(f'The normalization method {normalization} is not available'))
        
        self.i = self.normalization(i)
        
        # Assign the loss function based on the input
        if loss.lower() == 'mean squared error' or loss.lower() == 'mse':
            self.loss = self.loss_mse
        elif loss.lower() == 'cross entropy':
            self.loss = self.loss_cross_entropy
        else:
            raise(Exception(f'The normalization method {normalization} is not available'))
        
        # assign m and n        
        self.m = i.shape[0] # The amount of samples
        self.n = i.shape[1] # The amount of features without the bias
        
        # Add a column of ones for the bias
        self.i = np.hstack((np.ones((self.m, 1)), self.i))
        
        # Now, the labels are a list of string or numbers,
        # This makes them into a 2D numpy array where the classes are
        # an extra dimension, so the labels have the shape (m, j), where m 
        # is the amount of samples and j is the amount of categories
        self.label_names = np.unique(labels)
        label_copy = np.zeros(labels.shape, dtype = int)
        for pos, val in enumerate(self.label_names):
            label_copy[np.where(labels == val)] = int(pos)
            
        # The label is represented by a numpy array of 0's for the
        # classes it's not, and a 1 where it it is.
        self.labels = np.eye(len(self.label_names))[label_copy]
        
        # Initiate the weights
        self.theta = np.ones((self.n + 1, self.labels.shape[1]))
        
        # Assign optimizer
        if optimizer.lower() == 'gradient descent':
            self.optimizer = self.GradientDescent
        elif optimizer.lower() == 'adam':
            self.optimizer = self.Adam
            
            # Initialize momentii m1 and m2 for weights, 
            # mb1, mb2 for the biases
            self.m1 = np.zeros(self.theta.shape)
            self.m2 = np.zeros(self.theta.shape)
            
            # set additional hyperparameters
            self.beta1 = beta1
            self.beta2 = beta2
            self.eps = eps
        else:
            raise(Exception(f'The optimizer method {optimizer} is not available'))    
        
        # Set some metrics
        self.history = []
        self.epoch = 0
        
        # The learning rate
        self.learning_rate = learning_rate
     
    def norm_0_to_1(self, pre_norm):
        # normalize between 0 and 1
        self.norm_max = np.max(pre_norm, axis = 0)
        self.norm_min = np.min(pre_norm, axis = 0)
        norm = (pre_norm - self.norm_min)/(self.norm_max - self.norm_min)
        return norm
    
    def norm_range(self, pre_norm):
        # normalize over range: (x - mean)/(max - min)
        self.mean = np.mean(pre_norm, axis = 0)
        self.range = np.max(pre_norm, axis = 0) - np.min(pre_norm, axis = 0)
        norm = (pre_norm - self.mean)/self.range
        return norm
    
    def norm_variance(self, pre_norm):
        # Use the z-score normalization:
        # (x - mean)/(standard deviation)
        self.mean = np.mean(pre_norm, axis = 0)
        self.var = np.mean((pre_norm - self.mean)**2, axis = 0)
        norm = (pre_norm - self.mean)/np.sqrt(self.var)
        return norm
    
    def forward(self, inp):
        # Predict using the sigmoid function
        return 1/(1 + np.exp(-(inp @ self.theta)))

    def loss_cross_entropy(self, y_hat, y):
        # The binary cross entropy loss function
        return -np.sum(y * np.log(y_hat + 1e-9) + (1 - y) * np.log(1 - y_hat + 1e-9))/y.shape[0]
    
    def loss_mse(self, y_hat, y):
        # The mean squared error loss function
        return np.sum(np.square(y_hat - y))/(2*y.shape[0])
    
    def der_loss_mse(self, y_hat, y):
        # The derivative of the mean squared error loss function
        return np.subtract(y_hat, y)
    
    def GradientDescent(self, gradient):
        # Use the gradient to update the weights
        self.delta_theta = self.i.T @ gradient # calculate the update values of the weights
        self.theta -= self.learning_rate/self.m * self.delta_theta # update the weights
    
    def Adam(self, gradient):
        # ADAM method for gradient descent
        # update momentum
        self.m1 = self.beta1 * self.m1 + (1.0 - self.beta1) * (self.i.T @ gradient)/self.m
        self.m2 = self.beta2 * self.m2 + (1.0 - self.beta2) * (self.i.T @ np.power(gradient, 2))/self.m
        
        # Bias correction
        m1hat = self.m1 / (1. - np.power(self.beta1, self.epoch + 1))
        m2hat = self.m2 / (1. - np.power(self.beta2, self.epoch + 1))
        
        # The gradients for the update of the weights            
        self.delta_theta = m1hat/(np.sqrt(np.abs(m2hat)) + self.eps) 
        
        # Update the weights
        self.theta -= self.learning_rate * self.delta_theta # update the weights
        
    def Train(self, training_epochs):
        # Train the model
        for _ in range(training_epochs):
            # Make the prediction
            self.y_hat = self.forward(self.i) 
            
            # compute the cost and safe it in the history for metrics. NB: not for training
            cost = self.loss(self.y_hat, self.labels) 
            self.history.append(cost) 
            
            # Compute the gradient of the cost function
            self.gradient = self.der_loss_mse(self.y_hat, self.labels)
            
            # Update the weights using the optimizer
            self.optimizer(self.gradient)
            
            self.epoch+=1
            
    def set_learning_rate(self, learning_rate):
        # Set the learning model after initiating or running the model
        self.learning_rate = learning_rate
        
    def Predict(self, p):
        # Make a prediction using the model
        # p has to be the same format as i when initializing the model:
        # (m, n) where m is the amount of samples and n the amount
        # of variables
        p = self.normalization(p)
        p = np.hstack((np.ones((p.shape[0], 1)), p)) 
        
        # It returns the approximate value for the prediction as a float
        return self.forward(p)