# -*- coding: utf-8 -*-
"""
Created by: bartdavids: https://github.com/bartdavids/ML/

Python executable to initate the Linear and Logistic Regression packages and
compare linear and logistic regression for the purpose of classification.

In addition some Adam vs. Gradient Descent is tested.

"""
import os
os.chdir(r'C:\Users\bart_\Documents\Python Scripts\ML linear regression')

# Load relevant packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the classes to be used
from Linear_regression import LinearRegression
from Logistic_regression import LogisticRegression

# Load the data
data = pd.read_csv('winequality-white.csv', sep = ';')

# Make sure the sample variabels (i) and labels are seperated
i = data[data.columns[:-1]]
labels = data[data.columns[-1]]

parameter_names = i.columns

# The input for the models is a numpy array with the shape (m, n), where m is the 
# amount of samples and n is the amount of variables
i = np.array(i)

# The labels can be a string, int or float. But it has to be a numpy array
# with size m, where m is the amount of samples and equal to the amount of
# samples in the one where the sample variables are stored
labels = np.array(labels, dtype = str)

# Initialize hyperparameters. For ADAM the default values are used.
epochs = 250
learning_rate = 0.05

# Initialize the model
logistic_model = LogisticRegression(i, labels, learning_rate = learning_rate, normalization = 'Z-score', optimizer = 'adam')

# Train the model
logistic_model.Train(epochs)

# The Predict function of the logistic model returns the probability 
# of the the class in a 2D numpy array with dimensions: (m, j). 
# Where m is the amount of samples and j is the amount of classes. 
# The highest number in each column refers to the position of the label 
# in the model.label_name variable
logistic_probability = logistic_model.Predict(i) 
logistic_prediction = np.argmax(logistic_probability, axis = 1)
logistic_prediction = logistic_model.label_names[logistic_prediction]

# Check it's accuracy
logistic_correct = np.sum(logistic_prediction == labels)/logistic_model.m*100
print(f'The logistic model has an accuracy of {np.round(logistic_correct,2)}%')

# The linear model returns the approximate value of the class as a 1D numpy array
linear_model = LinearRegression(i, labels, learning_rate = learning_rate, normalization = 'Z-score', optimizer = 'adam')
linear_model.Train(epochs)
linear_predict = np.round(linear_model.Predict(i)) 
linear_correct = np.sum(linear_predict == labels.astype(int))/linear_model.m*100 
print(f'The linear model has an accuracy of {np.round(linear_correct,2)}%')

# Plot the loss (mean squared error) of the two scenarios against each other
plt.plot(list(range(logistic_model.epoch)), logistic_model.history, 'k')
plt.plot(list(range(linear_model.epoch)), linear_model.history, 'r--')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.gca().legend(['logistic regression', 'linear regression'])
plt.show()

# The adam model
adam_model = LogisticRegression(i, labels, learning_rate = learning_rate, normalization = 'Z-score', optimizer = 'adam')
adam_model.Train(epochs)
adam_probability = adam_model.Predict(i) 
adam_prediction = np.argmax(adam_probability, axis = 1)
adam_prediction = adam_model.label_names[adam_prediction]
adam_correct = np.sum(adam_prediction == labels)/adam_model.m*100
print(f'The adam model has an accuracy of {np.round(adam_correct,2)}%')

# The gradient descent model
gd_model = LogisticRegression(i, labels, learning_rate = learning_rate, normalization = 'Z-score', optimizer = 'gradient descent')
gd_model.Train(epochs)
gd_probability = gd_model.Predict(i) 
gd_prediction = np.argmax(gd_probability, axis = 1)
gd_prediction = gd_model.label_names[gd_prediction]
gd_correct = np.sum(gd_prediction == labels)/gd_model.m*100
print(f'The gradient descent model has an accuracy of {np.round(gd_correct,2)}%')

# Plot the loss (mean squared error) of the two scenarios against each other
plt.plot(list(range(adam_model.epoch)), adam_model.history, 'k')
plt.plot(list(range(gd_model.epoch)), gd_model.history, 'r--')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.gca().legend(['ADAM', 'Gradient Descent'])
plt.show()

