# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 13:06:33 2018

@author: anjum
"""

# Creating SOM for fraud detection in credit card application data set

# Importing the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the data set

dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[: , :-1].values
y = dataset.iloc[:,-1].values

# Normalizing the data

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X_scaled = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
Som = MiniSom(x = 10, y = 10 ,input_len = 15, sigma = 3.0, learning_rate = 0.5)

# as per alogrithm we must inetilize the weights.
Som.random_weights_init(X_scaled)

# Training the som 
Som.train_random(data = X_scaled, num_iteration = 1000)

# Visvulaizing the results
from pylab import bone,pcolor,colorbar,plot,show
# Getting an empty plot
bone()
# Getting mean inter neuron distances
pcolor(Som.distance_map().T)
colorbar()

# Getting actual coustomers as markers from response variable
markers =['o','s']
colors = ['r' , 'g']
for i, x in enumerate(X_scaled):
    # getting the winning node of coustomer x
    w =Som.winner(x)
    # plotting markers at the centre of the winning nodes
    # 0.5 is added to get the markers in the centre
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2 )
show()    

# Finding the fraudalant transcations
mappings = Som.win_map(X_scaled) # getting the nodes mappings
fraud =mappings[(4,2)]
frauds =sc.inverse_transform(fraud)





