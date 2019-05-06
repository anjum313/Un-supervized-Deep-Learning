# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 18:33:31 2018

@author: anjum
"""

# Building RBM

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the data set
movies = pd.read_csv('ml-1m/movies.dat',
                     sep = '::',
                     header = None,
                     engine = 'python',
                     encoding ='latin-1') 
                     
users = pd.read_csv('ml-1m/users.dat',
                     sep = '::',
                     header = None,
                     engine = 'python',
                     encoding ='latin-1')
 
ratings = pd.read_csv('ml-1m/ratings.dat',
                     sep = '::',
                     header = None,
                     engine = 'python',
                     encoding ='latin-1')  
# Importing the training and test  set

training_set = pd.read_csv('ml-100k/u1.base',delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test',delimiter = '\t')
test_set = np.array(test_set , dtype = 'int')

# getting the number of users and the movies

nb_users = max(max(training_set[:,0]),max(test_set[:,0]))
nb_movies =max(max(training_set[:,1]),max(test_set[:,1])) 

# converting the data set in to array with users as lines and movies as columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

# converting the list of lists in to torch tensers
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Converting the data in to binary ratings 0 = not liked 1 = likes

training_set[training_set == 0] = -1 # np reviews equals to -1
training_set[training_set == 1] =  0
training_set[training_set == 2] =  0
training_set[training_set >= 3] =  1

test_set[test_set == 0] = -1 # np reviews equals to -1
test_set[test_set == 1] =  0
test_set[test_set == 2] =  0
test_set[test_set >= 3] =  1

# Creating the architecture of RBM.
class RBM():
    def __init__(self, nv , nh): # Inetilizing num of hiden and visible nodes as inputs
        self.W = torch.randn(nh , nv) # probabilty of the visible nodes according to the hidden nodes
        self.a = torch.randn(1, nh) # Bias for hidden node in form of 2D tensor where 1 respons to batch and nh reponds to bias
        self.b = torch.randn(1, nv) # Bias for hidden node in form of 2D tensor where 1 respons to batch and nv reponds to bias
    def sample_h(self, x):    
        # sampling the hidden nodes according to the condition probality
        # of a hidden node as per given visible node Where x EQUALS TO VISIBLE NODES
        wx = torch.mm(x, self.W.t()) # computing the weights times the neurons
        activation = wx + self.a.expand_as(wx) # expand is used to convert a in to dimensions of wx
        p_h_given_v = torch.sigmoid(activation) 
        return p_h_given_v , torch.bernoulli(p_h_given_v)
    def sample_v(self, y):
        wy = torch.mm(y, self.W) # computing the weights times the neurons
        activation = wy + self.b.expand_as(wy) # expand is used to convert a in to dimensions of wx
        p_v_given_h = torch.sigmoid(activation) 
        return p_v_given_h , torch.bernoulli(p_v_given_h)
    def train(self, v0,vk,ph0,phk):
        # train calculates contrastive divergence
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0) # zero is to get the 2d tensor
        self.a += torch.sum((ph0 - phk),0)

# Inetializing parameters for RBM class
nv = len(training_set[0])   # Number of movies        
nh = 100
batch_size = 100 # 1 for online learning
rbm = RBM(nv,nh)

# training the RBM
walks = 10 # number of walk steps required for gibbs sampling
nb_epoch = 10
for epoch in range(1, nb_epoch +1):
    train_loss = 0
    s=0. # counter for the loss
    for id_users in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_users:id_users + batch_size] # Target (Taking users for the batch size)
        v0 = training_set[id_users:id_users + batch_size] # Actual users to get the loss
        ph0,_ = rbm.sample_h(v0)
        
        # Making loop for K step contrastive divergence
        for k in range(walks):
            _,hk = rbm.sample_h(vk) # getting the probabilities for first hidden nodes
            _,vk = rbm.sample_v(hk) # getting the update for the visible node
            vk[v0<0] = v0[v0<0] # Removng the cells with no ratings
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0,vk,ph0,phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0]-vk[v0>=0]))
        s+= 1.
    print('epoch:  ' + str(epoch)+' loss:  '+ str(train_loss/s))
        
# Testing the RBM.
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print('test loss: '+str(test_loss/s))  
            
            
        
        
        
    
    
