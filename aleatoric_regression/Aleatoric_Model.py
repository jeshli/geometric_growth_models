#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 10:33:29 2022

@author: jeshli
"""


import torch
import torch.nn as nn
import torch.optim as optim


class Linear_Variance(nn.Module):
    
    def __init__(self,N_Features):
        super(Linear_Variance, self).__init__()
    
        self.mu = nn.Linear(N_Features,1)        
        self.mu.weight.data.fill_(0)
        self.mu.bias.data.fill_(0)
    
        self.sigma = nn.Linear(N_Features,1)        
        self.sigma.weight.data.fill_(0)        
        self.softplus = nn.Softplus()
        self.sigma.bias.data.fill_(1)
    
    def forward(self, Features):
        
        mu = self.mu(Features)
        sigma = self.softplus( self.sigma(Features) )            
        return mu, sigma


class Exponential_Variance(nn.Module):
    
    def __init__(self,N_Features):
        super(Exponential_Variance, self).__init__()
    
        self.mu = nn.LineExponentialar(N_Features,1)        
        self.mu.weight.data.fill_(0)
        self.mu.bias.data.fill_(0)
    
        self.sigma = nn.Linear(N_Features,1)        
        self.sigma.weight.data.fill_(0)        
        self.sigma.bias.data.fill_(0)
    
    def forward(self, Features):
        
        mu = self.mu(Features)
        sigma = torch.exp( self.sigma(Features) )
        return mu, sigma



class No_Variance(nn.Module):
    
    def __init__(self,N_Features):
        super(No_Variance, self).__init__()
    
        self.mu = nn.Linear(N_Features,1)        
        self.mu.weight.data.fill_(0)
        self.mu.bias.data.fill_(0)
    
    def forward(self, Features):
        
        mu = self.mu(Features)
        return mu, torch.zeros(Features.shape[0],1)



    
class Aleatoric_Regression_Model:
    
    def __init__(self):
        self.model = None

    def loss(self,u,s,y):
        L = ( ( (u - y)**2 ) / s**2 ).mean() + torch.log( s**2 ).mean()
        return L
      
    def MSE_loss(self,u,y):
        L = ( ( (u - y)**2 ) ).mean()
        return L        
    
    def fit(self,X,Y,exponential=False,no_variance=False,verbose=False,epsilon=1e-6,lr=1e-3,epochs=250,show_at=100):        
        if exponential:
            self.model = Exponential_Variance(X.shape[1])
        elif no_variance:
            self.model = No_Variance(X.shape[1])            
        else:
            self.model = Linear_Variance(X.shape[1])
            
        # do iterative learning
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(.9, .999))          
        tracked_loss = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            u,s = self.model.forward(X)
            #print(u,s,Y)
            if no_variance:
                error = self.MSE_loss(u.squeeze(-1),Y.squeeze(-1))  
            else:
                error = self.loss(u.squeeze(-1),s.squeeze(-1),Y.squeeze(-1))  
            #print(error)

            if epoch % show_at == 0: 
                tracked_loss.append(error.item());
                if verbose:
                    print('epoch', epoch, ':', error.item())   
                if len(tracked_loss) > 1:
                    if tracked_loss[-2] < tracked_loss[-1] + epsilon:
                        break
                        #pass
            
            error.backward()
            optimizer.step()
        if verbose:
            print('epoch', epoch, ':', error.item())  
    

    def predict(self,X):
        u,s = self.model(X)
        return u,s