#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 10:48:32 2022

@author: jeshli
"""


import torch
import numpy as np
import Aleatoric_Model
import matplotlib.pyplot as plt



#x = np.linspace(0,100,num=50).repeat(100)
#s = np.linspace(10,1,num=50).repeat(100)
#y = np.random.normal(np.zeros(x.shape),s)
x = np.linspace(1,100,num=50).repeat(100)
s = np.linspace(1,2,num=50).repeat(100)
y = np.random.normal(x,s)
t = np.linspace(1,100)

#s = np.linspace(1,10,num=50).repeat(100)
#y = np.random.normal(x,np.exp(s))
#y = y / y.var()

Y = torch.from_numpy(y).float().unsqueeze(-1)
X = torch.from_numpy(x).float().unsqueeze(-1)


AM = Aleatoric_Model.Aleatoric_Regression_Model()
AM.fit(X,Y,exponential=False,verbose=True,lr=1e-2,epochs=3000,show_at=500)
u,s = AM.predict(torch.from_numpy(t).float().unsqueeze(-1))


plt.scatter(x,y)
plt.plot(t,u.detach().numpy(), color='r')
plt.plot(t,u.detach().numpy() + 3 * s.detach().numpy(),color='orange')
plt.plot(t,u.detach().numpy() - 3 * s.detach().numpy(),color='orange')
plt.title('Linear Variance - Linear Variance Model')
plt.show()



################### this design with exponential s (self calc still works)##########

AM = Aleatoric_Model.Aleatoric_Regression_Model()
AM.fit(X,Y,exponential=True,verbose=True,lr=1e-4,epochs=3000,show_at=500)
u,s = AM.predict(torch.from_numpy(t).float().unsqueeze(-1))


plt.scatter(x,y)
plt.plot(t,u.detach().numpy(), color='r')
plt.plot(t,u.detach().numpy() + 3 * s.detach().numpy(),color='orange')
plt.plot(t,u.detach().numpy() - 3 * s.detach().numpy(),color='orange')
plt.title('Linear Variance - Exponential Variance Model')
plt.show()



x = np.linspace(1,100,num=50).repeat(100)
#s = (np.linspace(1,10,num=50)**2).repeat(100)
s = np.exp(np.linspace(1,5,num=50)).repeat(100)
y = np.random.normal(x,s)
y /= y.std()
Y = torch.from_numpy(y).float().unsqueeze(-1)
X = torch.from_numpy(x).float().unsqueeze(-1)

AM = Aleatoric_Model.Aleatoric_Regression_Model()
AM.fit(X,Y,exponential=False,verbose=True,lr=1e-2,epochs=3000,show_at=500)
u,s = AM.predict(torch.from_numpy(t).float().unsqueeze(-1))


plt.scatter(x,y)
plt.plot(t,u.detach().numpy(), color='r')
plt.plot(t,u.detach().numpy() + 3 * s.detach().numpy(),color='orange')
plt.plot(t,u.detach().numpy() - 3 * s.detach().numpy(),color='orange')
plt.title('Exponential Variance - Linear Variance Model')
plt.show()



################### this design with exponential s (self calc still works)##########

AM = Aleatoric_Model.Aleatoric_Regression_Model()
AM.fit(X,Y,exponential=True,verbose=True,lr=1e-4,epochs=3000,show_at=500)
u,s = AM.predict(torch.from_numpy(t).float().unsqueeze(-1))


plt.scatter(x,y)
plt.plot(t,u.detach().numpy(), color='r')
plt.plot(t,u.detach().numpy() + 3 * s.detach().numpy(),color='orange')
plt.plot(t,u.detach().numpy() - 3 * s.detach().numpy(),color='orange')
plt.title('Exponential Variance - Exponenital Variance Model')
plt.show()
