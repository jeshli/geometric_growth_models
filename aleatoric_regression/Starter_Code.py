#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 10:51:36 2022

@author: jeshli
"""



import pandas as pd
import torch
import numpy as np
import Aleatoric_Model
import matplotlib.pyplot as plt


no_variance=False
lr=1e-1

Synthetic_Demo = False
Cubic = False

if Synthetic_Demo:
    DF = pd.DataFrame()
    Time_Data = pd.date_range(start='2021-05-21', end='today')
    DF['Date'] = Time_Data
    t = ((Time_Data - Time_Data[0]).days / 365).values
    s = np.linspace(1e-1,2e-1,num=t.shape[0])
    y = np.random.normal(t,s)
    y /= y.std()
    log_Cycle_Burn_Data = y
    Cycle_Burn_Data = np.exp(log_Cycle_Burn_Data)
    DF['Cycles Burned'] = Cycle_Burn_Data



DF = pd.concat([pd.read_csv('../data/daily/Dates.csv', delimiter='|'), pd.read_csv('../data/daily/CyclesBurned.csv', delimiter='|')], axis=1)
DF['Cycles Burned'] = DF['Cycles Burned'].str.replace(',','')
DF['Cycles Burned'] = pd.to_numeric(DF['Cycles Burned'].values)
DF['Date'] = pd.to_datetime(DF['Date'])
DF = DF[DF['Date']>'2022-02-01']  #filter out an outlier spike                  142%    2%
#DF = DF[DF['Date']<'2022-10-01']                                               116%  -92%
#DF = DF[DF['Date']>'2022-10-01'] #filter out to chapter temporal shift         632%   42%  
DF.reset_index(inplace=True,drop=True)

###############################################################################

Cycle_Burn_Data = DF['Cycles Burned'].values
Time_Data = DF['Date']
t = ((Time_Data - Time_Data[0]).dt.days / 365).values
log_Cycle_Burn_Data  = np.log(Cycle_Burn_Data)
y = log_Cycle_Burn_Data


#y = np.log(log_Cycle_Burn_Data)

Y = torch.from_numpy(y).float().unsqueeze(-1)
#X = torch.from_numpy(t).float().unsqueeze(-1)

if Cubic:
    T = np.concatenate([np.expand_dims(t,-1), np.expand_dims(t,-1)**2, np.expand_dims(t,-1)**3],axis=1)
else:
    T = np.expand_dims(t,-1)
X = torch.from_numpy(T).float()


AM = Aleatoric_Model.Aleatoric_Regression_Model()
AM.fit(X,Y,exponential=False,no_variance=no_variance,verbose=True,lr=lr,epochs=5000,show_at=500)
u,s = AM.predict(torch.from_numpy(T).float())


plt.plot(Time_Data,y)
plt.plot(Time_Data,u.detach().numpy(), color='r')
plt.plot(Time_Data,u.detach().numpy() + 2 * s.detach().numpy(),color='orange')
plt.plot(Time_Data,u.detach().numpy() - 2 * s.detach().numpy(),color='orange')
plt.title('Log Space - Modeled Burn Rate Prediction')
plt.xlabel('Date')
plt.xticks(rotation = 45) 
plt.ylabel('Log Cycles Burned')
plt.legend(['data','mean','2 sigma'])
plt.show()


plt.plot(Time_Data,Cycle_Burn_Data)
plt.plot(Time_Data,np.exp(u.detach().numpy()), color='r')
plt.plot(Time_Data,np.exp(u.detach().numpy() + 2 * s.detach().numpy()),color='orange')
plt.plot(Time_Data,np.exp(u.detach().numpy() - 2 * s.detach().numpy()),color='orange')
plt.title('Modeled Burn Rate Prediction')
plt.xlabel('Date')
plt.xticks(rotation = 45) 
plt.ylabel('Cycles Burned')
plt.legend(['data','mean','2 sigma'])
plt.show()

if Cubic:
    print('Exponential Rate of Increase YoY', str( np.round(100 * AM.model.mu.weight.detach().numpy().ravel(),2) ) + '%')
    if not no_variance:
        print('Variance Rate of Increase YoY', str( np.round(100 * AM.model.sigma.weight.detach().numpy().ravel(),2) ) + '%')
else:
    print('Exponential Rate of Increase YoY', str( np.round(100 * AM.model.mu.weight.item(),2) ) + '%')
    if not no_variance:
        print('Variance Rate of Increase YoY', str( np.round(100 * AM.model.sigma.weight.item(),2) ) + '%')
