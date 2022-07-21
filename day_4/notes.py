# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 09:24:16 2022

@author: Elle Lavichant
"""
names= ['Ahmed','Becky','Cantor']
ages= [21,30,45]
favorite_colors=['Pink', 'Grey','Blue']
print(list(zip(names, ages ,favorite_colors)))
for name,age,color in zip(names,ages,favorite_colors): 
    print(names,ages, color)
#bring in the list together
#%%
from datetime import datetime
num_of_days=10
years=[2009]*num_of_days
months=[12]*num_of_days
days=list(range(1,11))
times=[datetime(year,month,day)
       for year,month,day
       in zip(years,months,days)]

for time in times:
    print(time.isoformat())
#%%pcolormesh
import numpy as np
import matplotlib.pyplot as plt

num_of_x= 10
num_of_y= 20
x=np.linspace(0,1,num_of_x)
y=np.linspace(0,1,num_of_y)
z=np.random.randn(num_of_y,num_of_x)
plt.pcolormesh(x,y,z)
plt.colorbar()
#%% NetCDF

import netCDF4 as nc
dataset= nc.Dataset(r'C:\Users\ellel\Downloads\Space_Summer\Data\wfs.t12z.ipe05.20220721_140000.nc')
print(dataset)
dataset['tec'][:] #How you get the numpy array of the data
dataset['tec'].units # How you get the units of data
#%%