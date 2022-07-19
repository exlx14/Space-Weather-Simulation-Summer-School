# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 14:31:22 2022

__author__: Elle Lavichant
"""
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import argparse
def read_ascii_file(filename,index):
    "This reads an ascii file of omni data"
    with open(filename) as f:
        #year=[]
        #day=[]
        #hour=[]
        #minute=[]
        #data=[]
        #time=[]
        data_dic={"time":[],"year":[],"day":[],"hour":[],"minute":[],"symh":[]}
        for line in f:
            tmp=line.split()
            data_dic["year"].append(int(tmp[0])) #convert string to intergers 
            data_dic["day"].append(int(tmp[1]))
            data_dic["hour"].append(int(tmp[2]))
            data_dic["minute"].append(int(tmp[3]))
            data_dic["symh"].append(int(tmp[4]))
            #create datetime in each line
            time0= dt.datetime(int(tmp[0]),1,1,int(tmp[2]),int(tmp[3]),0)\
                 + dt.timedelta(days=int(tmp[1])-1)
            data_dic["time"].append(time0)
            #data_dic["year"].append(float(tmp[index]))
        return data_dic
filename='omni_min_case_2013.lst'
index=-1
data= read_ascii_file(filename,index)
time= data["time"]
data1=data["symh"]
fig,ax= plt.subplots()
ax.plot(time,data1,marker='.',c='gray',
       label='All Events', alpha=0.5)
ax.set_xlabel('Year of 2013')
ax.set_ylabel('SYMH (nt)')
ax.grid(True)
ax.legend
outfile='plot_graph2013.png'
print('Writing file:'+outfile)
plt.savefig(outfile)
plt.close()



