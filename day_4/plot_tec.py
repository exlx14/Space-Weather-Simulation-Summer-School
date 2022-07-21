# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 10:11:08 2022

__author__: Elle lavichant 
"""
import netCDF4 as nc
import matplotlib.pyplot as plt
import argparse
def plot_tec(dataset, figsize=(12,6)):
    "This is a function that will plot the total electron content"
    z=dataset['tec'][:] #obtain the data for the total electron 
    fig,ax= plt.subplots(1,figsize=figsize)
    x= dataset['lon'][:] #this is the xaxis
    y=dataset['lat'][:] #this is the yaxis
    p=ax.pcolormesh(x,y,z) #plot the colorbar 
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    fig.colorbar(p)
    ax.set_title('The total electron')
    
    return fig,ax
def parse_args():
    parser = argparse.ArgumentParser(description = \
                                 'This function is the convert the file to png')
  
    # -filein: the file that is going in to convert to png
    parser.add_argument('-filein', nargs='+',\
                      help = 'This is the file going into the program',
                      type=str)

    args = parser.parse_args()
    return args
##main code
if __name__ == '__main__':
    args = parse_args()
    for filein in args.filein:
        dataset= nc.Dataset(filein) #call the dataset
        plot_tec(dataset) #plot the function 
        fileout= filein + '.png' #save the file to a picture 
        plt.savefig(fileout) #save the figure 