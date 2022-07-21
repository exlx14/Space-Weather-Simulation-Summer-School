# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 14:11:50 2022

@author: Elle Lavichant
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import argparse
def tiegcam_plot(alt):
    """This is a function that will generated a plot of the density at a
    certain altitude"""
    loaded_data= h5py.File(r'C:\Users\ellel\Downloads\Space_Summer\Data-20220720T150923Z-002\Data\TIEGCM\2002_TIEGCM_density.mat')
    localSolarTimes_JB2008 = np.linspace(0,24,24)
    latitudes_JB2008 = np.linspace(-87.5,87.5,20)
    altitudes_JB2008 = np.linspace(100,800,36)
    tiegcm_jb_2008_grid= np.zeros((24,20))
    tiegcm_dens = (10**np.array(loaded_data['density'])*1000).T # convert from g/cm3 to kg/m3
    altitudes_tiegcm = np.array(loaded_data['altitudes']).flatten() #flatten put into a column vector 
    latitudes_tiegcm = np.array(loaded_data['latitudes']).flatten() #can use flatten or squeeze
    localSolarTimes_tiegcm = np.array(loaded_data['localSolarTimes']).flatten()
    nofAlt_tiegcm = altitudes_tiegcm.shape[0]
    nofLst_tiegcm = localSolarTimes_tiegcm.shape[0]
    nofLat_tiegcm = latitudes_tiegcm.shape[0]
    tiegcm_dens_reshaped = np.reshape(tiegcm_dens,(nofLst_tiegcm,nofLat_tiegcm,nofAlt_tiegcm,8760), order='F')
    time_index=31*24
    tiegcm_function=RegularGridInterpolator((localSolarTimes_tiegcm,
                                             latitudes_tiegcm,altitudes_tiegcm),
                                            tiegcm_dens_reshaped[:,:,:,time_index],
                                            bounds_error=False, fill_value=None)
    for lst_i in range(24):
        for lat_i in range(20):
            tiegcm_jb_2008_grid[lst_i,lat_i]= tiegcm_function((localSolarTimes_tiegcm[lst_i],
                                                               latitudes_tiegcm[lat_i],
                                                               alt))
    fig, axs = plt.subplots(1, figsize=(15, 10*2), sharex=True)
    cs = axs.contourf(localSolarTimes_JB2008, latitudes_JB2008, tiegcm_jb_2008_grid.T)
    axs.set_title('TIE-GCM density at 400 km, t = {} hrs'.format(time_index), fontsize=18)
    axs.set_ylabel("Latitudes", fontsize=18)
    axs.tick_params(axis = 'both', which = 'major', labelsize = 16)
    
    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = fig.colorbar(cs)
    cbar.ax.set_ylabel('Density')
    axs.set_xlabel("Local Solar Time", fontsize=18)
    plt.show()
def parse_args():

    # Create an argument parser:
    parser = argparse.ArgumentParser(description = \
                                     'This is a function for plotting at a certain altitude  ')
    
    
    # alt: the altitude
    parser.add_argument('-alt', \
                        help = 'the altitude', \
                        type=int, default= 100)
   # parser.add_argument('-filein', \
                    #  help = 'This is the file going into the program',
                      #type=str)

    parser.add_argument('-fileout', \
                    help = 'This is to export to png',
                    type=str)

    

    # actually parse the data now:
    args = parser.parse_args()
    return args
    
  # main code block
args = parse_args()
print(args)
alt=args.alt
print(alt)
#filein=args.filein
#print(filein)
fileout=args.fileout
print(fileout)
plt.savefig(fileout)
plt.close()
    