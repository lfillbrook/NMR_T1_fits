# -*- coding: utf-8 -*-
"""
Module for processing saturation recovery NMR experiments.

Includes the following functions:
    MonoExp - monoexponential fitting equation.
    SR_T1 - Calculates T1 values from saturation recovery experiements, using monoexponential fit.
    MovingSR_T1 - Calculate T1 values (using moving-fit approach) from saturation recovery experiements, using monoexponential fit.

@author: lfill
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def MonoExp(tr, M0, T1):
    '''
    Monoexponential fitting equation.

    Parameters
    ----------
    tr : float
        Recovery time (in seconds).
    M0 : float
        Initial maximum magnetiztion.
    T1 : float
        Longitudinal relaxation constant (in seconds).

    Returns
    -------
    I0 : float
        Intensity of NMR signal.

    '''
    I0 = M0*(1-np.exp(-tr/T1)) 
    return I0

def SR_T1(data, vdlength=8, slicetime=2):
    '''
    Calculates T1 values from repeated saturation recovery experiements, using monoexponential fit.
    
    Parameters
    ----------
    data : DataFrame. 
        1st column = repeated vdlist; subsequent columns = integrals for each peak (column for each peak).
    vdlength : int, optional. 
        Length of the vdlist. The default is 8.
    slicetime : int, optional. 
        Time taken to acquire single spectrum, used to calculate time points. The default is 2.
    
    Returns
    -------
    T1s : DataFrame. 
        Fitted time-resolved T1 values for each peak.
    T1ers : DataFrame. 
        Errors (standard deviation) on T1 values for each peak.
    M0s : DataFrame. 
        Fitted time-resolved M0 values for each peak.
    M0ers : DataFrame. 
        Errors (standard deviation) on M0 values for each peak.
    
    '''
    npoints = int(data.shape[0]/vdlength) # number of points, for unoverlapped sections of vdlength
    tpoints = np.arange(0,npoints)*slicetime # time index for each point
    vds = data.iloc[:,0] # variable delay list
    cols = data.columns[1:] # remaining data (peak integrals)
    # make DataFrame of correct size to collect the fitted data for each point
    T1s = T1ers = M0s = M0ers = pd.DataFrame(index=tpoints, columns=cols)
    # loop over each column/peak
    for col in cols:
        # assign DataFrame to collect fitted data for single peak
        T1 = T1er = M0 = M0er = pd.DataFrame(index=tpoints, columns=[col])
        # loop over number of points
        for i in range(npoints):
            x = vds[i*vdlength:(i+1)*vdlength]
            y = data[col].iloc[i*vdlength:(i+1)*vdlength]
            # fit mono exponential curve to data
            popt, pcov = curve_fit(MonoExp, x, y, p0=(1000,0.01))
            # add fitted values to preprepared DataFrames
            T1.loc[tpoints[i],col] = popt[1] 
            T1er.loc[tpoints[i],col] = np.sqrt(pcov[1,1])
            M0.loc[tpoints[i],col] = popt[0] 
            M0er.loc[tpoints[i],col] = np.sqrt(pcov[0,0])
        T1s[col] = T1
        T1ers[col] = T1er      
        M0s[col] = M0
        M0ers[col] = M0er      
    return T1s, T1ers, M0s, M0ers


def MovingSR_T1(data, vdlength=8, slicetime=2):
    '''
    Calculate T1 values (using moving-fit approach) from saturation recovery experiements, using monoexponential fit.
    
    Parameters
    ----------
    data : DataFrame. 
        1st column = repeated vdlist; subsequent columns = integrals for each peak.
    vdlength : int, optional. 
        Length of the vdlist. The default is 8.
    slicetime : int, optional. 
        Time taken to acquire single spectrum, used to calculate time points. The default is 8.
    
    Returns
    -------
    T1s : DataFrame. 
        Fitted time-resolved T1 values for each peak.
    T1ers : DataFrame. 
        Errors (standard deviation) on T1 values for each peak.
    M0s : DataFrame. 
        Fitted time-resolved M0 values for each peak.
    M0ers : DataFrame. 
        Errors (standard deviation) on M0 values for each peak.
    
    '''
    npoints = int(data.shape[0]-vdlength) # number of points, for moving overlapped sections of vdlength
    tpoints = np.arange(0,npoints)*slicetime/vdlength+slicetime/vdlength # time index for each point
    vds = data.iloc[:,0] # variable delay list
    cols = data.columns[1:] # remaining data (peak integrals)
    # make DataFrame of correct size to collect the fitted data for each point
    T1s = T1ers = M0s = M0ers = pd.DataFrame(index=tpoints, columns=cols)
    # loop over each column/peak
    for col in cols:
        # assign DataFrame to collect fitted data for single peak
        T1 = T1er = M0 = M0er = pd.DataFrame(index=tpoints, columns=[col])
        # loop over number of points
        for i in range(npoints):
            x = vds[i:i+vdlength]
            y = data[col].iloc[i:i+vdlength]
            # fit mono exp curve to data
            popt, pcov = curve_fit(MonoExp, x, y, p0=(1000,0.01))
            # add fitted T1 value to T1 and error DataFrames
            T1.loc[tpoints[i],col] = popt[1] 
            T1er.loc[tpoints[i],col] = np.sqrt(pcov[1,1])
            M0.loc[tpoints[i],col] = popt[0] 
            M0er.loc[tpoints[i],col] = np.sqrt(pcov[0,0])
        T1s[col] = T1
        T1ers[col] = T1er      
        M0s[col] = M0
        M0ers[col] = M0er      
    return T1s, T1ers, M0s, M0ers


#%%
import os
from tkinter import filedialog

# for triplicate 
path = filedialog.askdirectory() # Set path to directory containing all CSV files
name = path # Set name to save figures under
endtime = 250
vdlength = 8
slicetime = 2

processed = ()
for entry in os.scandir(path):
    # Ensure that each of the csv files has the expected number of columns, then process
    inttab = pd.read_csv(entry, nrows=1)
    if len(inttab.columns) == 6:
        data = SR_T1(pd.read_csv(entry), vdlength, slicetime)
        for item in data:
            item.name=entry.name[:-4] #Set the name of each dataset to the name of the corresponding .csv file
        processed += (data,)
    # Alerts you when the number of columns is not as expected so you can fix the csvs before porcessing
    else:
        print('Wrong number of columns:'+entry.name[:-4])

def MovingAverage(data,window,step):
    '''
    Average across time frame

    Parameters
    ----------
    data : pd.DataFrame
        All the NMR peak integrals for a single experiment.
    window : int
        Size of averaging window.
    step : int
        Time frame over which a point will be generated.

    Returns
    -------
    DFout : pd.DataFrame
        Averaged data, including mean and error (standard deviation) as columns with time as the index.

    '''
    centre = window/2
    time,val,err = [],[],[]
    while max(data.index) > centre:
        # Compute values within specified timeframe single column
        vals = data.loc[centre-window/2:centre+window/2].values.flatten()
        if vals.size > 0:
            # Create index with new times
            time.append(centre)
            # Calculate mean across all values in time frame
            val.append(np.nanmean(vals))
            err.append(np.nanstd(vals))
        centre += step
    output = np.vstack([val,err])
    DFout = pd.DataFrame(np.transpose(output),index=time,columns=['Mean','Error'])
    return DFout

labels=[r'$\bf{1}$, 4.04 ppm', r'$\bf{2}$, 4.23 ppm', 'HOD, 4.79 ppm', 'Ascorbate, 4.95 ppm', r'$\bf{3}$, 5.27 ppm']
click_noHs = [2, 1, 2, 2, 2]
# Set up dataframes to contain data for each of the peaks
clickdata = ([],[])
# loop over the peaks to get data for each
for i in range(len(labels)):
    T1concat = M0concat = pd.DataFrame()
    for data in processed:
        T1concat = pd.concat([T1concat, data[0].iloc[:,i]], axis=1)
        M0concat = pd.concat([M0concat, data[1].iloc[:,i]], axis=1)
    # Assign appropriate names to the dataframes (per peak)
    T1concat.name = M0concat.name = labels[i]
    # Ensure that the dataframe index is numeric
    T1concat.index = T1concat.index.tolist()
    M0concat.index = M0concat.index.tolist()
    # Generate average over the datasets (each column in the dataframe of each peak)
    clickdata[0].append(MovingAverage(T1concat,10,2))
    clickdata[1].append(MovingAverage(M0concat,10,2))




































