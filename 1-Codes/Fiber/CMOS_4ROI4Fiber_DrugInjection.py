#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 12:31:10 2022

@author: chunyangdong
"""

#%% Import libraries
import seaborn as sns
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from photometry_analysis import JRfp_dat, tfc_dat, JRfp_viz, TianFP_dat
%matplotlib inline
import peakutils
import scipy.stats as stats

#%% Initialize plot and color palette

# Initialize plot style
JRfp_viz.plot_style()

# Initialize color palette
jr_pal = JRfp_viz.set_palette()
JRfp_viz.set_palette(show=True)

#%%
def load_CMOS_drug(file_dir, TTL_session_ch=1, TTL_on=0.0):
    df_list = []
    session_files = [file for file in sorted(os.listdir(file_dir)) if '.csv' in file]    
    for file in session_files:
        current_file = f'{file_dir}{file}'
        # Import data
        df_temp = import_CMOS_drug(current_file, animal_id=file)
        df_list.append(df_temp)
    df = pd.concat(df_list).reset_index(drop=True)
    
    return df
#%%
def import_CMOS_drug(file_name, animal_id):
    df = pd.read_csv(f'{file_name}', header=1, low_memory=False)
    lowIndex = df[pd.notna(df['CAM 1 EXC 1'])].index.tolist()
    lowIndex.pop(0)
    lowIndex.pop(-1)
    midIndex = [index+34 for index in lowIndex]
    TTLIndex = [point-1 for point in lowIndex] 
    
    times = []
    TTL1 = []
    TTL2 = []
    cam1exc1 = []
    cam2exc3 = []
    cam1exc21 = []
    cam1exc12 = []
    cam1exc2= []
    cam1exc11=[]
    cam2exc31=[]
    cam1exc22=[]
    
    for index in lowIndex:
        times.append(df.at[index,'Time(s)'])
        cam1exc1.append(df.at[index,'CAM 1 EXC 1'])
        cam2exc3.append(df.at[index,'CAM 2 EXC 3'])
        cam1exc21.append(df.at[index,'CAM 1 EXC 2.1'])
        cam1exc12.append(df.at[index,'CAM 1 EXC 1.2'])
#     for index in midIndex:
#         cam1exc2.append(df.at[index,'CAM 1 EXC 2'])
#         cam1exc11.append(df.at[index,'CAM 1 EXC 1.1'])
#         cam2exc31.append(df.at[index,'CAM 2 EXC 3.1'])
#         cam1exc22.append(df.at[index,'CAM 1 EXC 2.2'])
    for index in TTLIndex:
        TTL1.append(df.at[index,'-.5'])
        TTL2.append(df.at[index,'-.6'])
    
    times = [round(value,2) for value in times]

    file_chop = animal_id.split('#')
    session = file_chop[0].split('_')[1]
    mice = file_chop[1].split('_')
    
    df_out = []
    
    for idm, mouse in enumerate(mice):
        if mouse != 'XX':
            if idm == 0:
                lowMouse = cam1exc1
                midMouse = cam1exc2
            elif idm == 1:
                lowMouse = cam2exc3
                midMouse = cam1exc11
            elif idm == 2:
                lowMouse = cam1exc21
                midMouse = cam2exc31
            elif idm == 3:
                lowMouse = cam1exc12
                midMouse = cam1exc22
            
            mouseOut = mouse+'_'+session
            
#             mouseDat = {'Animal':mouseOut,'time':times,'TTL1':TTL1,'TTL2':TTL2,'490nm':lowMouse,'560nm':midMouse}
            mouseDat = {'Animal':mouseOut,'time':times,'TTL1':TTL1,'TTL2':TTL2,'490nm':lowMouse}
            df_mouse = pd.DataFrame(mouseDat)
            df_out.append(df_mouse)
            
            
    df = pd.concat(df_out).reset_index(drop=True)
    return df
#%%
def resample_data(df, freq=10):

    period = 1/freq #might need to use round(, ndigits=3) if getting error with freq
    df_list = []
    for idx in df['Animal'].unique():
        # temporary df for specific subject
        df_subj = df.loc[df['Animal'] == idx, :]
        # convert index to TimeDeltaIndex for resampling
        df_subj.index = df_subj['time']
        df_subj.index = pd.to_timedelta(df_subj.index, unit='S')
        df_subj = df_subj.resample(f'{period}S').mean()
        # interpolate if there are NaNs
        if pd.isnull(df_subj['time']) is True:
            df_subj = df_subj.interpolate()
        df_subj.loc[:, 'Animal'] = idx
        df_subj['time'] = df_subj.index.total_seconds()  
        df_list.append(df_subj)
    
    df = pd.concat(df_list).reset_index(drop=True)
    # resample also moves 'Animal' to end of DataFrame, put it back at front
    cols = df.columns.tolist()
    cols.insert(0, cols.pop(cols.index('Animal')))
    df = df.reindex(columns=cols)
    
    return df
#%% Define project path and get training data

project_path = '/Users/chunyangdong/Desktop/drug actions in claustrum/Isak Rotation CLA/5MeO/iSero/'
#project_path = '/Users/chunyangdong/Desktop/opioid sensor new data/in vivo/CA3- k1.3 N=4/k1.3 CA3 10mgkg U50/'

df_drug = load_CMOS_drug(project_path)

#df_drug_resamp = resample_data(df_drug, freq=.5)

#%% Calculate biexponential fit for isosbestic signal to correct for bleaching
#df_isoBiexp = TianFP_dat.iso_biexponential(df_drug_resamp, fit='490nm')
df_isoBiexp = TianFP_dat.iso_biexponential(df_drug, fit='490nm')
#%% Predict 465 signal from biexponential of isosbestic channel
df_biexp_fit = TianFP_dat.fit_biexponential(df_isoBiexp, fit='490nm', yvar='490nm')

#%% calculate dFF from baseline
df_equal_norm = JRfp_dat.DREADDs_normalize(df_biexp_fit, '490nm', 550)

#%%
def fp_trace(df, yvar='490nm',
             session = 'G6f-5MeO',
             xvar='time',
             trace='raw',
             fig_size=(20, 10),
             xlim=None,
             ylim =None,
             save_fig=False, fig_path=None, **kwargs):
    yvar = yvar  if trace is 'raw' else yvar + f'_{trace}'
    # plot aesthetics variables
    xlab = 'Time (sec)'
    ylab =  r'$\Delta F/F$ (%)'
    tick_size = 18
    label_size = 24
    title_size = 28
    for idx in df['Animal'].unique():
        df_plot = df.loc[df['Animal'] == idx, :]
        
        fig = plt.figure(figsize=fig_size)  # create a figure object
        ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
        # plot aesthetics
        ax.set_ylabel(ylab, size=label_size)
        ax.set_xlabel(xlab, size=label_size)
        ax.margins(x=0)
        if trace is 'raw':
            title = f'{session}: {idx}: {yvar}'
            ax.set_title(title, size=title_size)
            X = df_plot.loc[:, xvar]
            Y = df_plot.loc[:, yvar]
            ax.plot(X, Y, color=jr_pal[4], label=yvar)
            
        elif trace in ['dFF', 'dFF_zscore', 'dff']:
            title = f'{session}: {idx}: {yvar}'
            ax.set_title(title, size=title_size)
            X = df_plot.loc[:, xvar]
            if pltBiexp is True:
                Y = df_plot.loc[:, yvar + '_biexp']
            else:
                Y = df_plot.loc[:, yvar]
            ax.axhline(y=0, linestyle='-', color='black') # Add horizontal line
            ax = plt.plot(X, Y, color=jr_pal[4])
        plt.tick_params(axis='both', labelsize=tick_size, width=2, length=6)
        if xlim:
            plt.xlim(xlim)
        if xlim:
            plt.xlim(xlim)
        # # save figure
        if save_fig is True:
            fig_path = kwargs.pop('fig_path', '~/Desktop/') if fig_path is None else fig_path
            file_name = f'{session} - {idx}{trace} traces'
            fig.savefig(f'{fig_path}{file_name}.png', bbox_inches='tight')

#%% Plot
#df_plot = df_equal_norm
df_plot = df_biexp_fit
fp_trace(df_plot, trace='raw', yvar='490nm_norm', session = 'G6f-5MeO', pltBiexp = False, pltIso=True,
                 save_fig=False)
#%% Save data
df_ = df_equal_norm[['Animal','time', '490nm_norm']].copy()
df_m = df_.loc[df_['Animal']=='iSeroR1_5MeO']
Time = df_m['time'].tolist()
timeNew = [ '%.2f' % elem for elem in Time ]

datOut = {'time':timeNew}
for animal in df_['Animal'].unique():
    datOut.update({animal:df_.loc[df_['Animal']==animal]['490nm_norm'].tolist()})
    
df_new = pd.DataFrame(datOut)


df_new.to_csv('/Users/chunyangdong/Desktop/drug actions in claustrum/Isak Rotation CLA/5MeO/Summary/iSero.csv')
#df_new.to_csv('/Users/chunyangdong/Desktop/Data_analysis/Fiber/Python/Spyder/Drug.csv')













