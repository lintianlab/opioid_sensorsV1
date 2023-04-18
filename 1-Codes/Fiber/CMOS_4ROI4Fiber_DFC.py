#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 11:21:56 2022

@author: chunyangdong
"""
     
#%% Import libraries
import seaborn as sns
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from photometry_analysis import TianFP_dat,TianFP_dat_test, JRfp_viz, JRfp_dat
%matplotlib inline
import peakutils
import scipy.stats as stats
import matplotlib.transforms as mtrans

#%% Initialize plot and color palette

# Initialize plot style
TianFP_dat.plot_style()

# Initialize color palette
jr_pal = TianFP_dat.set_palette()
TianFP_dat.set_palette(show=True)

#%% Define Experiment File Path

#project_path = '/Users/chunyangdong/Desktop/opioid sensor new data/in vivo/CA3- k1.3 N=4/k1.3 CA3 DFC N=4/'

project_path = '/Volumes/TianLabDrive/1-Projects/7-Opioid sensors/1-in vivo/dNAc - CAG-DIO-k13 fear with 6 sec gap/'
#%% Load data
df_train = TianFP_dat_test.load_CMOS_fear(project_path)
df_train = df_train[['Animal','time','TTL1', '490nm']].copy()

#if any files have TTL input problem and data imports and chooped more than 2200 data points per column
#df_train2 = df_train[['Animal','time', '490nm']].copy()
#%% resample
#df_train = df_train.drop('560nm', 1)
df_train_resamp = TianFP_dat.resample_data(df_train,1)

#if any files have TTL input problem and data imports and chooped more than 2200 data points per column
# df_train_resamp = TianFP_dat.resample_data(df_train2,1)
# df_train_resamp = df_train_resamp.loc[df_train_resamp['time']<2200]
# df_train_resamp.loc[df_train_resamp['490nm'].isnull()]

#%% Calculate biexponential fit for isosbestic signal to correct for bleaching
#df_biexp = TianFP_dat.iso_biexponential(df_train, fit='490nm')
df_biexp = TianFP_dat.iso_biexponential(df_train_resamp, fit='490nm')

#%% Predict 465 signal from biexponential of isosbestic channel
df_fit_biexp= TianFP_dat.fit_biexponential(df_biexp, fit='490nm', yvar='490nm')
#%% Sort tri
df_sorted = TianFP_dat_test.trial_sort(df_fit_biexp)

#%% Calculates 'yvar' normalized and 'yvar' normalized zscore and adds to dataframe
df_normalize = TianFP_dat.trial_normalize(df_sorted, yvar = '490nm_dFF_biexp')

# if want to plot specific time range, use following line. 
#df_trim = df_normalize.loc[(df_normalize['Trial Time']>-10)&(df_normalize['Trial Time']<55)]
#%% Plot dff sessions
#sessions = ['day1_noShock', 'day1_Shock','day2_Shock','day3_Shock', 'day3_noShock']
sessions = ['day1_noShock', '_Shock', 'day3_noShock']

TianFP_dat_test.trial_avg_plot(df_normalize, yvar = '490nm_dFF_zscore_biexp', sessions= sessions, xvar = 'Trial Time', trace='zscore', ylim=(-2,2))

# if want to plot df_trim, use following line.
# sessions = ['day1_noShock', 'day1_Shock','day2_Shock','day3_Shock', 'day3_noShock']
# TianFP_dat.trial_avg_plot(df_trim, yvar = '490nm_dFF_biexp_norm', sessions= sessions, xvar = 'Trial Time', trace='dFF', ylim=(-2.5,2.5))
#%% Heatmap
TianFP_dat.fear_heatmap(df_normalize, '490nm_dFF_biexp_znorm')

#%% AVG dff vs trials
sessions2 = ['day1','day2','day3']
TianFP_dat.trial_avg_epoch_plot(df_normalize, '490nm_dFF_biexp', sessions2)

#%% AUC analysis
def avg_auc_analysis (df,yvar,sessions):
    
    plotLabels = ['trial 1-5','trial 6-10', 'trial 11-15']
    
    for session in sessions:
        df_temp = df.loc[(df['Animal'].str.contains(session.split('_')[0]))]
        trialarea1_5 = []
        trialarea6_10 = []
        trialarea11_15 = []
        for Animal in df_temp['Animal'].unique():
            df_animal = df_temp.loc[df_temp['Animal']==Animal]
            for trial in df_animal['Trial'].unique():
                df_trial = df_animal.loc[df_animal['Trial'] == trial]
                x = df_trial['Trial Time']
                y = df_trial[yvar]

                toneArea = np.trapz(y,[30.5, 33.5])

                trial = int(trial)
                if trial <= 5:
                    trialarea1_5.append(toneArea)
                elif (trial >5) & (trial < 11):
                    trialarea6_10.append(toneArea)
                elif trial>10:
                    trialarea11_15.append(toneArea)
        data = [trialarea1_5, trialarea6_10, trialarea11_15]
#         df_sess = pd.DataFrame(data, index=['Trial 1-5', 'Trial 6-10', 'Trial 11-15'])
#         df_sess.to_csv(f'C:/Users/jacob/Box/Tian Lab Data/Jacob/2022-03 dLight2.1 PFC Fear/downSample/{session} Avg AUC.csv')

        sns.set_style("white")
        fig, ax = plt.subplots(figsize=(8,6))
        color = ['#556B2F', '#BDB76B', '#008080', '#20B2AA', '#800000', '#CD5C5C']
        sns.set_palette(color)
        sns.boxplot(data=data)
        ax.set_xticklabels(plotLabels, size = 20)
        plt.yticks(fontsize= 15)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        ax.yaxis.set_major_locator((plt.LinearLocator(numticks=2)))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        plt.title(f'Average Area Under the Curve: {session}', fontdict= { 'fontsize': 26, 'fontweight':'bold'}, y =1.05)
            
    return data

dat = avg_auc_analysis(df_normalize, '490nm_dFF_biexp_znorm', ['day1','day2','day3'])
areaDict = TianFP_dat.auc_analysis(df_normalize, '490nm_dFF_biexp_norm')
areas = TianFP_dat.auc_plot(areaDict, 'day3',[2,8,15], 'postshock')

#%% Down sample and Save data
df_ = df_normalize[['Animal','Trial','Trial Time', '490nm_dFF_biexp']].copy()
df_m = df_.loc[df_['Animal']=='blue_day1']
trials = df_m['Trial'].tolist()
trialTime = df_m['Trial Time'].tolist()
timeNew = [ '%.2f' % elem for elem in trialTime ]

datOut = {'Trial':trials,'Trial Time':timeNew}
for animal in df_['Animal'].unique():
    datOut.update({animal:df_.loc[df_['Animal']==animal]['490nm_dFF_biexp'].tolist()})
    
df_new = pd.DataFrame(datOut)

df_new.to_csv('/Volumes/TianLabDrive/1-Projects/7-Opioid sensors/1-in vivo/dNAc - CAG-DIO-k13 fear with 6 sec gap/summary.csv')

#%% save by trials
# Change the last variable to the variable you want to export
df_out = df_normalize[['Animal','Trial','Trial Time', '490nm_dFF_biexp']].copy()
# Input any unique mouse session ex. m5_day1
df_m = df_out.loc[(df_out['Animal']=='blue_day1')]
# Gets list of trials and trial times
trials = df_m['Trial'].tolist()
trialTime = df_m.loc[df_m['Trial']=='1']['Trial Time'].tolist()
timeNew = [ '%.2f' % elem for elem in trialTime ]
#Initialize output data frame
datOut = {'Trial Time':timeNew}
# Loop through each day and trial to construct new dataframe
for day in ['day1', 'day2', 'day3']:
    df_day = df_out.loc[df_out['Animal'].str.contains(day)]
    for trial in trials:
        df_trial = df_day.loc[df_day['Trial']==trial]
        for animal in df_trial['Animal'].unique():
            # Change the variable in the final bracket to the variable you want to export
            datOut.update({str(animal)+' Trial: ' + str(trial) + ' 490nm_dFF_biexp':df_trial.loc[df_trial['Animal']==animal]['490nm_dFF_biexp'].tolist()})
df_new = pd.DataFrame(datOut)

df_new.to_csv('/Volumes/TianLabDrive/1-Projects/7-Opioid sensors/1-in vivo/dNAc - CAG-DIO-k13 fear with 6 sec gap/summary_separatetrials.csv')