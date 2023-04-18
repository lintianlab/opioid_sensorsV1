
"""Jupyter keyboard shortcut:
Shift + enter - run and move to the next cell
Control + enter - run the current cell. 

Run, run all above selected cells. 

esc esc A - insert cell above current cell
esc esc B - insert cell above current cell

esc esc D D - delet current cell

new cell, ##xxx, make title. 
"""

""" Load photometry data.
General workflow: Wrap these steps around load_session_data()
    1. load_session_data()
    2. TTL_session() Only include data from behavior session
    3. resample_data()
    4. fit_linear() 
"""
import os
import scipy.signal as signal
import scipy.stats as stats
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import yaml
import statsmodels.api as sm
import matplotlib.pyplot as plt
import peakutils
import seaborn as sns
import matplotlib.transforms as mtrans

################################################################################
################################################################################

# define color palette:
jr_pal = ['#33AE81', #light green
          '#EC5656', #orange
          '#9147B1', #purple
          '#FFAC00', #yellow
          '#3049AD', #blue
          '#224624', #black
          '#E03FD8', #pink
          '#bdbdbd'] #gray


def set_palette(color_pal=None, show=False):
    """Set default color palette."""
    color_pal = jr_pal if color_pal is None else color_pal
    sns.set_palette(color_pal)
    if show:
        sns.palplot(color_pal)
    else:
        return color_pal
    
################################################################################

def plot_style(figure_size=None):
    """Set default plot style."""
    figure_size = [30, 20] if figure_size is None else figure_size
    size_scalar = (sum(figure_size)/2)/25
    # figure and axes info
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rc('axes', facecolor='white', linewidth=2*size_scalar,
           labelsize=40*size_scalar, titlesize=32*size_scalar,
           labelpad=5*size_scalar)

    plt.rc('axes.spines', right=False, top=False)
    # plot-specific info
    plt.rcParams['lines.linewidth'] = 2*size_scalar
    # tick info
    plt.rcParams['xtick.labelsize'] = 32*size_scalar
    plt.rcParams['ytick.labelsize'] = 30*size_scalar
    plt.rcParams['xtick.major.size'] = 10*size_scalar
    plt.rcParams['ytick.major.size'] = 10*size_scalar
    plt.rcParams['xtick.major.width'] = 2*size_scalar
    plt.rcParams['ytick.major.width'] = 2*size_scalar
    # legend info
    plt.rc('legend', fontsize=32*size_scalar, frameon=False)
    
################################################################################

def import_Tian_fear_data(file_name, yvar='465nm', yvar_out=2,
                input_ch=3, animal_id='file_name'):
    """
    Import photometry data from Doric equipment.

    Parameters
    ----------
    file_name : str
        Name of file to import.
    yvar : str, optional
        Specify type of photometry data to load. 
        Must be '465nm' or '560nm', by default '465nm'
    yvar_out: int, optional
        Specify the output ch (1-4) used for yvar LED excitation, by default 2.
    input_ch : int, optional
        Input channel from Doric photometry rig, by default 1
    animal_id : str, optional
        Specify animal ID, by default 'file_name'

    Returns
    -------
    DataFrame
        DataFrame containing data imported from file.

    Raises
    ------
    ValueError
        Raises error if yvar is not '465nm' or '560nm'.
    """
    # raise error for incorrect yvar input
    if yvar not in ['465nm', '560nm']:
        raise ValueError("'yvar' must be: '465nm' or '560nm'")
        
    df = pd.read_csv(f'{file_name}', low_memory=False)
    df['Animal'] = animal_id
    ref_out = 1
    df = (df.rename(columns={'Digital I/O | Ch.1':'TTL1',
                             'Digital I/O | Ch.2':'TTL2',
                             'Digital I/O | Ch.3':'TTL3',
                             'Digital I/O | Ch.4':'TTL4', 
                             '---':'time',
                             f'Analog In. | Ch.{input_ch}.{yvar_out}':f'raw_{input_ch}', 
                             f'Analog In. | Ch.{input_ch}':'405nm',
                             f'Analog In. | Ch.{input_ch}.{ref_out}':f'{yvar}',
                             f'Analog Out. | Ch.{ref_out}':'405nm_LED', 
                             f'Analog Out. | Ch.{yvar_out}':f'{yvar}_LED'})
            .reindex(columns=['Animal', 'time', 'TTL1', '405nm', yvar]))
    df = df.iloc[1: , :]
    df['time']=df['time'].astype(float)
    df['TTL1']=df['TTL1'].astype(float)
    df['405nm']=df['405nm'].astype(float)
    df[yvar]=df[yvar].astype(float)
    return df

################################################################################

def import_Jason_data(file_name, yvar='465nm', yvar_out=2,
                input_ch=3, animal_id='file_name'):
    """
    Import photometry data from Doric equipment.

    Parameters
    ----------
    file_name : str
        Name of file to import.
    yvar : str, optional
        Specify type of photometry data to load. 
        Must be '465nm' or '560nm', by default '465nm'
    yvar_out: int, optional
        Specify the output ch (1-4) used for yvar LED excitation, by default 2.
    input_ch : int, optional
        Input channel from Doric photometry rig, by default 1
    animal_id : str, optional
        Specify animal ID, by default 'file_name'

    Returns
    -------
    DataFrame
        DataFrame containing data imported from file.

    Raises
    ------
    ValueError
        Raises error if yvar is not '465nm' or '560nm'.
    """
    # raise error for incorrect yvar input
    if yvar not in ['465nm', '560nm']:
        raise ValueError("'yvar' must be: '465nm' or '560nm'")

    df = pd.read_csv(f'{file_name}')
    df['Animal'] = animal_id
    ref_out = 1
    df = (df.rename(columns={'DI/O-1':'TTL1', 
                             'Time(s)':'time',
                             'AIn-3 - Dem (AOut-1)':'sensor_405nm', 
                             'AIn-3 - Dem (AOut-2)':'sensor_465nm',
                             'AIn-1 - Dem (AOut-1)':'control_405nm',
                             'AIn-2 - Dem (AOut-2)':'control_465nm'})
            .reindex(columns=['Animal', 'time', 'TTL1', 'sensor_405nm', 'sensor_465nm', 'control_405nm', 'control_465nm']))
    return df

################################################################################

def load_Tian_fear_session(file_dir='fp_file_path', yvar='465nm', yvar_out=2,
                      input_ch=3, TTL_session_ch=1, TTL_on=0, Jason = False):
    """
    Load photometry session data from a directory.

    Parameters
    ----------
    file_dir : str, optional
        Path to directory of photometry data files, by default 'fp_file_path'
    yvar : str, optional
        Specify type of photometry data to load. 
        Must be '465nm' or '560nm', by default '465nm'
    yvar_out : int, optional
        Specify the output ch (1-4) used for yvar LED excitation, by default 2.
    input_ch : int, optional
        Input channel from Doric photometry rig, by default 1.
    TTL_session_ch : int, optional
        Specify which TTL input channel signals session start and end, by default 1.
    TTL_on : int, optional
        Specify TTL value when SG-231 is ON, by default 0

    Returns
    -------
    DataFrame object containing data imported from .csv files in the provided directory.
    """
    df_list = []
    session_files = [file for file in sorted(os.listdir(file_dir)) if '.csv' in file]    
    for file in session_files:
        current_file = f'{file_dir}{file}'
        # Import data
        if Jason == True:
            df_temp = import_Jason_data(current_file, yvar=yvar, yvar_out=yvar_out, 
                              input_ch=input_ch, animal_id=file)
        else:
            df_temp = import_Tian_fear_data(current_file, yvar=yvar, yvar_out=yvar_out, 
                              input_ch=input_ch, animal_id=file)
        # Only include data from the time that MedAssociates SG-231 TTL Generator is on
        df_TTL = TTL_session(df_temp, TTL_session_ch, TTL_on)
        df_list.append(df_TTL)
    df = pd.concat(df_list).reset_index(drop=True)
#     df = df[df['time'] < int(max(df['time']))].reset_index(drop=True)
    
    return df

################################################################################

def import_Tian_reward_data(file_name, yvar='465nm', yvar_out=2,
                input_ch=3, animal_id='file_name'):
    """
    Import photometry data from Doric equipment.

    Parameters
    ----------
    file_name : str
        Name of file to import.
    yvar : str, optional
        Specify type of photometry data to load. 
        Must be '465nm' or '560nm', by default '465nm'
    yvar_out: int, optional
        Specify the output ch (1-4) used for yvar LED excitation, by default 2.
    input_ch : int, optional
        Input channel from Doric photometry rig, by default 1
    animal_id : str, optional
        Specify animal ID, by default 'file_name'

    Returns
    -------
    DataFrame
        DataFrame containing data imported from file.

    Raises
    ------
    ValueError
        Raises error if yvar is not '465nm' or '560nm'.
    """
    # raise error for incorrect yvar input
    if yvar not in ['465nm', '560nm']:
        raise ValueError("'yvar' must be: '465nm' or '560nm'")
        
    df = pd.read_csv(f'{file_name}', low_memory=False)
    df['Animal'] = animal_id
    ref_out = 1
    df = (df.rename(columns={'Digital I/O | Ch.1':'TTL1',
                             'Digital I/O | Ch.2':'TTL2',
                             'Digital I/O | Ch.3':'TTL3',
                             'Digital I/O | Ch.4':'TTL4', 
                             '---':'time',
                             f'Analog In. | Ch.{input_ch}.{yvar_out}':f'raw_{input_ch}', 
                             f'Analog In. | Ch.{input_ch}':'405nm',
                             f'Analog In. | Ch.{input_ch}.{ref_out}':f'{yvar}',
                             f'Analog Out. | Ch.{ref_out}':'405nm_LED', 
                             f'Analog Out. | Ch.{yvar_out}':f'{yvar}_LED'})
            .reindex(columns=['Animal', 'time', 'TTL2', 'TTL3', 'TTL4', '405nm', yvar]))
    df = df.iloc[1: , :]
    df['time']=df['time'].astype(float)
    df['TTL2']=df['TTL2'].astype(float)
    df['TTL3']=df['TTL3'].astype(float)
    df['TTL4']=df['TTL4'].astype(float)
    df['405nm']=df['405nm'].astype(float)
    df[yvar]=df[yvar].astype(float)
    return df

################################################################################

def load_Tian_reward_data(file_dir='fp_file_path', yvar='465nm', yvar_out=2,
                      input_ch=3, TTL_session_ch=1, TTL_on=0):
    """
    Load photometry session data from a directory.

    Parameters
    ----------
    file_dir : str, optional
        Path to directory of photometry data files, by default 'fp_file_path'
    yvar : str, optional
        Specify type of photometry data to load. 
        Must be '465nm' or '560nm', by default '465nm'
    yvar_out : int, optional
        Specify the output ch (1-4) used for yvar LED excitation, by default 2.
    input_ch : int, optional
        Input channel from Doric photometry rig, by default 1.
    TTL_session_ch : int, optional
        Specify which TTL input channel signals session start and end, by default 1.
    TTL_on : int, optional
        Specify TTL value when SG-231 is ON, by default 0

    Returns
    -------
    DataFrame object containing data imported from .csv files in the provided directory.
    """
    df_list = []
    session_files = [file for file in sorted(os.listdir(file_dir)) if '.csv' in file] 
    for file in session_files:
        current_file = f'{file_dir}{file}'
        # Import data
        df_temp = import_Tian_reward_data(current_file, yvar=yvar, yvar_out=yvar_out, 
                              input_ch=input_ch, animal_id=file)
        # Only include data from the time that MedAssociates SG-231 TTL Generator is on
        df_list.append(df_temp)
    df = pd.concat(df_list).reset_index(drop=True)
    
    return df

################################################################################

def TTL_session(df, TTL_session_ch=1, TTL_on=0):
    """
    Find first and last TTL input (to indicate start and end of FC session).
    - In the Doric recording TTL value is 1.
    - When Med-Assocaites SG-231 is ON, TTL value set to 0.

    Parameters
    ----------
    df : DataFrame
        Data to find start and end TTL pulses for.
    TTL_on : int, optional
        TTL value when SG-231 is ON, by default 0

    Returns
    -------
    DataFrame
        Data clipped to first and last TTL transition.
    """
    ttl_ch = 'TTL' + str(TTL_session_ch)
    first_row = min(df[df[ttl_ch] == TTL_on].index.tolist())
    last_row = max(df[df[ttl_ch] == TTL_on].index.tolist())
    df_new = df[(df.index >= first_row) & (df.index <= last_row)]
    df_new = df_new.reset_index(drop=True)
    # reset 'Time' and 'sec' to start at zero
    df_new['time'] = df_new['time'] - df_new['time'][0]
    
    return df_new

################################################################################

def resample_data(df, freq=10):
    """
    Resamples data to a specified frequency. Converts index to Timedelta,
    and uses .resample() to resample data.

    Parameters
    ----------
    df : DataFrame
        DataFrame object containing data from load_session_data()
    freq : int
        Value of frequency to resample data, by default 10.
        
    Returns
    -------
    Resampled DataFrame
    """
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
    # for some reason this function converts TTL cols to float64
    ttl_cols = df.columns.str.contains('TTL')
    df.loc[:, ttl_cols] = df.loc[:, ttl_cols].astype(int)
    # resample also moves 'Animal' to end of DataFrame, put it back at front
    cols = df.columns.tolist()
    cols.insert(0, cols.pop(cols.index('Animal')))
    df = df.reindex(columns=cols)
    
    return df

################################################################################

def shift_data(df, yvar='465nm', control='405nm'):

    for idx in df['Animal'].unique():
        # define vars for biexponential model:
        X = df.loc[df['Animal'] == idx, 'time']
        Y = df.loc[df['Animal'] == idx, f'{yvar}']
        Ycontrol = df.loc[df['Animal']==idx, f'{control}']
        meanY = np.mean(Y)
        meanYcontrol = np.mean(Ycontrol)
        Yshift = 1-meanY
        YcontrolShift = 0.5-meanYcontrol
        df.loc[df['Animal']==idx,f'{yvar}_shift']=Y+Yshift
        df.loc[df['Animal'] == idx, f'{control}_shift'] = Ycontrol+YcontrolShift
        
    return df

################################################################################

def dSig_ref(df, Ysig='465nm', Yref='405nm'):
    """
    Linear regression motion & debleaching correction:
    - Use OLS regression to model fluorescence signal as a function of the isosbestic signal.
    - Use Ypred values to calculate %dFF as: 100*(Y-Ypred)/Ypred

    Parameters
    ----------
    df : DataFrame
        Data to apply linear fit on.
    Y_sig : str
        Column of fluorescence values to predict.
    Y_ref : str
        Column of isosbestic values used to predict fluorescence.
    
    Returns
    -------
    DataFrame with %dFF calculated as: 100*(Y-Ypred)/Ypred
    """

    for idx in df['Animal'].unique():
        # define vars for linear model:
        X = df.loc[df['Animal'] == idx, Yref]
        Y = df.loc[df['Animal'] == idx, Ysig]
        dSig = Y/X
        df.loc[df['Animal'] == idx, f'{Ysig}/{Yref}'] = dSig
    
    return df

################################################################################

def fit_linear(df, Y_sig='465nm', Y_ref='405nm'):
    """
    Linear regression motion & debleaching correction:
    - Use OLS regression to model fluorescence signal as a function of the isosbestic signal.
    - Use Ypred values to calculate %dFF as: 100*(Y-Ypred)/Ypred

    Parameters
    ----------
    df : DataFrame
        Data to apply linear fit on.
    Y_sig : str
        Column of fluorescence values to predict.
    Y_ref : str
        Column of isosbestic values used to predict fluorescence.
    
    Returns
    -------
    DataFrame with %dFF calculated as: 100*(Y-Ypred)/Ypred
    """

    for idx in df['Animal'].unique():
        # define vars for linear model:
        X = df.loc[df['Animal'] == idx, Y_ref]
        Y = df.loc[df['Animal'] == idx, Y_sig]
        mod = sm.OLS(Y, X).fit()
        Ypred = mod.predict(X)
        dFF = (Y-Ypred)/Ypred*100
        df.loc[df['Animal'] == idx, f'{Y_sig}_pred'] = Ypred
        df.loc[df['Animal'] == idx, f'{Y_sig}_dFF'] = dFF
        df.loc[df['Animal'] == idx, f'{Y_sig}_dFF_zscore'] = stats.zscore(dFF, ddof=1)
    
    return df

################################################################################

def iso_biexponential(df, fit='405nm'):
    """
    Fits isosbestic channel with biexponential to correct bleaching in signal channel:
    - Use least square regression to fit the isosbestic signal.

    Parameters
    ----------
    df : DataFrame
        Data to apply linear fit on.
    
    Returns
    -------
    
    """
    
    for idx in df['Animal'].unique():
        # define vars for biexponential model:
        X = df.loc[df['Animal'] == idx, 'time']
        Y = df.loc[df['Animal'] == idx, fit]
        popt, pcov = curve_fit(biexponential,X,Y,p0=(0.2,0,0.2,0),maxfev=10000)
        isoBiexp = biexponential(X,*popt)
        df.loc[df['Animal'] == idx, f'{fit}_biexp'] = isoBiexp
    
    return df
    
################################################################################    

def biexponential(x, a, b, c, d):
    return a * np.exp(b * x) + c * np.exp(d * x)

################################################################################  

def fit_biexponential(df, fit='405nm', yvar='465nm'):
    
    for idx in df['Animal'].unique():
        X = df.loc[df['Animal'] == idx, 'time']
        Y = df.loc[df['Animal'] == idx, yvar]
        isoFitY = df.loc[df['Animal'] == idx, f'{fit}_biexp']
        isoBiexp = np.vstack((X,isoFitY)).T
        model = LinearRegression().fit(isoBiexp, Y)
        Ypred = model.predict(isoBiexp)
        df.loc[df['Animal'] == idx, f'{yvar}_biexp'] = Ypred
        dFFBiexp = 100*((Y-Ypred)/Ypred)
        df.loc[df['Animal'] == idx, f'{yvar}_dFF_biexp'] = dFFBiexp
        df.loc[df['Animal'] == idx, f'{yvar}_dFF_zscore_biexp'] = stats.zscore(dFFBiexp, ddof=1)

    return df

################################################################################

def subtraction(df, Y_sig='465nm', Y_ref='405nm'):
    """
    Linear regression motion & debleaching correction:
    - Use OLS regression to model fluorescence signal as a function of the isosbestic signal.
    - Use Ypred values to calculate %dFF as: 100*(Y-Ypred)/Ypred

    Parameters
    ----------
    df : DataFrame
        Data to apply linear fit on.
    Y_sig : str
        Column of fluorescence values to predict.
    Y_ref : str
        Column of isosbestic values used to predict fluorescence.
    
    """

    for idx in df['Animal'].unique():
        # define vars for linear model:
        X = df.loc[df['Animal'] == idx, Y_ref]
        Y = df.loc[df['Animal'] == idx, Y_sig]
        dSig = Y-X
        df.loc[df['Animal'] == idx, f'{Y_sig}-{Y_ref}'] = dSig
    
    return df

################################################################################

def trial_sort(df):
    """
    Linear regression motion & debleaching correction:
    - Use OLS regression to model fluorescence signal as a function of the isosbestic signal.
    - Use Ypred values to calculate %dFF as: 100*(Y-Ypred)/Ypred

    Parameters
    ----------
    df : DataFrame
        Data to apply linear fit on.
    
    """

    sessionStart = 3000
    trialStart = [sessionStart + i*1200 for i in range(15)]
    dfs=[]
    for idx in df['Animal'].unique():
        df_list=[]
        df_animal = df.loc[df['Animal'] == idx, :].reset_index(drop=True)
        for trial, trialTime in enumerate(trialStart):
            df_temp = df_animal.iloc[(trialTime-300):(trialTime+900)]
            df_temp.loc[:, 'Trial Time'] = np.linspace(-30,90,len(df_temp))
            df_temp.loc[:,'Trial'] = f'{trial+1}'
            df_list.append(df_temp)
        df_cat=pd.concat(df_list)
        dfs.append(df_cat)
    df=pd.concat(dfs).reset_index(drop=True)
    day1shock = [*range(6,16,1)]
    day1noshock = [*range(1,6,1)]
    day3shock = [*range(1,6,1)]
    day3noshock = [*range(6,16,1)]
    df.loc[(df['Animal'].str.contains('day1'))&(df['Trial'].astype(int).isin(day1shock)), 'Shock']= 'Shock'
    df.loc[(df['Animal'].str.contains('day1'))&(df['Trial'].astype(int).isin(day1noshock)), 'Shock']= 'noShock'
    df.loc[(df['Animal'].str.contains('day2')),'Shock']= 'Shock'
    df.loc[(df['Animal'].str.contains('day3'))&(df['Trial'].astype(int).isin(day3shock)), 'Shock']= 'Shock'
    df.loc[(df['Animal'].str.contains('day3'))&(df['Trial'].astype(int).isin(day3noshock)), 'Shock']= 'noShock'
    return df

################################################################################

def trial_normalize(df, yvar):
    """
    Linear regression motion & debleaching correction:
    - Use OLS regression to model fluorescence signal as a function of the isosbestic signal.
    - Use Ypred values to calculate %dFF as: 100*(Y-Ypred)/Ypred

    Parameters
    ----------
    df : DataFrame
        Data to apply linear fit on.
    
    """
    subj_data = []
    subj_zdata = []
    for idx in df['Animal'].unique():
        df_temp = df.loc[df['Animal'] == idx, :]
        znorm_vals = []
        norm_vals = []
        for i in df_temp['Trial'].unique():
            df_trial_baseline = df_temp.loc[(df_temp['Trial'] == i) & 
                                            (df_temp['Trial Time'] < 0), :]
            pre_CS_mean = np.mean(df_trial_baseline[yvar])
            pre_CS_std = np.std(df_trial_baseline[yvar])
            norm_vals.append((df_temp.loc[df_temp['Trial'] == i, yvar] - pre_CS_mean)/pre_CS_std)
            znorm_vals.append(stats.zscore(df_temp.loc[(df_temp['Trial'] == i), yvar], ddof=1))
            # flatten normalized values from each trial
            norm_vals_flat = [item for sublist in norm_vals for item in sublist]
            znorm_vals_flat = [item for sublist in znorm_vals for item in sublist]
        subj_data.append(norm_vals_flat)
        subj_zdata.append(znorm_vals_flat)
    # flatten normalized values from each subject
    df[f'{yvar}_norm'] = [item for sublist in subj_data for item in sublist]
    df[f'{yvar}_znorm'] = [item for sublist in subj_zdata for item in sublist]
    return df

################################################################################

def trial_avg_plot(df, yvar, sessions, xvar ='Trial Time', trace='dFF', xlim=None, ylim=None):
    label_size = 24
    title_size = 28
    xlab = 'Time From Tone (second)'
    ylab = r'$\Delta F/F$ (%)' if trace is 'dFF' else r'$\Delta R/R$ (%)'
    color = 0 if trace is 'dFF' else 4
    tick_size = 18
    label_size = 24
    title_size = 28

    for session in sessions:
        df_temp = df.loc[(df['Animal'].str.contains(session.split('_')[0])) & (df['Shock'] == session.split('_')[1])]
        df_temp.name = session
        animal_means = df_temp.groupby([xvar]).mean().reset_index()
        animal_stds = (df_temp.groupby([xvar, 'Animal']).mean()
                       .groupby([xvar]).std().reset_index())
        # grab variables for plotting
        x = animal_means.loc[:, xvar]
        y = animal_means.loc[:, yvar]
        yerror = animal_stds.loc[:, yvar]
        fig = plt.figure(figsize=(16,10))  # create a figure object
        ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
        # plot aesthetics
        ax.set_ylabel(ylab, size=label_size)
        ax.set_xlabel(xlab, size=label_size)
        ax.margins(x=0)
        title = f'{session}: {trace}'
        ax.set_title(title, size=title_size)
        # plot the data
        line = ax.plot(x, y, color=jr_pal[color])
        ax.axvspan(0, 30, facecolor=jr_pal[7], alpha=0.2)
        ax.axvspan(27.5, 29, facecolor='black', alpha=0.2)
        ax.axhline(y=0, linestyle='--', color=jr_pal[5], linewidth=0.6)
        ax.fill_between(x, y-yerror, y+yerror, facecolor=line[0].get_color(), alpha=0.15)
        
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)

################################################################################

def compare_session_plot(dfs, yvar, xvar ='Trial Time', trace='dFF', xlim=None, ylim=None):
    label_size = 24
    title_size = 28
    xlab = 'Time From Tone (second)'
    ylab = r'$\Delta F/F$ (%)' if trace is 'dFF' else r'$\Delta R/R$ (%)'
    tick_size = 18
    label_size = 24
    title_size = 28
    
    fig = plt.figure(figsize=(16,10))  # create a figure object
    ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
    # plot aesthetics
    ax.set_ylabel(ylab, size=label_size)
    ax.set_xlabel(xlab, size=label_size)
    ax.margins(x=0)
    title = f'High vs Low Shock: {trace}'
    ax.set_title(title, size=title_size)

    for df in dfs:
        df_temp = df.loc[(df['Animal'].str.contains(df.name.split('_')[0])) & (df['Shock'] == df.name.split('_')[1])]
        df_temp.name = session
        animal_means = df_temp.groupby([xvar]).mean().reset_index()
        animal_stds = (df_temp.groupby([xvar, 'Animal']).mean()
                       .groupby([xvar]).std().reset_index())
        # grab variables for plotting
        x = animal_means.loc[:, xvar]
        y = animal_means.loc[:, yvar]
        yerror = animal_stds.loc[:, yvar]
        
        # plot the data
        line = ax.plot(x, y, label=df.name)
        ax.axvspan(0, 30, facecolor=jr_pal[7], alpha=0.2)
        ax.axvspan(27.5, 29, facecolor='black', alpha=0.2)
        ax.axhline(y=0, linestyle='--', color=jr_pal[5], linewidth=0.6)
        ax.fill_between(x, y-yerror, y+yerror, alpha=0.15)
        
#         if xlim:
#             plt.xlim(xlim)
#         if ylim:
#             plt.ylim(ylim)
        
################################################################################

def session_plot(df, yvar='465nm',
                 xvar ='time',trace='raw', pltBiexp=False, 
                 pltIso=True, yiso='405nm',xlim=None, 
                 ylim=None,fig_size=(12,8)):
    
    xlab = 'Time (sec)'
    ylab = 'Fluorescence (au)' if trace is 'raw' else r'$\Delta F/F$ (%)'
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
            title = f'{idx}: {yiso} (purple) + {yvar}'
            ax.set_title(title, size=title_size)
            X = df_plot.loc[:, xvar]
            Y = df_plot.loc[:, yvar]
            Yiso = df_plot.loc[:, yiso]
            ax.plot(X, Y, color=jr_pal[4], label=yvar)
            if pltIso is True:
                ax.plot(X, Yiso, color=jr_pal[3], label='isosbestic')
            ax.legend(fontsize=16, loc='upper right', bbox_to_anchor=(1, 1.05))

        elif trace in ['dFF', 'dFF_zscore', 'dff']:
            title = f'{session}: {idx}: {yvar}'
            ax.set_title(title, size=title_size)
            X = df_plot.loc[:, xvar]
            Y = df_plot.loc[:, yvar]

            ax.axhline(y=0, linestyle='-', color='black') # Add horizontal line
            ax = plt.plot(X, Y, color=jr_pal[4])

    plt.tick_params(axis='both', labelsize=tick_size, width=2, length=6)

    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
        
################################################################################

def categorize_data(df):
    pd.options.mode.chained_assignment = None
    df_list = []
    for idx in df['Animal'].unique():
        rewardedMags=[]
        unrewardedMags=[]
        toremove=[]
        # subset individual animal
        df_animal = df.loc[df['Animal'] == idx, :]
        # find indices of rewarded lever presses and filter to only select single timepoint
        rewards = df_animal.loc[(df_animal['TTL4'] == 0)].index.tolist()
        index=1
        while index < len(rewards):
            if rewards[index] - rewards[index - 1] == 1:
                del rewards[index]
            else:
                index += 1  
        # find indices of magazine entires and filter by rewarded vs unrewarded, and only select single timepoint
        for reward in rewards:
            if len(df_animal.loc[(df_animal.index>reward) & (df_animal['TTL2'] == 0),:])>0:
                rewardedMags.append(df_animal.loc[(df_animal.index>reward) & (df_animal['TTL2'] == 0),:].iloc[0].name)
                unrewardedMags = df_animal.loc[(df_animal.index>reward)&(df_animal['TTL2'] == 0),:].iloc[1:].index.tolist()
        index=1
        while index < len(unrewardedMags):
            if unrewardedMags[index] - unrewardedMags[index - 1] == 1:
                del unrewardedMags[index]
            else:
                index += 1
        # find indices of unrewarded lever presses and filter to only select single timepoint        
        levers = df_animal.loc[(df_animal['TTL3'] == 0)].index.tolist()
        index=1
        while index < len(levers):
            if levers[index] - levers[index - 1] == 1:
                del levers[index]
            else:
                index += 1
        for lever in levers:
            if lever in rewards:
                toremove.append(lever)
        for remove in toremove:
            levers.remove(remove)

        for reward in rewards:
            df_animal.at[reward,'Category'] = 'Reward'
        for mag in rewardedMags:
            df_animal.at[mag,'Category'] = 'Rewarded Mag'
        for unmag in unrewardedMags:
            df_animal.at[unmag,'Category'] = 'Unrewarded Mag'
        for lever in levers:
            df_animal.at[lever,'Category'] = 'Lever'


        df_list.append(df_animal)

    df = pd.concat(df_list).reset_index(drop=True)
    return df

################################################################################

def normalize_reward_data(df, yvar='465nm'):
    dfs=[]
    for idx in df['Animal'].unique():
        df_cat = []
        rewards=[]
        levers=[]
        rewardedMags=[]
        unrewardedMags=[]
        # subset individual animal
        df_animal = df.loc[df['Animal'] == idx, :].reset_index(drop=True)
        # find indices of rewarded lever presses and filter to only select single timepoint
        rewards = df_animal.loc[(df_animal['Category'] == 'Reward')].index.tolist()
        levers = df_animal.loc[(df_animal['Category'] == 'Lever')].index.tolist()
        rewardedMags = df_animal.loc[(df_animal['Category'] == 'Rewarded Mag')].index.tolist()
        unrewardedMags = df_animal.loc[(df_animal['Category'] == 'Unrewarded Mag')].index.tolist()
        for reward in rewards:
            df_temp = df_animal.iloc[(reward-30):(reward+51)]
            df_temp.loc[:,'Reward Time'] = np.linspace(-3,5,len(df_temp))
            df_baseline = df_temp.loc[df_temp['Reward Time']<-1]
            baseline_mean = np.mean(df_baseline[yvar])
            baseline_std = np.std(df_baseline[yvar])
            df_temp.loc[:,f'{yvar}_norm'] = (df_temp.loc[:,yvar] - baseline_mean)/baseline_std
            df_temp.loc[:,f'{yvar}_znorm'] =  stats.zscore(df_temp.loc[:,f'{yvar}_norm'], ddof=1)
            df_cat.append(df_temp)
        for lever in levers:
            df_temp = df_animal.iloc[(lever-30):(lever+51)]
            df_temp.loc[:,'Lever Time'] = np.linspace(-3,5,len(df_temp))
            df_baseline = df_temp.loc[df_temp['Lever Time']<-1]
            baseline_mean = np.mean(df_baseline[yvar])
            baseline_std = np.std(df_baseline[yvar])
            df_temp.loc[:,f'{yvar}_norm'] = (df_temp.loc[:,yvar] - baseline_mean)/baseline_std
            df_temp.loc[:,f'{yvar}_znorm'] =  stats.zscore(df_temp.loc[:,f'{yvar}_norm'], ddof=1)
            df_cat.append(df_temp)
        for rewardedMag in rewardedMags:
            df_temp = df_animal.iloc[(rewardedMag-30):(rewardedMag+51)]
            if len(df_temp)==81:
                df_temp.loc[:,'Rewarded Mag Time'] = np.linspace(-3,5,len(df_temp))
                df_baseline = df_temp.loc[df_temp['Rewarded Mag Time']<-1]
                baseline_mean = np.mean(df_baseline[yvar])
                baseline_std = np.std(df_baseline[yvar])
                df_temp.loc[:,f'{yvar}_norm'] = (df_temp.loc[:,yvar] - baseline_mean)/baseline_std
                df_temp.loc[:,f'{yvar}_znorm'] =  stats.zscore(df_temp.loc[:,f'{yvar}_norm'], ddof=1)
                df_cat.append(df_temp)
        for unrewardedMag in unrewardedMags:
            df_temp = df_animal.iloc[(unrewardedMag-30):(unrewardedMag+51)]
            if len(df_temp)==81:
                df_temp.loc[:,'Unrewarded Mag Time'] = np.linspace(-3,5,len(df_temp))
                df_baseline = df_temp.loc[df_temp['Unrewarded Mag Time']<-1]
                baseline_mean = np.mean(df_baseline[yvar])
                baseline_std = np.std(df_baseline[yvar])
                df_temp.loc[:,f'{yvar}_norm'] = (df_temp.loc[:,yvar] - baseline_mean)/baseline_std
                df_temp.loc[:,f'{yvar}_znorm'] =  stats.zscore(df_temp.loc[:,f'{yvar}_norm'], ddof=1)
                df_cat.append(df_temp)
        df_idx = pd.concat(df_cat).reset_index(drop=True)
        dfs.append(df_idx)
    df_time = pd.concat(dfs).reset_index(drop=True)
    return df_time
    
################################################################################

def event_avg_plot(df, yvar, trace='dFF', xlim=None, ylim=None):
    
    label_size = 24
    title_size = 28
    xlab = 'Time From Event (second)'
    ylab = r'$\Delta F/F$ (%)' if trace is 'dFF' else r'$\Delta R/R$ (%)'
    tick_size = 18
    label_size = 24
    title_size = 28
    events = ['Reward', 'Lever', 'Rewarded Mag', 'Unrewarded Mag']
    color = 0 if trace is 'dFF' else 4
    for event in events:
        event_mean = df.groupby([f'{event} Time']).mean().reset_index()
        Xevent = event_mean.loc[:,f'{event} Time']
        Yevent = event_mean.loc[:,f'{yvar}']
        event_error = df.groupby([f'{event} Time','Animal']).mean().groupby([f'{event} Time']).std().reset_index()
        Yerr = event_error.loc[:,f'{yvar}']
        fig = plt.figure(figsize=(16,10))  # create a figure object
        ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
        # plot aesthetics
        ax.set_ylabel(ylab, size=label_size)
        ax.set_xlabel(xlab, size=label_size)
        ax.margins(x=0)
        title = f'{event}: {trace}'
        ax.set_title(title, size=title_size)
        # plot the data
        line = ax.plot(Xevent, Yevent, color=jr_pal[color])
        ax.axhline(y=0, linestyle='--', color='black', linewidth=0.6)
        ax.axvline(x=0, linestyle='-', color='black', linewidth=0.6)
        ax.fill_between(Xevent, Yevent-Yerr, Yevent+Yerr, facecolor=line[0].get_color(), alpha=0.15)
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        
################################################################################

def reward_summary_plots(df, yvar, trace='dFF', magPos=1, clickPos=1, levPos=1):
    
    CRF = df.loc[df['Animal'].str.contains('CRF')]
    RR5 = df.loc[df['Animal'].str.contains('RR5')]
    sessions = [CRF, RR5]

    events = ['Reward', 'Lever', 'Rewarded Mag', 'Unrewarded Mag']

    toPlot = []
    toPlotError = []
    for i, session in enumerate(sessions):
        for event in events:
            event_mean = session.groupby([f'{event} Time']).mean().reset_index()
            Xevent = event_mean.loc[:,f'{event} Time']
            Yevent = event_mean.loc[:,f'{yvar}']
            event_error = session.groupby([f'{event} Time','Animal']).mean().groupby([f'{event} Time']).std().reset_index()
            Yerr = event_error.loc[:,f'{yvar}']
            toPlot.append(Yevent)
            toPlotError.append(Yerr)
    # CRF Reward = 0, CRF Lever = 1, CRF Rewarded Mag = 2, CRF Unrewarded Mag = 3
    # RR5 Reward = 4, RR5 Lever = 5, RR5 Rewarded Mag = 6, RR5 Unrewarded Mag = 7
    toPlot.append((toPlot[3]+toPlot[7])/2)
    toPlotError.append((toPlotError[3]+toPlotError[7])/2)
    plots = ['Levers', 'Magazine Entries']
    col=[0,1,2] if trace == 'dFF' else [3,4,6]
    for j, plot in enumerate(plots):
        fig = plt.figure(figsize=(16,10))  # create a figure object
        ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
        label_size = 28
        title_size = 28
        xlab = 'Time From Event (second)'
        ylab = r'$\Delta F/F$ (%)' if trace is 'dFF' else r'$\Delta R/R$ (%)'
        tick_size = 18
        label_size = 24
        title_size = 28
        # plot aesthetics
        ax.set_ylabel(ylab, size=label_size)
        ax.set_xlabel(xlab, size=label_size)
        ax.margins(x=0)
        ax.axhline(y=0, linestyle='--', color='black', linewidth=0.6)
        ax.axvline(x=0, linestyle='-', color='black', linewidth=0.6)
        if j==1:
            # plot the data
            line = ax.plot(Xevent, toPlot[2], color=jr_pal[col[0]], label = 'CRF')
            ax.fill_between(Xevent,toPlot[2]-toPlotError[2],toPlot[2]+toPlotError[2],facecolor=jr_pal[col[0]], alpha=0.15)
            line = ax.plot(Xevent, toPlot[6], color = jr_pal[col[1]], label = 'RR5')
            ax.fill_between(Xevent, toPlot[6]-toPlotError[6],toPlot[6]+toPlotError[6],facecolor=jr_pal[col[1]], alpha=0.15)
            line = ax.plot(Xevent, toPlot[8], color = jr_pal[col[2]], label = 'Unrewarded')
            ax.fill_between(Xevent,toPlot[8]-toPlotError[8],toPlot[8]+toPlotError[8],facecolor=jr_pal[col[2]], alpha=0.15)
            ax.legend(prop={'size': 20})
            title = f'{plot}: {trace}'
            ax.set_title(title, size=title_size)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.annotate('Overlap With Lever Press', (-2.9,magPos), fontsize=14)
            ax.axvline(x=-1.3, linestyle='dotted', color='gray', linewidth=1.5)
        elif j==0:
            # plot the data
            line = ax.plot(Xevent, toPlot[0], color=jr_pal[col[0]], label = 'CRF')
            ax.fill_between(Xevent,toPlot[0]-toPlotError[0],toPlot[0]+toPlotError[0],facecolor=jr_pal[col[0]], alpha=0.15)
            line = ax.plot(Xevent, toPlot[4], color = jr_pal[col[1]], label = 'RR5')
            ax.fill_between(Xevent, toPlot[4]-toPlotError[4],toPlot[4]+toPlotError[4],facecolor=jr_pal[col[1]], alpha=0.15)
            line = ax.plot(Xevent, toPlot[5], color = jr_pal[col[2]], label = 'Unrewarded')
            ax.fill_between(Xevent, toPlot[5]-toPlotError[5],toPlot[5]+toPlotError[5],facecolor=jr_pal[col[2]], alpha=0.15)
            ax.legend(prop={'size': 20})
            title = f'{plot}: {trace}'
            ax.set_title(title, size=title_size)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.annotate('Click if Rewarded',(-1.3,clickPos), fontsize = 16)
            plt.annotate('Overlap With Magazine Entry', (1.5,levPos), fontsize=14)
            ax.axvline(x=1.4, linestyle='dotted', color='gray', linewidth=1.5)
            
################################################################################

def categorize_TST_data(df):
    pd.options.mode.chained_assignment = None
    df_list = []
    for idx in df['Animal'].unique():
        # subset individual animal
        df_animal = df.loc[df['Animal'] == idx, :]
        # find indices of rewarded lever presses and filter to only select single timepoint
        lifts = df_animal.loc[(df_animal['TTL2'] == 0)].index.tolist()
        index=1
        while index < len(lifts):
            if lifts[index] - lifts[index - 1] == 1:
                del lifts[index]
            else:
                index += 1 

        for i, lift in enumerate(lifts):
            df_animal.at[lift,'Category'] = f'Lift {i}'

        df_list.append(df_animal)

    df = pd.concat(df_list).reset_index(drop=True)
    return df

################################################################################

def fp_TST_trace(df, yvar='465nm_shift/405nm_shift_norm',
             xvar='time',
             session='Tail Suspension Test',
             trace='raw',
             fig_size=(20, 10),
             xlim=None,
             ylim =None,
             save_fig=False, fig_path=None, **kwargs):
    """
    Plot excitation and isosbestic fluorescence. 

    Parameters
    ----------
    df : DataFrame
        Data to plot. If using trace = 'dFF' DataFrame must contain dFF values.
    yvar : str, optional
        Column containing excitation fluorescence, by default '465nm'
    yiso : str, optional
        Name of column with raw isosbestic values, by default '405nm'
    xvar : str, optional
        Column containing time values, by default 'time'
    session : str, optional
        Name of session, used for figure title and file name if saving, by default 'Training'
    trace : str, optional
        Plot raw traces or dFF traces, by default 'raw'
    Yiso : bool, optional
        Plot isosbestic signal with excitation, by default True
    fig_size : tuple, optional
        Specify figure size, by default (20, 10)
    xlim : tuple, optional
        Specify x-axis limits, by default None
    pltBiexp
    save_fig : bool, optional
        Save the figure, by default False. See Notes for more info.
        
    Notes
    -----
    If using save_fig, can specify a fig_path (default is ~/Desktop).
    """
    yvar = yvar  if trace is 'raw' else yvar + f'_{trace}'
    
    # plot aesthetics variables
    xlab = 'Time (sec)'
    ylab = r'$\Delta R/R$ (%)'
    
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
            title = f'{session}: {idx}: dRR'
            ax.set_title(title, size=title_size)
            X = df_plot.loc[:, xvar]
            Y = df_plot.loc[:, yvar]
            ax.plot(X, Y, color=jr_pal[4], label=yvar)
            ax.legend(fontsize=16, loc='upper right', bbox_to_anchor=(1, 1.05))
            for lift in df_plot.loc[df_plot['Category']=='Lift']['time']:
                ax.axvline(x=lift, linestyle='dotted', color='black', linewidth=1.5)


        plt.tick_params(axis='both', labelsize=tick_size, width=2, length=6)
        
        if xlim:
            plt.xlim(xlim)
        if xlim:
            plt.xlim(xlim)