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
kp_pal = ['#2b88f0', #blue
          '#EF862E', #orange
          '#00B9B9', #cyan
          '#9147B1', #purple
          '#28A649', #green
          '#F97B7B', #salmon
          '#490035', #violet
          '#bdbdbd'] #gray


def set_palette(color_pal=None, show=False):
    """Set default color palette."""
    color_pal = kp_pal if color_pal is None else color_pal
    sns.set_palette(color_pal)
    if show:
        sns.palplot(color_pal)
    else:
        return color_pal
    
################################################################################

def load_expt_config(config_path):
    """Load expt_info.yaml to obtain project_path, raw_data_path, fig_path,
    and dict containing any group info.
    
     Parameters
     ----------
     config_path : str
        Path to the project yaml file.
     
     Returns
     -------
     expt_info : YAML object
        YAML object containing relevant experiment information.
    """
    try:
        with open(config_path, 'r') as file:
            expt = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file')
        raise e

    return expt

################################################################################

def import_data(file_name, yvar='465nm', yvar_out=2,
                input_ch=1, animal_id='file_name'):
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
                             'DI/O-2':'TTL2',
                             'DI/O-3':'TTL3',
                             'DI/O-4':'TTL4', 
                             'Time(s)':'time',
                             f'AIn-{input_ch} - Raw':f'raw_{input_ch}', 
                             f'AIn-{input_ch} - Dem (AOut-{ref_out})':'405nm',
                             f'AIn-{input_ch} - Dem (AOut-{yvar_out})':f'{yvar}',
                             f'AOut-{ref_out}':'405nm_LED', 
                             f'AOut-{yvar_out}':f'{yvar}_LED'})
          .reindex(columns=['Animal', 'time', 'TTL1', 'TTL2', 'TTL3', 'TTL4', 
                            '405nm', yvar]))
    # drop unused TTL columns, convert used to int
    df = df.dropna(axis=1)
    # When using Med-Associates SG-231 TTL generator, last value before signal onset is not 0 or 1.
    # It is < 0.5 when 0->1 and > 0.5 when 1->0
    ttl_cols = df.columns.str.contains('TTL')
    df.loc[:, ttl_cols] = np.round(df.loc[:, ttl_cols])
    df.loc[:, ttl_cols] = df.loc[:, ttl_cols].astype(int)
    # drop any TTL channels with all 1s or 0s
    for col in df.loc[:, ttl_cols].columns:
        if len(pd.unique(df.loc[:, col])) == 1:
            df = df.drop(col, axis=1)
    return df

################################################################################

def load_session_data(file_dir='fp_file_path', yvar='465nm', yvar_out=2,
                      input_ch=1, TTL_session_ch=1, TTL_on=0):
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
        df_temp = import_data(current_file, yvar=yvar, yvar_out=yvar_out, 
                              input_ch=input_ch, animal_id=file)
        # Only include data from the time that MedAssociates SG-231 TTL Generator is on
        df_temp = TTL_session(df_temp, TTL_session_ch, TTL_on)
        df_list.append(df_temp)
    df = pd.concat(df_list)
    df = df[df['time'] < int(max(df['time']))].reset_index(drop=True)
    
    return df

################################################################################

def load_session_noTTL(file_dir='fp_file_path', yvar='465nm', yvar_out=2,
                      input_ch=1, TTL_session_ch=1, TTL_on=0):
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
        df_temp = import_data(current_file, yvar=yvar, yvar_out=yvar_out, 
                              input_ch=input_ch, animal_id=file)
        # Only include data from the time that MedAssociates SG-231 TTL Generator is on
#         df_temp = TTL_session(df_temp, TTL_session_ch, TTL_on)
        df_list.append(df_temp)
    df = pd.concat(df_list).reset_index(drop=True)
#     df = df[df['time'] < int(max(df['time']))]
    
    return df

################################################################################

def import_Tian_data(file_name, yvar='465nm', yvar_out=2,
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
                             'DI/O-2':'TTL2',
                             'DI/O-3':'TTL3',
                             'DI/O-4':'TTL4', 
                             'Time(s)':'time',
                             f'AIn-{input_ch} - Raw':f'raw_{input_ch}', 
                             f'AIn-{yvar_out} - Dem (AOut-{yvar_out})':f'{yvar} control',
                             f'AIn-{input_ch} - Dem (AOut-{yvar_out})':f'{yvar} sensor',
                             f'AOut-{ref_out}':'405nm_LED', 
                             f'AOut-{yvar_out}':f'{yvar}_LED'})
            .reindex(columns=['Animal', 'time', 'TTL1', f'{yvar} control', f'{yvar} sensor']))
    return df


################################################################################

def load_Tian_session_data(file_dir='fp_file_path', yvar='465nm', yvar_out=2,
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
            df_temp = import_Tian_data(current_file, yvar=yvar, yvar_out=yvar_out, 
                              input_ch=input_ch, animal_id=file)
        # Only include data from the time that MedAssociates SG-231 TTL Generator is on
        df_temp = TTL_session(df_temp, TTL_session_ch, TTL_on)
        df_list.append(df_temp)
    df = pd.concat(df_list)
    df = df[df['time'] < int(max(df['time']))].reset_index(drop=True)
    
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
#         df_temp = TTL_session(df_temp, TTL_session_ch, TTL_on)
        df_list.append(df_temp)
    df = pd.concat(df_list).reset_index(drop=True)
#     df = df[df['time'] < int(max(df['time']))].reset_index(drop=True)
    
    return df

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
        df_temp = import_Tian_fear_data(current_file, yvar=yvar, yvar_out=yvar_out, 
                              input_ch=input_ch, animal_id=file)
        # Only include data from the time that MedAssociates SG-231 TTL Generator is on
        df_temp = TTL_session(df_temp, TTL_session_ch, TTL_on)
        df_list.append(df_temp)
    df = pd.concat(df_list).reset_index(drop=True)
#     df = df[df['time'] < int(max(df['time']))]
    
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
    first_row = min(df[df[ttl_ch] == TTL_on].index)
    last_row = max(df[df[ttl_ch] == TTL_on].index)
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

def iso_biexponential(df):
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
        Y = df.loc[df['Animal'] == idx, '405nm']
        popt, pcov = curve_fit(biexponential,X,Y,p0=(0.2,0,0.2,0),maxfev=10000)
        isoBiexp = biexponential(X,*popt)
        df.loc[df['Animal'] == idx, '405nm_biexp'] = isoBiexp
    
    return df
    
################################################################################    

def biexponential(x, a, b, c, d):
    return a * np.exp(b * x) + c * np.exp(d * x)

################################################################################  

def fit_biexponential(df):
    
    for idx in df['Animal'].unique():
        X = df.loc[df['Animal'] == idx, 'time']
        Y = df.loc[df['Animal'] == idx, '465nm']
        isoFitY = df.loc[df['Animal'] == idx, '405nm_biexp']
        isoBiexp = np.vstack((X,isoFitY)).T
        model = LinearRegression().fit(isoBiexp, Y)
        Ypred = model.predict(isoBiexp)
        df.loc[df['Animal'] == idx, '465nm_biexp'] = Ypred
        dFFBiexp = 100*((Y-Ypred)/Ypred)
        df.loc[df['Animal'] == idx, '465nm_dFF_biexp'] = dFFBiexp
        df.loc[df['Animal'] == idx, '465nm_dFF_zscore_biexp'] = stats.zscore(dFFBiexp, ddof=1)

    return df

################################################################################

def plt_raw_with_biexp(df, channel):
    
    for idx in df['Animal'].unique():
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 2, 2])
        X = df.loc[df['Animal'] == idx, 'time']
        Y = df.loc[df['Animal'] == idx, channel]
        isoBiexp = df.loc[df['Animal'] == idx, channel+'_biexp']
        ax.scatter(X, Y, s=20, color='#00b3b3', label='Data')
        ax.plot(X,isoBiexp,linestyle = '--')
        
################################################################################

def plt_biexp_dFF(df):
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 2, 2])
    
    for idx in df['Animal'].unique():
        X = df.loc[df['Animal'] == idx, 'time']
        Y = df.loc[df['Animal'] == idx, '465nm_dFF_biexp']
        ax.plot(X,Y,linestyle = '-')

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

def dSig_ref(df, Y_sig='465nm', Y_ref='405nm'):
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
        dSig = Y/X
        df.loc[df['Animal'] == idx, f'{Y_sig}/{Y_ref}'] = dSig
    
    return df

################################################################################

def trial_drr(df, yvar):
    """
    Trial-normalize data aligned to a stimulus onset.

    Parameters
    ----------
    df : DataFrame
        Trial-level data.
    yvar : str
        Column in df to trial-normalize.

    Returns
    -------
    DataFrame
        Adds column named {yvar}_norm to df.
    """
    subj_data = []
    for idx in df['Animal'].unique():
        df_temp = df.loc[df['Animal'] == idx, :]
        norm_vals = []
        for i in df_temp['Trial'].unique():
            df_trial_baseline = df_temp.loc[(df_temp['Trial'] == i) & 
                                            (df_temp['time_trial'] < 0), :]
            pre_CS_mean = np.mean(df_trial_baseline[yvar])
            norm_vals.append((df_temp.loc[df_temp['Trial'] == i, yvar] - pre_CS_mean)/pre_CS_mean)
            # flatten normalized values from each trial
            norm_vals_flat = [item for sublist in norm_vals for item in sublist]
        subj_data.append(norm_vals_flat)
    # flatten normalized values from each subject
    df[f'{yvar}_drr'] = [item for sublist in subj_data for item in sublist]
    
    return df

################################################################################

def baseline_fit(df):
    for idx in df['Animal'].unique():
        X = df.loc[df['Animal'] == idx, 'time']
        Y = df.loc[df['Animal'] == idx, '465nm']
        baselineY = peakutils.baseline(Y)
        df.loc[df['Animal'] == idx, '465nm_baseline'] = baselineY
        dFFBaseline = 100*((Y-baselineY)/baselineY)
        df.loc[df['Animal'] == idx, '465nm_dFF_baseline'] = dFFBaseline
        df.loc[df['Animal'] == idx, '465nm_dFF_zscore_baseline'] = stats.zscore(dFFBaseline, ddof=1)

    return df

################################################################################

def DREADDs_equal_sess(df):
    """
        Plot trial-averaged dFF signal.

        Parameters
        ----------
        df : DataFrame
            Trial-level DataFrame from trials_df()
        yvar : str, optional
            Column containing fluorescence values to plot, by default '465nm_dFF_znorm'
        xvar : str, optional
            Column containing trial-level timepoints, by default 'time_trial'
        """    
    
    # find shortest session and resize sessions so all sessions are same length
    sessTime = min([len(df[df['Animal'].str.contains(Animal)]) for Animal in df['Animal'].unique()])
    df_list = []
    for idx in df['Animal'].unique():
        # subset individual animal to plot
        df_animal = df.loc[df['Animal'] == idx, :]
        sessStart = round((len(df_animal)-sessTime)/2)
        dt = df_animal.iloc[1].loc['time']-df_animal.iloc[0].loc['time']
        df_temp = df_animal.iloc[sessStart:(sessStart+sessTime)]
        df_temp = df_temp.assign(time = np.linspace(0.0,dt*sessTime,sessTime))
        df_list.append(df_temp)

    df = pd.concat(df_list).reset_index(drop=True)
    return df

################################################################################

def DREADDs_normalize(df, yvar, baseline = 300):
    """
    Normalize data to baseline recording of session.

    Parameters
    ----------
    df : DataFrame
        Trial-level data.
    yvar : str
        Column in df to trial-normalize.
    baseline : int
        Amount of time to use as baseline for normalization

    Returns
    -------
    DataFrame
        Adds column named {yvar}_norm to df.
    """
    subj_data = []
    subj_zdata = []
    for idx in df['Animal'].unique():
        df_temp = df.loc[df['Animal'] == idx, :]
        znorm_vals = []
        norm_vals = []

        df_baseline = df_temp.loc[(df_temp['time'] < baseline), :]
        baseline_mean = np.mean(df_baseline[yvar])
        baseline_std = np.std(df_baseline[yvar])
        norm_vals.append((df_temp.loc[:,yvar] - baseline_mean)/baseline_std)
        znorm_vals.append(stats.zscore(df_temp.loc[:,yvar], ddof=1))
        # flatten normalized values from each trial
        norm_vals_flat = [item for sublist in norm_vals for item in sublist]
        znorm_vals_flat = [item for sublist in znorm_vals for item in sublist]
        subj_data.append(norm_vals_flat)
        subj_zdata.append(znorm_vals_flat)
        
        animalArea = np.trapz(norm_vals_flat)
        df.loc[df['Animal'] == idx, '465nm_dFF_zscore_norm_area'] = animalArea
        animalZArea = np.trapz(znorm_vals_flat)
        df.loc[df['Animal'] == idx, '465nm_dFF_zscore_norm_zarea'] = animalZArea
    # flatten normalized values from each subject
    df[f'{yvar}_norm'] = [item for sublist in subj_data for item in sublist]
    df[f'{yvar}_znorm'] = [item for sublist in subj_zdata for item in sublist]
    return df

################################################################################

def DREADDs_AUC(dfGroups, yvar, xvar='time'):
    """
    Calculate the area under the curve using trapezoidal method of integration

    Parameters
    ----------
    df : DataFrame
        Trial-level data.
    yvar : str
        Column in df to trial-normalize.

    Returns
    -------
    DataFrame
        Adds column named {yvar}_AUC to df.
    """
    area_dict = {group.name:[] for group in dfGroups}
    
    for group in dfGroups:
        for idx in group['Animal'].unique():
            animalArea = group.loc[group['Animal'] == idx, yvar].iloc[0]
            area_dict[group.name].append(animalArea)

    return area_dict
        
################################################################################

def trim_sess(df, trim):
    """
        Plot trial-averaged dFF signal.

        Parameters
        ----------
        df : DataFrame
            Trial-level DataFrame from trials_df()
        trim : int
            Amount of time to cut from session
        """
    df_list = []
    
    for idx in df['Animal'].unique():
        # subset individual animal to plot
        df_animal = df.loc[df['Animal'] == idx, :]
        df_animal_trim = df_animal.loc[df_animal['time']>trim, :]
        dt = df_animal_trim.iloc[1].loc['time']-df_animal_trim.iloc[0].loc['time']
        sessLength =len(df_animal_trim)
        df_animal_trim = df_animal_trim.assign(time = np.linspace(0.0,dt*sessLength,sessLength))
        df_list.append(df_animal_trim)

    df = pd.concat(df_list).reset_index(drop=True)
    return df            

################################################################################

def reward_TTL_sess(df, TTL_ch = 2):
    """
        Plot trial-averaged dFF signal.

        Parameters
        ----------
        df : DataFrame
            Trial-level DataFrame from trials_df()
        TTL_ch : int
            Amount of time to cut from session
        """
    df_list = []
    
    for idx in df['Animal'].unique():
        # subset individual animal to plot
        df_animal = df.loc[df['Animal'] == idx, :]
        lastFrame = df_animal.loc[(df_animal[f'TTL{TTL_ch}'] == 0)].index[-1]
        firstFrame = df_animal.loc[(df_animal[f'TTL{TTL_ch}'] == 0)].index[0]
        behaviorSession = df_animal.loc[firstFrame:lastFrame, :]
        dt = behaviorSession.iloc[1].loc['time']-behaviorSession.iloc[0].loc['time']
        sessLength =len(behaviorSession)
        behaviorSession = behaviorSession.assign(time = np.linspace(0.0,dt*sessLength,sessLength))
        df_list.append(behaviorSession)

    df = pd.concat(df_list).reset_index(drop=True)
    return df      

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
            rewardedMags.append(df_animal.loc[(df_animal.index>reward) & (df_animal['TTL2'] == 0),:].iloc[0].name)
            unrewardedMags = df_animal.loc[(df_animal.index>reward) & (df_animal['TTL2'] == 0),:].iloc[1:].index.tolist()
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

def calc_dFF_dRR(df,yvar='465nm',yiso='405nm'):

    df_index_list=[]
    for idx in df['Animal'].unique():
        df_animal = df.loc[df['Animal'] == idx,:]
        rewarded_lever =  df_animal.loc[df_animal['Category'] == 'Rewarded Lever'].index.tolist()
        unrewarded_lever = df_animal.loc[df_animal['Category'] == 'Unrewarded Lever'].index.tolist()
        rewarded_mag = df_animal.loc[df_animal['Category'] == 'Rewarded Mag'].index.tolist()
        unrewarded_mag = df_animal.loc[df_animal['Category'] == 'Unrewarded Mag'].index.tolist()
        categories = [rewarded_lever,unrewarded_lever,rewarded_mag,unrewarded_mag]

        for i, category in enumerate(categories):
            df_list=[]
            for index in category:
                dFF_vals = []
                dRR_vals = []
                dFF_vals_flat = []
                dRR_vals_flat = []
                dFF_index_data = []
                dRR_index_data= []
                dFF_zscore_vals = []
                dRR_zscore_vals = []
                dFF_zscore_vals_flat = []
                dRR_zscore_vals_flat = []
                dFF_zscore_index_data = []
                dRR_zscore_index_data= []

                time = df_animal.loc[index,'time']
                df_baseline = df_animal.loc[(df_animal['time']>time-3) & (df_animal['time']<time-1)]
                df_temp = df_animal.loc[(df_animal['time']>time-3) & (df_animal['time']<time+3)]
                if i == 0:
                    df_temp.at[index,'Category']=f'True Rewarded Lever'
                elif i ==1:
                    df_temp.at[index,'Category']=f'True Unrewarded Lever'
                elif i==2:
                    df_temp.at[index,'Category']=f'True Rewarded Mag'
                elif i==3:
                    df_temp.at[index,'Category']=f'True Unrewarded Mag'
                baseline_mean = np.mean(df_baseline[yvar])
                baseline_405_mean = np.mean(df_baseline[yiso])
                baseline_ratio = baseline_mean/baseline_405_mean
                dFF_vals.append((df_temp.loc[:,yvar] - baseline_mean)/baseline_mean)
                dRR_vals.append(((df_temp.loc[:,yvar]/df_temp.loc[:,yiso])-baseline_ratio)/baseline_ratio)
                dFF_vals_flat = [item for sublist in dFF_vals for item in sublist]
                dRR_vals_flat = [item for sublist in dRR_vals for item in sublist]
                dFF_index_data.append(dFF_vals_flat)
                dRR_index_data.append(dRR_vals_flat)

                df_temp[f'{yvar}_dFF'] = [item for sublist in dFF_index_data for item in sublist]
                df_temp[f'{yvar}_dRR'] = [item for sublist in dRR_index_data for item in sublist]

                dFF_zscore_vals.append(stats.zscore(df_temp.loc[:,f'{yvar}_dFF'], ddof=1))
                dRR_zscore_vals.append(stats.zscore(df_temp.loc[:,f'{yvar}_dRR'], ddof=1))
                dFF_zscore_vals_flat = [item for sublist in dFF_zscore_vals for item in sublist]
                dRR_zscore_vals_flat = [item for sublist in dRR_zscore_vals for item in sublist]
                dFF_zscore_index_data.append(dFF_zscore_vals_flat)
                dRR_zscore_index_data.append(dRR_zscore_vals_flat)
                df_temp[f'{yvar}_dFF_zscore'] = [item for sublist in dFF_zscore_index_data for item in sublist]
                df_temp[f'{yvar}_dRR_zscore'] = [item for sublist in dRR_zscore_index_data for item in sublist]
                df_list.append(df_temp)
            
            if(len(df_list)!=0):
                category = pd.concat(df_list)
                df_index_list.append(category)

    df = pd.concat(df_index_list).reset_index(drop=True)
    return df

################################################################################

def reward_trials(df):
    pd.options.mode.chained_assignment = None
    rewardIndex = df.loc[(df['TTL3'] == 0) & (df['TTL4'] == 0)].index.tolist()

    df_list = []
    for idx in df['Animal'].unique():
        # subset individual animal
        df_animal = df.loc[df['Animal'] == idx, :]
        rewards = []
        rewards=[idr for idr in rewardIndex if (idr>df_animal.index[0]) & (idr<df_animal.index[-1])]
        trialStartX = []
        trialStartSkip = []
        trialStartX = [rewards[rID] for rID in range(len(rewards)-1) if rewards[rID+1] != rewards[rID]+1]

        for i,trial in enumerate(trialStartX):
            current = i+1
            while (current < len(trialStartX)-1):
                df_trial = df_animal.loc[(df_animal.index > trial) & (df_animal.index <trialStartX[current])]
                if 0 in df_trial['TTL2'].tolist():
                    break
                else:
                    trialStartSkip.append(current)
                    current = current+1
        trialStartSkip = sorted(list(set(trialStartSkip)))
        for i,skip in enumerate(trialStartSkip):
            trialStartX.pop(skip-i)

        df_animal.loc[(df_animal.index < trialStartX[0]-1), 'trial'] = 'Baseline'
        for i,trial in enumerate(trialStartX):
            if i < (len(trialStartX)-1):
                trialNum = i+1
                df_animal.loc[(df_animal.index>trial-2)&(df_animal.index<trialStartX[i+1]-1),'trial'] = f'Trial {trialNum}'
                #be careful if ever using a frequency different than 10 Hz since number of frames with TTL may change
            else:
                trialNum = i+1
                df_animal.loc[(df_animal.index > trial-2), 'trial'] = f'Trial {trialNum}'

        df_list.append(df_animal)

    df = pd.concat(df_list).reset_index(drop=True)
    return df  

################################################################################

def reward_normalize_by_trial(df, yvar ='465nm', baseline = 2, prePressX = 3, plot = False):
    """
    Normalize data to baseline recording of session.

    Parameters
    ----------
    df : DataFrame
        Trial-level data.
    yvar : str
        Column in df to trial-normalize.
    baseline : int
        Amount of time to use as baseline for normalization

    Returns
    -------
    DataFrame
        Adds column named {yvar}_norm to df.
    """
    
    df_list = []
    for idx in df['Animal'].unique():
        df_animal = df.loc[df['Animal'] == idx, :]
        for trial in df_animal['trial'].unique():
            if trial != 'Baseline':
                x=0
                X=[]
                Y=[]
                subj_data = []
                subj_zdata = []
                znorm_vals = []
                norm_vals = []
                rewardsTrue = []
                trialTimes = df_animal.loc[df_animal['trial']==trial]['time'].tolist()
                trialStart = trialTimes[0]
                trialEnd = trialTimes[-1]
                base = trialStart - prePressX
                baselineEnd = base+baseline
                df_temp = df_animal.loc[(df_animal['time']>base) & (df_animal['time']<trialEnd),:].reset_index(drop=True)
                df_indices = df_temp.index.tolist()
                df_baseline = df_temp.loc[(df_temp['time']>base)&(df_temp['time']<baselineEnd), :]
                rewards = df_temp.loc[(df_temp['TTL3'] == 0) & (df_temp['TTL4'] == 0)].index.tolist()
                index = 1
                while index < len(rewards):
                    if rewards[index] - rewards[index - 1] == 1:
                        del rewards[index]
                    else:
                        index += 1
                mags = df_temp.loc[(df_temp['TTL2'] == 0)].index.tolist()
                index=1
                while index < len(mags):
                    if mags[index] - mags[index - 1] == 1:
                        del mags[index]
                    else:
                        index += 1
                levers = df_temp.loc[(df_temp['TTL3'] == 0)].index.tolist()
                index=1
                while index < len(levers):
                    if levers[index] - levers[index - 1] == 1:
                        del levers[index]
                    else:
                        index += 1
                baseline_mean = np.mean(df_baseline[yvar])
                baseline_std = np.std(df_baseline[yvar])
                norm_vals.append((df_temp.loc[:,yvar] - baseline_mean)/baseline_std)
                znorm_vals.append(stats.zscore(df_temp.loc[:,yvar], ddof=1))
                # flatten normalized values from each trial
                norm_vals_flat = [item for sublist in norm_vals for item in sublist]
                znorm_vals_flat = [item for sublist in znorm_vals for item in sublist]
                for i, value in enumerate(norm_vals_flat):
                    if i<len(df_baseline):
                        df_temp.loc[df_indices[i], [f'{yvar}_norm_baseline']] = norm_vals_flat[i]
                        df_temp.loc[df_indices[i], [f'{yvar}_znorm_baseline']] = znorm_vals_flat[i]
                    else:
                        df_temp.loc[df_indices[i], [f'{yvar}_norm']] = norm_vals_flat[i]
                        df_temp.loc[df_indices[i], [f'{yvar}_znorm']] = znorm_vals_flat[i]
                        Y.append(norm_vals_flat[i])
                        X.append(x)
                        x+=1
                        
                
                if plot == True:
                    label_size = 24
                    title_size = 28
                    session = 'dLight Reward'
                    trace = 'dFF'
                    xlab = 'Time (millisecond)'
                    ylab = 'Fluorescence (au)' if trace is 'raw' else r'$\Delta F/F$ (%)'
                    yvar='465nm'
                    yiso = '405nm'
                    xvar='time'
                    fig = plt.figure(figsize=(20,10))  # create a figure object
                    ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
                    # plot aesthetics
                    ax.set_ylabel(ylab, size=label_size)
                    ax.set_xlabel(xlab, size=label_size)
                    ax.margins(x=0)
                    title = f'{idx}: {trial}: dFF'
                    ax.set_title(title, size=title_size)
                    ax.plot(X, Y, color=kp_pal[5], label= f'{yvar} dFF',linewidth=2)
                    ax.axhline(y=0, linestyle='--', color='black', linewidth=1.2)
                    for lever in levers:
                        ax.axvline(x=lever, linestyle='--', color = 'black', linewidth=1.2, label = 'Lever Press')
                    for mag in mags:
                        ax.axvline(x=mag, linestyle='-', color = kp_pal[7], linewidth=1.5, label = 'Magazine Entry')
                    
                    handles, labels = plt.gca().get_legend_handles_labels()
                    newLabels, newHandles = [], []
                    for handle, label in zip(handles, labels):
                      if label not in newLabels:
                        newLabels.append(label)
                        newHandles.append(handle)
                    plt.legend(newHandles, newLabels, fontsize=16, loc=1)

                df_list.append(df_temp)
                
    df = pd.concat(df_list).reset_index(drop=True)
            
    return df

        