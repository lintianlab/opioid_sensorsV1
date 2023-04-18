""" Load photometry data.
General workflow: Wrap these steps around load_session_data()
    1. load_session_data()
    2. low-pass filter()
    3. resample_data()
    4. fit_linear() 
"""
import os
import scipy.signal as signal
import scipy.stats as stats
import numpy as np
import pandas as pd
import yaml
import statsmodels.api as sm
#from photometry_analysis import fp_viz, tfc_dat
################################################################################
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

def butter_bandpass_filter(data, LFcut=25, fs=100, order=4):
    """
    Butterworth bandpass wrapper function.
    """
    def _butter_bandpass(low_cutoff, sampfreq, filt_order):
        nyq = 0.5 * sampfreq
        low = low_cutoff / nyq
        return signal.butter(N=filt_order, Wn=low, analog=False, btype='low', output='sos')
    
    sos = _butter_bandpass(low_cutoff=LFcut, sampfreq=fs, filt_order=order)
    
    return signal.sosfiltfilt(sos, data)


################################################################################

def lowpass_filter(df, x1, x2, cutoff, fsamp, order):
    """
    Apply low-pass Butterworth filter to data

    Parameters
    ----------
    data : array_like
        Array containing time series data to apply low-pass filter to
    
    cutoff : int
        low-pass cutoff frequency (in Hz)
    fs : int
        The sampling frequency of data
    order : int
        The order of the filter

    Resources
    ---------
    https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter #pylint: disable=line-too-long
    """
    
    df[x1+'_raw'] = df[x1]
    df[x2+'_raw'] = df[x2]
    df[x1] = np.hstack([butter_bandpass_filter(df.query("Animal == @idx").loc[:, x1].values,
                                               LFcut=cutoff, fs=fsamp, order=order) 
                        for idx in df['Animal'].unique()])
    df[x2] = np.hstack([butter_bandpass_filter(df.query("Animal == @idx").loc[:, x2].values, 
                                               LFcut=cutoff, fs=fsamp, order=order) 
                        for idx in df['Animal'].unique()])
    return df


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
        df_subj.index = pd.to_timedelta(df_subj.index, unit='s')
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

def trial_normalize(df, yvar):
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
    subj_zdata = []
    for idx in df['Animal'].unique():
        df_temp = df.loc[df['Animal'] == idx, :]
        znorm_vals = []
        norm_vals = []
        for i in df_temp['Trial'].unique():
            df_trial_baseline = df_temp.loc[(df_temp['Trial'] == i) & 
                                            (df_temp['time_trial'] < 0), :]
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

def create_prism_data(data, yvar, epoch, agg_func):
    """
    Create wide-format data for loading statistcal software (e.g., Prism).
    
    Parameters
    ----------
    df_trial : DataFrame
        pandas DataFrame object containing data from trials_df()
    yvar : str 
        Independent variable used for agg_func (e.g., 465nm_dFFnorm)
    epoch : str
        must be 'shock', 'tone_on', 'tone_trace'
    agg_func : str
        How to aggregate the data. Must be 'mean', 'max', or 'min'
    
    """
    def convert_to_prism(df, yvar):
    
        col_order = df['phase'].unique()
        df = df.pivot_table(values=yvar, index=['Animal'], columns='phase')
        df = df.reindex(col_order, axis=1).reset_index()

        return df
    
    # average subject data across trials
    df_trial_avg = data.groupby(['Animal', 'time_trial']).mean().reset_index()
    
    if epoch == 'shock':
        df_trial_avg.loc[(df_trial_avg.time_trial >= 38) & (df_trial_avg.time_trial < 40), 
                         'phase'] = 'pre_shock'
        df_trial_avg.loc[(df_trial_avg.time_trial >= 40) & (df_trial_avg.time_trial < 42), 
                         'phase'] = 'shock'
        
    elif epoch == 'tone_on':
        df_trial_avg.loc[(df_trial_avg.time_trial >= -5) & (df_trial_avg.time_trial < 0), 
                         'phase'] = 'pre_tone'
        df_trial_avg.loc[(df_trial_avg.time_trial >= 0) & (df_trial_avg.time_trial < 5), 
                         'phase'] = 'tone'
    
    elif epoch == 'tone_trace':
        df_trial_avg.loc[(df_trial_avg.time_trial >= -20) & (df_trial_avg.time_trial < 0), 
                         'phase'] = 'baseline'
        df_trial_avg.loc[(df_trial_avg.time_trial >= 0) & (df_trial_avg.time_trial < 20), 
                         'phase'] = 'tone'
        df_trial_avg.loc[(df_trial_avg.time_trial >= 20) & (df_trial_avg.time_trial < 40), 
                         'phase'] = 'trace'
    
    else:
        raise ValueError('epoch must be: "shock", "tone_on", or "tone_trace" ')
    
    if agg_func == 'mean':
        df_trial_prism = convert_to_prism(df_trial_avg, yvar=yvar)
        df_trial_prism = df_trial_prism.dropna(axis=1)
    
    elif agg_func == 'max':
        df_trial_prism = convert_to_prism(df_trial_avg, yvar=yvar)
        df_trial_prism = df_trial_prism.dropna(axis=1)
        df_trial_max = df_trial_avg.loc[:, ['Animal', 'phase', yvar]]
        df_trial_max = df_trial_max.groupby(['Animal', 'phase']).max().reset_index()
        df_trial_prism = convert_to_prism(df_trial_max, yvar=yvar)

    elif agg_func == 'min':
        df_trial_prism = convert_to_prism(df_trial_avg, yvar=yvar)
        df_trial_prism = df_trial_prism.dropna(axis=1)
        df_trial_min = df_trial_avg.loc[:, ['Animal', 'phase', yvar]]
        df_trial_min = df_trial_min.groupby(['Animal', 'phase']).min().reset_index()
        df_trial_prism = convert_to_prism(df_trial_max, yvar=yvar)
    else:
        raise ValueError('agg_func must be: "mean", "max", or "min" ')

    return df_trial_prism


################################################################################
def calc_pre_post(df, event, t_pre, t_post, measure='mean'):
    """
    Compute the average over a defined pre and post period.
    
    Parameters
    ----------    
    df : DataFrame
        Pandas DataFrame with data to calculate over.
    t_pre: tuple
        Time points for pre-event period (start, end)
    t_post : tuple
        Time points for pre-event period (start, end)
    measure : str, optional
        Specify metric used to calculate pre-post, by default 'mean'.
    
    Returns
    -------
    DataFrame
        Averaged data across the give t_pre and t_post
    """
    
    df_pre = df[df['time_trial'].between(t_pre[0], t_pre[1])].reset_index(drop=True)
    df_post = df[df['time_trial'].between(t_post[0], t_post[1])].reset_index(drop=True)
    # add `epoch` column
    df_pre['epoch'] = f'pre-{event}'
    df_post['epoch'] = f'post-{event}'
    # recombine values and groupby new epoch var
    df_prepost = pd.concat([df_pre, df_post])
    
    if measure == 'mean':
        return df_prepost.groupby(['Animal', 'epoch']).mean().reset_index()
    elif measure == 'max':
        df_prepost = df_prepost.groupby(['Animal', 'time_trial', 'epoch']).mean().reset_index()
        return df_prepost.groupby(['Animal', 'epoch']).max().reset_index()


################################################################################

def pre_post_stats(df_prepost, yvar='465nm_dFF_znorm', values=False):
    """
    Compute a paired t-test for pre and post event.

    Parameters
    ----------
    df_prepost : DataFrame
        Output from calc_pre_post
    yvar : str
        Name of independent variable to compare, by default '465nm_dFF_znorm'.
    values : bool, optional
        Return the tstat and pval for the t-test, by default False.

    Returns
    -------
    (tstat, pval)
        Returns the t-statistic and the p-value from the paired t-test.
    """
    pre = df_prepost.loc[df_prepost['epoch'].str.contains('pre'), yvar]
    post = df_prepost.loc[df_prepost['epoch'].str.contains('post'), yvar]
    tstat, pval = stats.ttest_rel(pre, post)
    
    print(f't-statistic: {tstat}')
    print(f'p-value: {pval}')
    
    if values is True:
        return (tstat, pval)


################################################################################
