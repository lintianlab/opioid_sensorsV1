""" Process fear conditioning photometry data. """
from pathlib import Path
import pandas as pd
import numpy as np
from photometry_analysis import fp_dat
from photometry_analysis import JRfp_dat

################################################################################
################################################################################
def make_tfc_comp_times(baseline, n_trials, cs_dur, trace_dur, us_dur, iti_dur):
    """
    Generate a table of session components, their duration, and start/end times.

    Parameters
    ----------
    baseline : int
        Duration of baseline epoch.
    n_trials : int
        Number of trials in session.
    cs_dur : int
        CS duration.
    trace_dur : int
        Trace interval duration.
    us_dur : int
        US duration.
    iti_dur : int
        Length of inter-trial interval.

    Returns
    -------
    DataFrame
        Table containing 'component', 'duration', 'start', and 'end' for each component.
    """
    
    comp_times = []
    session_time = 0
    comp_times.append(['baseline', baseline, 0, baseline])
    session_time += baseline
    for t in range(n_trials):
        comp_times.append([f'tone-{t+1}', cs_dur, session_time, session_time+cs_dur])
        session_time += cs_dur
        comp_times.append([f'trace-{t+1}', trace_dur, session_time, session_time+trace_dur])
        session_time += trace_dur
        comp_times.append([f'shock-{t+1}', us_dur, session_time, session_time+us_dur])
        session_time += us_dur
        comp_times.append([f'iti-{t+1}', iti_dur, session_time, session_time+iti_dur])
        session_time += iti_dur
    
    comp_times_df = pd.DataFrame(comp_times, 
                                 columns=['component', 'duration', 'start', 'end'])

    return comp_times_df[comp_times_df['duration'] != 0]


################################################################################

def tfc_comp_times(session='train'):
    """
    Load component times for TFC protocols.
    
    Parameters
    ----------
    session : str
        Session to load times for, by default 'train'.
    
    Returns
    -------
    DataFrame of protocol component times.
    
    """
    curr_dir = str(Path(__file__).parents[0]) + '/files/'
    comp_labs_file = curr_dir + 'TFC phase components.xlsx'

    return pd.read_excel(comp_labs_file, sheet_name=session)

################################################################################

def find_tfc_components(df, session='train'):
    """
    Label pandas DataFrame with TFC session components.
    
    Parameters
    ----------
    df : DataFrame
        Data to label with `session` labels.
    
    session : str
        Session to load times for, by default 'train'.
    
    Returns
    -------
    DataFrame of protocol component times.
    
    """
    comp_labs = tfc_comp_times(session=session)
    session_end = max(comp_labs['end'])
    df_new = df.drop(df[df['time'] >= session_end].index)
    # search for time in sec, index into comp_labels
    # for start and end times
    for i in range(len(comp_labs['phase'])):
        df_new.loc[df_new['time'].between(comp_labs['start'][i], 
                                          comp_labs['end'][i]), 
                   'Component'] = comp_labs['phase'][i]
        
    return df_new

################################################################################

def label_phases(df, session='train'):
    """
    Label DataFrame with 'Phases' (used to trial average data)
    
    Parameters
    ----------
    df : DataFrame
        Data to label.
    
    session : str
        Session to load times for, by default 'train'.
    
    Returns
    -------
    DataFrame with new Phase column.
    
    """
    session_list = ['train', 'tone', 'ctx', 'extinction', 'cs_response', 'shock_response']
    session_type = [s for s in session_list if s in session][0]
    df = find_tfc_components(df, session=session_type)
    df.loc[:, 'Phase'] = df.loc[:, 'Component']
    # label tone, trace, and iti for all protocols
    df.loc[df['Phase'].str.contains('tone'), 'Phase'] = 'tone'
    df.loc[df['Phase'].str.contains('trace'), 'Phase'] = 'trace'
    df.loc[df['Phase'].str.contains('iti'), 'Phase'] = 'iti'
    # label shock phases for training data
    df.loc[df['Phase'].str.contains('shock'), 'Phase'] = 'shock'
    
    return df


################################################################################

def trials_df(df, session='train', 
              yvar='465nm_dFF', normalize=True,
              trial_start=-20, cs_dur=20, trace_dur=20, us_dur=2, iti_dur=120, drr = False):
    """
    1. Creates a dataframe of "Trial data", from (trial_start, trial_end) around each CS onset
    2. Normalizes dFF for each trial to the avg dFF of each trial's pre-CS period
    
    ! Session must be a sheet name in 'TFC phase components.xlsx'

    Parameters
    ----------
    df : DataFrame
        Session data to calculate trial-level data.
    session : str, optional
        Name of session used to label DataFrame, by default 'train'
    yvar : str, optional
        Name of data to trial-normalize, by default '465nm_dFF'
    normalize : bool, optional
        Normalize yvar to baseline of each trial, by default True
    trial_start : int, optional
        Start of trial, by default -20
    cs_dur : int, optional
        CS duration used to calculate trial time, by default 20
    us_dur : int, optional
        US duration, by default 2
    trace_dur : int, optional
        Trace interval duration, by default 20
    iti_dur : int, optional
        Length of inter-trial-interval; used to calculate trial time, by default 120
    
   
    Returns
    -------
    DataFrame
        Trial-level data with `yvar` trial-normalized.
    """
    df = label_phases(df, session=session)
    
    
    comp_labs = tfc_comp_times(session=session)
    tone_idx = [tone for tone in range(len(comp_labs['phase'])) 
                if 'tone' in comp_labs['phase'][tone]]
    iti_idx = [iti for iti in range(len(comp_labs['phase'])) 
               if 'iti' in comp_labs['phase'][iti]]
    # determine number of tone trials from label
    n_trials = len(tone_idx)
    n_subjects = len(df.Animal.unique())
    trial_num = int(1)
    # subset trial data (-20 prior to CS --> 100s after trace/shock)
    for tone, iti in zip(tone_idx, iti_idx):
        start = comp_labs.loc[tone, 'start'] + trial_start
        end = comp_labs.loc[iti, 'start'] + iti_dur + trial_start
        df.loc[(start <= df.time) & (df.time < end), 'Trial'] = int(trial_num)
        trial_num += 1
    # remove extra time points
    df = df.dropna().reset_index(drop=True)
    # check if last_trial contains extra rows and if so, drop them
    first_trial = df.query("Trial == Trial.unique()[0]")
    last_trial = df.query("Trial == Trial.unique()[-1]")
    extra_row_cnt = last_trial.shape[0] - first_trial.shape[0]
    df = df[:-extra_row_cnt] if extra_row_cnt > 0 else df
    df.loc[:, 'Trial'] = df.loc[:, 'Trial'].astype(int)
    # create common time_trial 
    n_trial_pts = len(df.query("Animal == Animal[0] and Trial == Trial[0]"))
    time_trial = np.linspace(trial_start, 
                             trial_start + cs_dur + trace_dur + us_dur + iti_dur,
                             n_trial_pts)
    df['time_trial'] = np.tile(np.tile(time_trial, n_trials), n_subjects)
    # normalize data
    if normalize:
        if drr:
            return JRfp_dat.trial_drr(df, yvar=yvar)
        else:
            return fp_dat.trial_normalize(df, yvar=yvar)
    else:
        return df


################################################################################
