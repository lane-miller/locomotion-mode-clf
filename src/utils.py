# Imports
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, sosfiltfilt

def load_data(data_dir, col_map):
    """
    Load PAMAP2 .dat files from data_dir into a single concatenated DataFrame.
    
    Parameters
    ----------
    data_dir : str or Path
        Path to directory containing subject*.dat files.
    col_map : dict
        Mapping of column names to their integer indices in the .dat file.
        Example: {'timestamp': 0, 'activity_id': 1, ...}
    
    Returns
    -------
    pd.DataFrame
        Concatenated data from all subjects with a 'subject' column added.
        Activity ID 0 (transient) is excluded.
    """

    data_dir = Path(data_dir)
    dfs = []

    for filepath in sorted(data_dir.glob('subject*.dat')):
        subject_id = filepath.stem
        df = pd.read_csv(filepath, sep=' ', header=None)
        df = df[list(col_map.values())].copy()
        df.columns = list(col_map.keys())
        df = df[df['activity_id'] != 0].reset_index(drop=True)
        df.insert(0, 'subject', subject_id)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def exclude_subjects(df, subjects):
    """
    Remove specified subjects from the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'subject' column.
    subjects : list of str
        Subject IDs to exclude, e.g. ['subject103', 'subject104'].

    Returns
    -------
    pd.DataFrame
        DataFrame with specified subjects removed, index reset.
    """
    return df[~df['subject'].isin(subjects)].reset_index(drop=True)


def filter_activities(df, activities):
    """
    Retain only rows matching specified activity IDs.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with an 'activity_id' column.
    activities : dict or list
        Activity IDs to retain. Accepts a dict (keys used) or a list.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with index reset.
    """
    activity_ids = activities.keys() if isinstance(activities, dict) else activities
    return df[df['activity_id'].isin(activity_ids)].reset_index(drop=True)


def preprocess(df, accel_cols, gyro_cols, fc=15, fs=100, order=4, norm_percentile=95):
    """
    Apply full preprocessing pipeline to raw IMU data.
    
    Steps applied per subject:
        1. Min-max normalization to [-1, 1] at specified percentile
        2. Linear interpolation over NaN values
        3. Zero-phase Butterworth low-pass filter per activity segment
        4. DC removal per activity segment
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw data with 'subject' and 'activity_id' columns.
    accel_cols : list of str
        Accelerometer column names.
    gyro_cols : list of str
        Gyroscope column names.
    fc : float
        Low-pass filter cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int
        Butterworth filter order.
    norm_percentile : float
        Percentile used for normalization bounds (e.g. 95).
    
    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame with same structure as input.
    """
    signal_cols = accel_cols + gyro_cols
    sos = butter(order, fc, fs=fs, btype='low', output='sos')
    preprocessed = []

    for subject_id in sorted(df['subject'].unique()):
        subj = df[df['subject'] == subject_id].copy()

        # Normalize per sensor group
        for cols in [accel_cols, gyro_cols]:
            low = subj[cols].stack().quantile((100 - norm_percentile) / 100)
            high = subj[cols].stack().quantile(norm_percentile / 100)
            denom = high - low
            if denom == 0:
                subj[cols] = 0.0
            else:
                subj[cols] = 2 * (subj[cols].clip(lower=low, upper=high) - low) / denom - 1

        # Interpolate NaNs
        subj[signal_cols] = subj[signal_cols].interpolate(
            method='linear', limit_direction='both'
        )

        # Low-pass filter per activity segment to avoid boundary artifacts
        for activity_id in subj['activity_id'].unique():
            mask = subj['activity_id'] == activity_id
            if mask.sum() <= 15:  # skip segments too short for sosfiltfilt padding
                continue
            subj.loc[mask, signal_cols] = sosfiltfilt(
                sos, subj.loc[mask, signal_cols].values, axis=0
    )

        # DC removal per activity segment
        for activity_id in subj['activity_id'].unique():
            mask = subj['activity_id'] == activity_id
            subj.loc[mask, signal_cols] = (
                subj.loc[mask, signal_cols] - subj.loc[mask, signal_cols].mean()
            )

        preprocessed.append(subj)

    return pd.concat(preprocessed, ignore_index=True)


def train_test_split(df, hold_out):
    """
    Split DataFrame into training and test sets by subject.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'subject' column.
    hold_out : str
        Subject ID to use as the test set, e.g. 'subject101'.

    Returns
    -------
    tuple of pd.DataFrame
        (df_train, df_test)
    """
    df_train = df[df['subject'] != hold_out].reset_index(drop=True)
    df_test = df[df['subject'] == hold_out].reset_index(drop=True)
    return df_train, df_test