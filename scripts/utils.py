import pandas as pd
import numpy as np 
import json
import os
from tqdm.auto import tqdm
import datetime as dt
from multiprocessing import Pool
from sktime.forecasting.base import ForecastingHorizon
from pandas import datetime
import warnings
from math import sqrt
from pandas import read_csv
from pandas import datetime
import matplotlib.pyplot as plt


def get_returns(df, timedelta: int = 1):
    df["returns"] = ((df["wClose"] / df["wClose"].shift(timedelta)) - 1)
    return df.dropna(subset=["returns"])


def get_train_test(name):
    df_full = pd.read_json(f'data/Resample/{name}.json')
    df_temp = get_returns(df_full, timedelta=6).copy()
    df_temp['mean_returns'] = df_temp['returns'].expanding().mean()
    cutoff_ts = df_temp.index[-1] - pd.Timedelta(365, "days")
    train = df_temp[df_temp.index < cutoff_ts].copy()
    test = df_temp[df_temp.index >= cutoff_ts].copy()
    return train, test


def read_and_concat(path_to_json_files: str):
    """
    This function reads in, concatenates      
    """
    json_files = os.listdir(path_to_json_files)
    df_list = []
    
    
    for i in tqdm(json_files):
        
        if i != ".ipynb_checkpoints":
            name = i.replace('.json', '')
            df = pd.read_json(f'{path_to_json_files}/{i}')

            #df.insert(0, 'open_time', df.index)
            df.insert(1, 'crypto', [name] * df.shape[0])
            df_list.append(df)

        concat_df = pd.concat(df_list, axis = 0, ignore_index = True)
    
    return concat_df

    
def get_train_test_darts(df_full):
    df_full.set_index("open_time_", inplace = True)
    cutoff_ts = df_full.index[-1] - pd.Timedelta(365, "days")
    train = df_full[df_full.index < cutoff_ts].copy()
    test = df_full[df_full.index >= cutoff_ts].copy()
    return train, test


def get_train_test_(name):
    df_full = pd.read_json(f'data/Resample/{name}.json')
    df_full.set_index("open_time", inplace = True)
    cutoff_ts = df_full.index[-1] - pd.Timedelta(365, "days")
    train = df_full[df_full.index < cutoff_ts].copy()
    test = df_full[df_full.index >= cutoff_ts].copy()
    return train, test