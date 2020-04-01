from pandas import concat, DataFrame, read_csv, to_datetime, Series, MultiIndex
from datetime import timedelta, datetime
from horse.betsim import data
from s3fs.core import S3FileSystem
from horse.betsim.math import compute_probs_from_odds
from scipy.stats import entropy
from warnings import warn

import os
import pandas as pd
import numpy as np
def gen_names_wk(prefix):
    return [prefix + str(n) for n in np.arange(1,13)]
colnames_days_since_last_race_date = gen_names_wk('wk_days_since_last_race')
cols_fps = gen_names_wk('wk_feet_per_second')
cols_wk_since_lastrace = gen_names_wk('wk_is_since_lastrace_')
cols_numdays_wk = gen_names_wk('wk_numdays_prev')
colnames_days_since_last_race_date = gen_names_wk('wk_days_since_last_race')
cols_wk_diff_dist = gen_names_wk('wk_diff_dist')

cols_numdays_wk = gen_names_wk('wk_numdays_prev')
def avg_days_between_workout_all(df):
    wk_date_columns = [c for c in df.columns if c.startswith('wk_date')]
    return df[wk_date_columns].diff(periods=-1, axis=1).mean(axis=1)


def days_between_wk_date(df):
    wk_date_columns = [c for c in df.columns if c.startswith('wk_date')]
    return df[wk_date_columns].diff(periods=-1, axis=1)
df[cols_numdays_wk]=days_between_wk_date(df)

df[cols_wk_since_lastrace] = df[colnames_days_since_last_race_date].applymap(lambda x:int(x.days>0))

df['wk_avg_days_since'] = df['days_since_last_race']/df['wk_num_workouts_since_last_race']
cols_wk_dist = [c for c in df if c.startswith('wk_dist')]
cols_wk_time = [c for c in df if c.startswith('wk_time')]
cols_wk_date = [c for c in df if c.startswith('wk_date')]
cols_wk_num_at_distance = [c for c in df if c.startswith('wk_num')]
cols_wk_rank = [c for c in df if c.startswith('wk_rank')]
cols_wk_tkcond = [c for c in df if c.startswith('wk_tkcond')]
cols_wk_main = [c for c in df if c.startswith('wk_main')]
cols_wk_desc = [c for c in df if c.startswith('wk_desc')]


class Workouts():
    def __init__(self, df):
        self.dates_wk = df

if __name__ == '__main__':
    # history
    #hist = History()
    #hist.load()

    os.chdir("C:\\Users\\Saleem\projects\\x8313\\horse\\notebooks")
    #datapath = os.path.join("C:\\Users\\Saleem\projects\\x8313\\notebooks", "data")

    dfAQU = pd.read_csv('df_train_AQU.csv.gz', compression='gzip', low_memory=False,
                        parse_dates=['date', 'race_time'], nrows=100)
    wkdatecols = findcols('wk_date', dfAQU)
    ppdatecols = findcols('pp_date', dfAQU)

    ppcols = findcols('pp_', dfAQU)
    wkcols = findcols('wk_', dfAQU)
    commentcols = findcols('comment', dfAQU)
    condcols = findcols('condition', dfAQU)

    dfAQU = pd.read_csv('df_train_AQU.csv.gz', compression='gzip', low_memory=False,
                        parse_dates=['date', 'race_time'] + wkdatecols + ppdatecols)

    [clean_text_column(dfAQU, col) for col in commentcols]

