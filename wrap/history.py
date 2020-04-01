
from pandas import DataFrame, merge, concat, read_pickle, isnull
from time import time
import glob
import requests
from horse.betsim.math import mean_best_N_of_K, dmetric_L1_weighted
# file paths for pp.df
files_ww_win = glob.glob('/animal_crackers/waw/*.p')

# file paths for pp.df
files_pp_df = glob.glob('/animal_crackers/pp/*.p')

# file paths for jcp.df         
files_jcp_df = glob.glob('/animal_crackers/jcp/jcp.df.*.p')

# file paths for jcp.df_payout
files_jcp_dfp = glob.glob('/animal_crackers/jcp/jcp.df_payout*.p')

# file paths for jcp.df_result_matrix
files_jcp_dfr = glob.glob('/animal_crackers/jcp/jcp.df_result_matrix*.p')

x8race_class_cols = ['jcp_track_sym', 'race_race_type', 'race_distance', 'race_surface', 'race_age_sex_restriction']

map_bet_type = {'WN': 'WN',
                'PL': 'PL',
                'SH': 'SH',
                'E': 'EX',
                'F': 'EX',
                'T': 'TR',
                'S': 'SU',
                'D': 'DB',
                '3': 'P3',
                '4': 'P4',
                '5': 'P5',
                '6': 'P6'}


def add_speed_cols(df):
    factors_speed_HDW = [c for c in df.columns if c.find('speed_HDW') > -1]
    factors_speed_DRF = [c for c in df.columns if c.find('speed_DRF') > -1]


    df['x8speed_HDW_2of3_mean'] = df.loc[:, ['pp_speed_HDW_0', 'pp_speed_HDW_1', 'pp_speed_HDW_2']].apply(lambda row: mean_best_N_of_K(row, n=2, k=3), axis=1)
    df['x8_speed_HDW_best_mean_2of3_lag_1'] = df.loc[:, ['pp_speed_HDW_1', 'pp_speed_HDW_2', 'pp_speed_HDW_3']].apply(lambda row:mean_best_N_of_K(row,2,3),axis=1)
    df['x8_speed_HDW_best_mean_2of3_lag_2'] = df.loc[:, ['pp_speed_HDW_2', 'pp_speed_HDW_3', 'pp_speed_HDW_4']].apply(lambda row:mean_best_N_of_K(row,2,3),axis=1)
    df['x8speed_HDW_2of3_rank'] = df.groupby('race_id')['x8speed_HDW_2of3_mean'].transform(lambda x: x.rank(ascending=False))
    df['x8speed_HDW_2of3_norm_par'] = df['x8speed_HDW_2of3_mean'] / df.race_speed_HDW_par_class_level

    # DRF RUNNER
    df['x8speed_DRF_2of3_mean'] = df.loc[:, ['pp_speed_DRF_0', 'pp_speed_DRF_1', 'pp_speed_DRF_2']].apply(lambda row: mean_best_N_of_K(row, n=2, k=3), axis=1)
    df['x8speed_DRF_2of3_rank'] = df.groupby('race_id')['x8speed_DRF_2of3_mean'].transform(lambda x:x.rank(ascending=False))
    df['x8_is_secondtimestarter'] = df.runner_horse_lifetime_starts.map(lambda x: int(x == 1))
    df['x8_is_firsttimestarter'] = df.runner_horse_lifetime_starts.map(lambda x: int(x == 0))
    df['median_speed_HDW'] = df[factors_speed_HDW].median(axis=1)
    df['median_speed_DRF'] = df[factors_speed_DRF].median(axis=1)

    # speed sum
    df['x8diffspeed_HDWPSRRating__HDWPar'] = df['runner_HDWPSRRating'] - df['race_speed_HDW_par_class_level']
    df['x8diffspeed_x8speed_HDW_2of3_mean__HDWPar'] = df['x8speed_HDW_2of3_mean'] - df['race_speed_HDW_par_class_level']
    df['x8diffspeed_x8max_speed__HDWPar'] = df['x8max_speed_HDW'] - df['race_speed_HDW_par_class_level']
    df['x8diffspeed_runner_speed_HDWBest_turf__HDWPar'] = df['runner_speed_HDWBest_turf'] - df['race_speed_HDW_par_class_level']
    df['x8diffspeed_runner_runner_speed_HDWBest_distance__HDWPar'] = df['runner_speed_HDWBest_distance'] - df['race_speed_HDW_par_class_level']
    speed_sum_cols = [c for c in df.columns if c.startswith('x8diffspeed')]
    df['x8speed_sum_par'] = df[speed_sum_cols].applymap(lambda x: int(x > 0)).sum(axis=1) + df['runner_morning_line_odds']
    return df


class History:
    """
    pickled historical training data sets on EC2_System 
    """
    def __init__(self, verbose=False):
        
        # data
        self.df_pp = DataFrame()
        self.df_jcp = DataFrame()
        self.df_payout = DataFrame()
        self.df_result_matrix = DataFrame()
        self.df_win_odds = DataFrame()

        # pp and jcp merged
        self.df_train = DataFrame()

        self.verbose = verbose

    def load(self):

        if self.verbose:
            print('History.load(): loading historical pickled data..')

        if requests.get("http://ipecho.net/plain?").text != '18.224.240.230':
            raise Exception('Must be using EC2_System to use horse.history object.')

        t = time()
        self.df_win_odds = concat([read_pickle(fp) for fp in files_ww_win], ignore_index=False, sort=False)
        s = round(time() - t)
        print('hist.df_win_odds: loaded %s days of ww.df_win_odds in %s seconds' % (self.df_win_odds['date'].nunique(), s))

        t = time()
        self.df_pp = concat([read_pickle(fp) for fp in files_pp_df], ignore_index=False, sort=False)
        s = round(time() - t)
        print('hist.df_pp: loaded %s days of pp.df in %s seconds' % (self.df_pp['date'].nunique(), s))

        t = time()
        self.df_jcp = concat([read_pickle(fp) for fp in files_jcp_df], ignore_index=False, sort=False)
        s = round(time() - t)
        print('hist.df_jcp: loaded %s days of jcp.df in %s seconds' % (self.df_jcp['date'].nunique(), s))

        t = time()
        self.df_payout = concat([read_pickle(fp) for fp in files_jcp_dfp], ignore_index=False, sort=False)
        s = round(time() - t)
        print('hist.df_payout: loaded jcp.df_payout in %s seconds' % s)

        t = time()
        self.df_result_matrix = concat([read_pickle(fp) for fp in files_jcp_dfr], ignore_index=False, sort=False)
        s = round(time() - t)
        print('hist.df_result_matrix: loaded jcp.df_result_matrix in %s seconds' % s)

        self._add_computed_columns()

        self._make_training_data()

    def _add_computed_columns(self):
        """
        add columns
        """
        if 'x8race_class' not in self.df_pp.columns:
            self.df_pp['x8race_class'] = self.df_pp[x8race_class_cols].apply(lambda row: tuple(row), axis=1)

    def _make_training_data(self):
        print('making training data.. merging hist.df_jcp and hist.df_pp on index.')

        shared_columns = list(set(self.df_pp.columns).intersection(set(self.df_jcp.columns)))
        shared_columns.remove('runner_id')
        self.df_jcp = self.df_jcp.drop(columns=shared_columns)

        # set index
        self.df_pp.set_index('runner_id', inplace=True)
        self.df_jcp.set_index('runner_id', inplace=True)

        self.df_train = merge(self.df_pp, self.df_jcp, left_index=True, right_index=True, how='left')

        self.df_train['diff_num_starters_pre_post'] = self.df_train['num_starters_pre'] - self.df_train['num_starters_post']


def add_sentinels(df):  #df['date']=df['date'].map(lambda x:pd.to_dateteim(x))
    df['is_ofp_0'] = df.official_finish_position.map(lambda x: int(x == 1))
    df['is_ofp_1'] = df.official_finish_position.map(lambda x: int(x == 2))
    df['is_ofp_2'] = df.official_finish_position.map(lambda x: int(x == 3))
    df['is_ofp_3'] = df.official_finish_position.map(lambda x: int(x == 4))
    df['is_otm'] = df.official_finish_position.map(lambda x: int(x > 3))
    df['is_outperform'] = df.underperformance_weighted.map(lambda x: int(x > 0))
    df['is_underperform'] = df.underperformance_weighted.map(lambda x: int(x < 0))
    df['dayofyear'] = df.date.dt.dayofyear
    df['weekofyear'] = df.date.dt.weekofyear
    return df

def add_window_factors(df, attr_accumulate, attr_datetime):
    pass

# to import either on Python2 or Python3
import pandas as pd
from time import time # not needed just for timing
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO


# def zgrep_data(f, string):
#     '''grep multiple items f is filepath, string is what you are filtering for'''
#
#     grep = 'grep' # change to zgrep for gzipped files
#     print('{} for {} from {}'.format(grep,string,f))
#     start_time = time()
#     if string == '':
#         out = subprocess.check_output([grep, string, f])
#         grep_data = StringIO(out)
#         data = pd.read_csv(grep_data, sep=',', header=0)
#
#     else:
#         # read only the first row to get the columns. May need to change depending on
#         # how the data is stored
#         columns = pd.read_csv(f, sep=',', nrows=1, header=None).values.tolist()[0]
#
#         out = subprocess.check_output([grep, string, f])
#         grep_data = StringIO(out)
#
#         data = pd.read_csv(grep_data, sep=',', names=columns, header=None)
#
#     print('{} finished for {} - {} seconds'.format(grep,f,time()-start_time))
#     return data

#iter_csv = pd.read_csv('file.csv', iterator=True, chunksize=1000)
#df = pd.concat([chunk[chunk['field'] > constant] for chunk in iter_csv])
def describe_dataframe(df=DataFrame()):
    """This function generates descriptive stats of a dataframe
    Args:
        df (dataframe): the dataframe to be analyzed
    Returns:
        None

    """
    print("\n\n")
    print("*" * 30)
    print("About the Data")
    print("*" * 30)

    print("Number of rows::", df.shape[0])
    print("Number of columns::", df.shape[1])
    print("\n")

    print("Column Names::", df.columns.values.tolist())
    print("\n")

    print("Column Data Types::\n", df.dtypes)
    print("\n")

    print("Columns with Missing Values::", df.columns[df.isnull().any()].tolist())
    print("\n")

    print("Number of rows with Missing Values::", len(isnull(df).any(1).nonzero()[0].tolist()))
    print("\n")

    print("Sample Indices with missing data::", isnull(df).any(1).nonzero()[0].tolist()[0:5])
    print("\n")

    print("General Stats::")
    print(df.info())
    print("\n")

    print("Summary Stats::")
    print(df.describe())
    print("\n")

    print("Dataframe Sample Rows::")
    print(df.head(5))


def index_entity_by_datetime(df, attr_entity, attr_result, attr_datetime):
    # Updates the performance of an entity to date specified
    # Lets summarize the
    #df = df.sort_values(attr_datetime)
    df = add_sentinels(df)
    df = df.sort_values([attr_entity, attr_datetime])
    df['time_diff'] = df[attr_datetime].diff()
    df.loc[df[attr_entity] != df[attr_entity].shift(), 'time_diff'] = None
    return df


def accumulate_entity_result(df, attr_entity, attr_result, attr_datetime):
    """df = df_train
    attr_entity = 'x8name'
    attr_result = 'earnings"""
    df = index_entity_by_datetime(df, attr_entity, attr_result, attr_datetime)
    df = df.sort_values([attr_entity, attr_datetime])
    df[attr_entity +'_' + 'cumsum_' + attr_result] = df.groupby(attr_entity)[attr_result].cumsum()
    df[attr_entity + '_' + 'cumcount_'+attr_result] = df.groupby(attr_entity)[attr_result].cumcount()
    return df


def pool_estimator(df_train_winners, target_date):
    # return mean/median/min pool size of some sample

    dfs = []
    # there are 12 wager_type and wager_pool columns each
    for i in range(1, 13):
        # select wager_type_1 and wager_pool_1 etc.
        pool_type = 'wager_type_%s' % i
        pool_total = 'wager_pool_%s' % i
        df = df_train_winners[['race_id', pool_type, pool_total]].rename({pool_type: 'wager_type', pool_total: 'wager_pool'}, axis=1)
        df = df.dropna(subset=['wager_type'])
        dfs.append(df)
    # stack all df's on top of eachother
    df_pool_estimator = concat(dfs)
    # map bet type
    df_pool_estimator['wager_type'] = df_pool_estimator['wager_type'].map(map_bet_type)

    return df_pool_estimator
