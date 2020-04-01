# wrapper for JCapper Race files - aka Past Performance (.jcp)

import os
from pandas import concat, DataFrame, read_csv, to_datetime, Series, MultiIndex
from datetime import timedelta, datetime
from horse.betsim import data
from s3fs.core import S3FileSystem
from horse.betsim.math import compute_probs_from_odds
from scipy.stats import entropy
from warnings import warn


class PastPerformance:
    """ PastPerformance wraps the JCapper Race files aka past performance (.jcp) files
        Attributes:
        df: DataFrame of processed Past Performance data (use for calculations)
        dfraw: DataFrame of raw data
    """
    def __init__(self, verbose=False):
        self.s3 = S3FileSystem(anon=False)
        self.dfraw = DataFrame()
        self.df = DataFrame()
        self.dfpp = DataFrame()
        self.dfwk = DataFrame()
        # now getting track detail exclusively from git file (horse/betsim/data/track_detail.csv) instead of relative path from where data is being loaded
        track_detail = os.path.join(data.__path__._path[0], 'track_detail.csv')
        dftrack = read_csv(track_detail)
        self.map_track_jcp_to_x8 = dftrack.set_index('jcp_track_sym')['x8_track_sym'].to_dict()
        self.map_track_x8_to_jcp = dftrack.set_index('x8_track_sym')['jcp_track_sym'].to_dict()
        self.map_track_x8_to_itsp = dftrack.set_index('x8_track_sym')['itsp_track_sym'].to_dict()
        self.map_track_chart_to_x8 = dftrack.set_index('chart_file_sym')['x8_track_sym'].to_dict()
        self.verbose = verbose

    def load(self, datelist, tracklist=[], path=''):
        """
        load all JCapper Race files (aka past performance files) for the specified dates
        :param datelist: list of dates
        :param tracklist: list of x8 track symbols e.g. DMR, etc
        :param path: str, if given, load data from given directory, otherwise, load data directly from s3
        """

        schema_filepath = os.path.join(os.path.dirname(__file__), 'schema_pastperformance.csv')
        columnDict = read_csv(schema_filepath)['field_name'].to_dict()

        # convert tracklist symbols to jcp track symbols
        tracklist = Series(tracklist).map(self.map_track_x8_to_jcp)
        if tracklist.isnull().any():
            raise ValueError('tracklist must be list of x8 track symbols in track_detail: \n%s' % tracklist)

        # raw data
        self.dfraw = DataFrame()

        if not path:
            # load each date and concat to the master raw df
            for d in datelist:
                # load the DataFrame for this date, e.g. DMR0831F.TXT for 2017-08-31
                year = d.strftime('%Y')
                month = d.strftime('%m')
                day = d.strftime('%d')
                # skip Christmas, no jcapper file
                if month == '12' and day in ['24', '25']:
                    continue
                key = 'x8-bucket/jcapper/%s/%s/%s/' % (year, month, day)
                s3_files = self.s3.ls(key)  # list of all files in a given direcetory - in this case, all files for a single day
                # filter for .jcp files, in this case for non Chart Files
                s3_files = [os.path.basename(fp) for fp in s3_files if fp[-5] != 'F']
                # filter tracks
                if len(tracklist) > 0:
                    s3_files = [fp for fp in s3_files if fp[:3] in list(tracklist)]
                idx_s3files = Series([n[:3] for n in s3_files]).drop_duplicates().index
                if self.verbose:
                    print('pp.load(%s) loading %s race cards..' % (d.strftime('%Y-%m-%d'), len(idx_s3files)))
                # load all past performance files for given date, track is no longer a condition
                for i in idx_s3files:
                    fp = os.path.join(key, s3_files[i])
                    if fp[-3:] == 'jcp':
                        df = read_csv(self.s3.open(fp, mode='rb'), header=None, encoding='ISO-8859-1')
                    else:
                        df = read_csv(self.s3.open(fp, mode='rb'), header=None, compression='zip', encoding='ISO-8859-1')

                    # concat in the master df
                    self.dfraw = concat([self.dfraw, df])
        else:
            # load each date and concat to the master raw df
            for d in datelist:
                # load the DataFrame for this date, e.g. DMR0831F.TXT for 2017-08-31
                year = d.strftime('%Y')
                month = d.strftime('%m')
                day = d.strftime('%d')
                # skip Christmas, no jcapper file
                if month == '12' and day in ['24', '25']:
                    continue
                path_day = os.path.join(path, 'jcapper', year, month, day)
                files = os.listdir(path_day)  # list of all files in a given direcetory - in this case, all files for a single day
                # filter tracks
                if tracklist.any():
                    files = [fp for fp in files if os.path.basename(fp)[:3] in list(tracklist)]
                if self.verbose:
                    print('pp.load(%s) loading %s race cards..' % (d.strftime('%Y-%m-%d'), len(files)))
                # load all past performance files for given date, track is no longer a condition
                for fp in files:
                    # filter for .jcp files, in this case for non Chart Files
                    if fp[-5] != 'F':
                        fp = os.path.join(path_day, fp)
                        if fp[-3:] == 'jcp':
                            df = read_csv(fp, header=None, encoding='ISO-8859-1')
                        else:
                            df = read_csv(fp, header=None, compression='zip', encoding='ISO-8859-1')

                        # concat in the master df
                        self.dfraw = concat([self.dfraw, df])

        try:
            # copy a subset of columns and replace the header names (numbers to text)
            cols = list(columnDict.keys())
            self.df = self.dfraw[cols].copy()
        except KeyError:
            raise Exception('No files available for given datelist and tracklist.')

        # column names
        self.df.rename(columns=columnDict, inplace=True)

        # normalize track sym and make race_id
        self.df['x8_track_sym'] = self.df['jcp_track_sym'].map(self.map_track_jcp_to_x8)
        # adding itsp_track_sym here so that we can filter for bettable tracks in daily races
        self.df['itsp_track_sym'] = self.df['x8_track_sym'].map(self.map_track_x8_to_itsp)

        # drop rows where we are missing jcp symbol mapping in track detail if any
        x8_isnull = self.df['x8_track_sym'].isnull()
        if x8_isnull.any():
            missing_jcp_symbols = self.df[x8_isnull]['jcp_track_sym'].unique()
            warn('pp.load() track_detail.csv is missing jcp symbols: %s\nDropping all rows with missing symbols' % missing_jcp_symbols)
            self.df = self.df[~x8_isnull]
            print('pp.load() dropping %s rows' % x8_isnull.sum())

        # convert dates and validate
        self.df['race_time_flag'] = self.df['race_time'].isnull()  # flag bad race_time values (sometimes is null)
        self.df['race_time'] = to_datetime(self.df['date'].astype(str) + self.df['race_time'].fillna(1000.0).astype(int).astype(str), format='%Y%m%d%H%M')
        self.df['race_time_utc'] = self.df['race_time'].map(lambda x: x + timedelta(hours=8))
        self.df['race_time_toronto'] = self.df['race_time'].map(lambda x: x + timedelta(hours=3))
        self.df['date'] = to_datetime(self.df['date'], format='%Y%m%d')
        self.df['date_str'] = self.df['date'].dt.strftime('%Y%m%d')
        self._birthdate_columns()

        # clean nans in wk date cols
        fields_wk_date = [c for c in self.df.columns if c.startswith('wk_date')]
        self.df['wk_date_1'].fillna(self.df['date_str'], inplace=True)
        self.df[fields_wk_date] = self.df[fields_wk_date].fillna(method='ffill', axis=1)
        self.df[fields_wk_date] = self.df[fields_wk_date].applymap(lambda x: to_datetime(str(int(x)), format='%Y%m%d'))

        # clean nans in pp date cols
        fields_pp_date = [c for c in self.df.columns if c.startswith('pp_date')]
        self.df['pp_date_0'].fillna(self.df['date_str'], inplace=True)
        self.df[fields_pp_date] = self.df[fields_pp_date].fillna(method='ffill', axis=1)
        self.df[fields_pp_date] = self.df[fields_pp_date].applymap(lambda x: to_datetime(str(int(x)), format='%Y%m%d'))

        self.df['race_id'] = self.df['x8_track_sym'] + '_' + self.df['date_str'] + '_' + self.df['race_race_num'].astype(str)
        self.df['runner_program_number'] = self.df['runner_program_number'].map(str)
        self.df['betting_interest'] = self.df['runner_program_number'].str.strip('A')
        self.df['coupled'] = self.df['runner_program_number'].str.count('A').astype(bool)
        self.df['coupled_race'] = self.df.groupby('race_id')['coupled'].transform('any')
        self.df['runner_id'] = self.df['race_id'] + '_' + self.df['runner_program_number']

        # additional time index data and day of week for seasonality
        self.df['month'] = self.df['date'].map(lambda x: x.month)
        self.df['weekday'] = self.df['date'].map(lambda x: x.strftime('%A'))
        self.df['year'] = self.df['date'].map(lambda x: x.year)
        self.df['weeknum'] = self.df['date'].map(lambda x: x.strftime('%w'))
        # normalize horse name
        self.df['x8name'] = self.df['name'].map(self._normalize_name)
        self.df['x8country'] = self.df['name'].map(self._country_from_name)

        # convert pp_track and wk_track columns to x8 symbol
        fields_pp_track = [c for c in self.df.columns if c.startswith('pp_track_')]
        self.df[fields_pp_track] = self.df[fields_pp_track].applymap(lambda x: self.map_track_chart_to_x8.get(x))
        fields_wk_track = [c for c in self.df.columns if c.startswith('wk_track_')]
        self.df[fields_wk_track] = self.df[fields_wk_track].applymap(lambda x: self.map_track_chart_to_x8.get(x))

        # make dataframes for historical pp columns and wk columns that are multiindexed by date
        self._index_pp_columns()
        self._index_wk_columns()

        # validate df
        self._validate(datelist)

    def mtp(self):
        """
        Simple method for getting the minutes to post for every race in the df
        For now, it will return a series, with mtp (timedelta) at the current time of being called.
        So if the user is making column, it is their responsibility to label correctly
        eg - pp.df['mtp_1219'] = pp.mtp()
        :return: Series of timedelta objects
        """
        return self.df['race_time_utc'] - datetime.utcnow()

    def _index_pp_columns(self):
        # creates a multiindex for pp groups
        # uses historical dates for pp columns (0-9) as index
        cols_pp = self.df.columns[self.df.columns.str.contains('pp_')]
        pp_groups = list(Series(["".join(c.split("_")[:-1]) for c in cols_pp.values]).unique())
        pp_ordinal = list(Series([int("".join(c.split("_")[-1])) for c in cols_pp.values]).unique())
        idx_pp_cols = MultiIndex.from_product([pp_groups,pp_ordinal], names=['pp_group', 'pp_ordinal'])
        idx_pp_rows = ['date', 'x8_track_sym', 'race_race_num','runner_program_number', 'x8name']
        self.dfpp = self.df.set_index(idx_pp_rows)[cols_pp]
        self.dfpp.columns = idx_pp_cols

    def _index_wk_columns(self):
        # creates a multiindex for wk groups
        # uses historical dates for wk columns (1-12) as index
        cols_wk = self.df.columns[self.df.columns.str.contains('wk_')]
        wk_groups = list(Series(["".join(c.split("_")[:-1]) for c in cols_wk.values]).unique())
        wk_ordinal = list(Series([int("".join(c.split("_")[-1])) for c in cols_wk.values]).unique())
        idx_wk_cols = MultiIndex.from_product([wk_groups,wk_ordinal], names=['wk_group', 'wk_ordinal'])
        idx_wk_rows = ['date', 'x8_track_sym', 'race_race_num','runner_program_number', 'x8name']
        self.dfwk = self.df.set_index(idx_wk_rows)[cols_wk]
        self.dfwk.columns = idx_wk_cols

    def _normalize_name(self, name):
        # this used to be called make_canonical_name in main, and is unchanged
        canonical_name = name.upper().replace("'", "").strip()
        ending = canonical_name.find("(")
        if ending != -1:
            canonical_name = canonical_name[0:ending].strip().replace(" ", "")
        return canonical_name.replace(" ", "")

    def _country_from_name(self, name):
        '''IF name has "(XXX)" we  use Ireland e.g.'''
        split_name = name.split("(")
        if len(split_name) > 1:
            return split_name[1].replace(")", "")
        else:
            return "USA"

    def _birthdate_columns(self):
        """
        runner birth date formatting
        runner_birthdate_act raw format: '%-m/%-d/%Y' i.e. 4/15/2015
        runner_birthdate_year raw format: int i.e. 15
        """

        # flag null birthdate_act
        self.df['runner_birthdate_act_flag'] = self.df['runner_birthdate_act'].isnull()

        # Jan 1st of year horse was born (runner_birthdate_act raw format)
        self.df['runner_birth_year'] = (self.df['runner_birthdate_year'] + 2000).astype(str)  # 2014
        self.df['runner_birthdate_year'] = '1/1/' + self.df['runner_birth_year']

        # fillna with ephemeral runner_birthdate_year
        self.df['runner_birthdate_act'] = self.df['runner_birthdate_act'].fillna(self.df['runner_birthdate_year'])

        # month and day as string
        self.df['runner_birth_month'] = self.df['runner_birthdate_act'].map(lambda x: x.split('/')[0])
        self.df['runner_birth_day'] = self.df['runner_birthdate_act'].map(lambda x: x.split('/')[1])

        # runner_birthdate_act to datetime
        self.df['runner_birthdate_act'] = self.df['runner_birth_year'] + self.df['runner_birth_month'] + self.df['runner_birth_day']
        self.df['runner_birthdate_act'] = to_datetime(self.df['runner_birthdate_act'], format='%Y%m%d')

        # re-format runner_birthdate_year
        self.df['runner_birthdate_year'] = to_datetime(self.df['runner_birth_year'], format='%Y')

    def add_computed_columns(self):
        # pp column selectors
        cols_pos_finish = self.df.columns[self.df.columns.str.contains('pp_call_finish_pos')]
        cols_lengths_finish = self.df.columns[self.df.columns.str.contains('pp_finish_call_btnLengthsLdr')]
        cols_raceType = self.df.columns[self.df.columns.str.contains('pp_raceType')]
        cols_lp_fig = self.df.columns[self.df.columns.str.contains('pp_pace_Late_')]
        cols_speed_DRF = self.df.columns[self.df.columns.str.contains('pp_speed_DRF')]
        cols_speed_HDW = self.df.columns[self.df.columns.str.contains('pp_speed_HDW')]

        # drop race from df if the whole race is missing morning line odds
        self.df['runner_morning_line_odds'] = self.df['runner_morning_line_odds'].fillna(0)
        empty_race = self.df.groupby('race_id')['runner_morning_line_odds'].transform(sum)
        if (empty_race == 0).any():
            print('pp.add_computed_columns(): dropping races because missing morning line odds: %s' % self.df[empty_race==0].race_id.unique())
            self.df = self.df[empty_race != 0]

        # morning line
        self.df['prob_morning_line'] = self.df.groupby('race_id')['runner_morning_line_odds'].transform(compute_probs_from_odds)
        self.df['entropy_morning_line'] = self.df.groupby('race_id')['prob_morning_line'].transform(lambda x: entropy(x, base=len(x)))
        self.df['rank_morning_line'] = self.df.groupby('race_id')['prob_morning_line'].rank(ascending=False)

        # jockey and trainer wins
        self.df['rank_jockey_wins'] = self.df.groupby('race_id')['runner_jockey_wins_meet'].transform(lambda x: x.rank(ascending=False, method='dense')).astype(int)
        self.df['rank_trainer_wins'] = self.df.groupby('race_id')['runner_trainer_wins_meet'].transform(lambda x: x.rank(ascending=False, method='dense')).astype(int)

        # matrix signals - e.g. pp_win is a matrix of 0/1's where 1 means the horse won that pp and 0 means they did not win
        # these are local variables that are used to generate factors
        # pp signifies only 10 race sample
        pp_win = (self.df[cols_pos_finish] < 1.1).astype(int)
        pp_place = (self.df[cols_pos_finish] < 2.1).astype(int)
        pp_show = (self.df[cols_pos_finish] < 3.1).astype(int)
        pp_finish_close = (self.df[cols_lengths_finish] < 2.75).astype(int)
        pp_G1 = (self.df[cols_raceType] == 'G1').astype(int)
        pp_G2 = (self.df[cols_raceType] == 'G2').astype(int)
        pp_lp_fig_100 = (self.df[cols_lp_fig] >= 100).astype(int)

        # intersection of any of the above conditions
        self.df['x8count_wins'] = pp_win.sum(axis=1)
        self.df['x8count_wins_G1'] = (pp_win.values * pp_G1.values).sum(axis=1)
        self.df['x8count_wins_G2'] = (pp_win.values * pp_G2.values).sum(axis=1)
        self.df['x8count_places_G1'] = (pp_place.values * pp_G1.values).sum(axis=1)
        self.df['x8count_places_G2'] = (pp_place.values * pp_G1.values).sum(axis=1)
        self.df['x8count_shows_G1'] = (pp_show.values * pp_G1.values).sum(axis=1)
        self.df['x8count_shows_G2'] = (pp_show.values * pp_G1.values).sum(axis=1)
        self.df['x8count_close_wins_G2'] = (pp_finish_close.values * pp_G2.values).sum(axis=1)
        self.df['x8count_lp_fig_100'] = (pp_win.values * pp_lp_fig_100.values).sum(axis=1)
        #self.df['parse'] = self.df.race_conditions_jcp.str.find('NW2     L') == 0

        # max/min
        self.df['x8max_speed_HDW'] = self.df[cols_speed_HDW].max(axis=1).fillna(0.0)
        self.df['x8max_speed_DRF'] = self.df[cols_speed_DRF].max(axis=1).fillna(0.0)
        self.df['x8min_speed_HDW'] = self.df[cols_speed_HDW].min(axis=1).fillna(0.0)
        self.df['x8min_speed_DRF'] = self.df[cols_speed_DRF].min(axis=1).fillna(0.0)

        # Flag foreign runners
        self.df['is_foreign'] = self.df['name'].map(lambda x: int(len(x.split("(")) > 1))

        # num horses beaten
        self.df['x8btn_horses_0'] = self.df['pp_fieldsize_0'].fillna(0) - self.df['pp_call_finish_pos_0'].fillna(0)
        self.df['x8lost_horses_0'] = self.df['pp_call_finish_pos_0'].fillna(0) - 1
        self.df['x8btn_horses_1'] = self.df['pp_fieldsize_1'].fillna(0) - self.df['pp_call_finish_pos_1'].fillna(0)
        self.df['x8lost_horses_1'] = self.df['pp_call_finish_pos_1'].fillna(0) - 1
        self.df['x8btn_horses_2'] = self.df['pp_fieldsize_2'].fillna(0) - self.df['pp_call_finish_pos_2'].fillna(0)
        self.df['x8lost_horses_2'] = self.df['pp_call_finish_pos_2'].fillna(0) - 1

        # Indicators of data quality at runner level
        self.df['x8is_runner_full_speed_DRF'] = (self.df.x8min_speed_DRF > 0.0).map(int)

        # Indicators of data quality race level
        self.df['x8is_race_full_speed_DRF'] = (self.df.groupby('race_id')['x8is_runner_full_speed_DRF']
                                                      .transform(lambda x: int(x.min() > 0)))

        # Indicators of number of horses
        self.df['x8_num_starters'] = self.df.groupby('race_id')['x8name'].transform(lambda x: len(x))

        # days since birth date and rank
        self.df['age_days_act'] = (self.df['date'] - self.df['runner_birthdate_act']).dt.days
        self.df['rank_age_days_act'] = self.df.groupby('race_id')['age_days_act'].rank(ascending=False)
        # days since Jan 1st of birth year
        self.df['age_days_year'] = (self.df['date'] - self.df['runner_birthdate_year']).dt.days
        # diff between age_act and age_year
        self.df['age_diff_days'] = self.df['age_days_year'] - self.df['age_days_act']
        # days since first race using pp
        self.df['days_since_first_race'] = (self.df['date'] - self.df['pp_date_9']).dt.days
        # days since first and last workout using pp
        self.df['days_since_first_work'] = (self.df['date'] - self.df['wk_date_12']).dt.days
        self.df['days_since_last_work'] = (self.df['date'] - self.df['wk_date_1']).dt.days
        # first time starter
        self.df['is_first_time'] = self.df['days_since_first_race'].map(lambda x: int(x == 0))

    def _validate(self, datelist):
        """check that the resulting DataFrame values are as expected"""
        if self.df['race_id'].isnull().any():
            raise Exception('betsim.wrap.PastPerformance Null race_id values exist, groupby(race_id) will break, fix')

        wrong_dates = set(self.df.date.dt.date) - set([d.date() for d in datelist])
        if wrong_dates:
            raise Exception('s3 is housing .jcp files with wrong date in target date folders.. \n'
                            'wrong_dates = %s' % wrong_dates)

        duplicate_runners = self.df['runner_id'].duplicated()
        if duplicate_runners.any():
            raise Exception('duplicate runners in df. check s3 for duplicated files. \nrunners: %s'
                            % self.df[duplicate_runners].runner_id)
