# wrapper for Result Files - aka Chart Files (.TXT)

import os
from horse.betsim.data import map
from horse.betsim import data
from s3fs.core import S3FileSystem
import math
from numpy import nan, zeros, fill_diagonal, isneginf, log
from pandas import DataFrame, Series, read_csv, to_datetime, merge, DatetimeIndex, date_range, concat
from horse.betsim.math import compute_probs_from_odds
from scipy.stats import entropy
from warnings import warn
from horse.betsim.data.db.engines import engine_ac
from sqlalchemy import Table, select, MetaData
from datetime import date, datetime
from horse.betsim.math import get_freq_wom
from itertools import product

# dictionary of column names
# unused columns: 36-50 and 104-130
columnDict = {0: 'chart_file_sym', 1: 'date', 2: 'race_number', 3: 'breed_code', 4: 'distance', 5: 'is_about_distance', 6: 'surface_code'}
columnDict.update({7: 'is_off_turf', 8: 'course_type_code', 9: 'race_type_code', 10: 'race_conditions'})
columnDict.update({11: 'race_name', 12: 'sex_restriction_code', 13: 'age_restriction_code', 14: 'is_restricted_to_statebred'})
columnDict.update({15: 'purse', 16: 'max_claiming_price', 17: 'min_claiming_price', 18: 'grade_code', 19: 'division'})
columnDict.update({20: 'num_starters_post', 21: 'track_condition_code', 22: 'weather_code', 23: 'post_time', 24: 'temp_rail_dist'})
columnDict.update({25: 'is_chute_start', 26: 'is_track_sealed'})
columnDict.update({27: 'race_f1_time', 28: 'race_f2_time', 29: 'race_f3_time', 30: 'race_f4_time', 31: 'race_f5_time'})
columnDict.update({32: 'race_winning_time', 33: 'wps_combined_pool', 34: 'run_up_dist'})
columnDict.update({50: 'horse_name', 51: 'age', 52: 'sex_code', 53: 'medication_code', 54: 'equipment_code', 55: 'weight'})
columnDict.update({56: 'claiming_price', 57: 'earnings', 58: 'program_number', 59: 'coupled_type', 60: 'morning_line'})
columnDict.update({61: 'final_tote_odds', 62: 'is_favorite', 63: 'is_non_betting'})
columnDict.update({64: 'payout_win', 65: 'payout_place', 66: 'payout_show'})
columnDict.update({67: 'post_position'})
columnDict.update({68: 'pos_call_start', 69: 'pos_call_1', 70: 'pos_call_2', 71: 'pos_call_3', 72: 'pos_call_4'})
columnDict.update({73: 'pos_call_stretch', 74: 'pos_call_finish', 75: 'official_finish_position'})
columnDict.update({76: 'call_1_lengths_behind', 77: 'call_2_lengths_behind', 78: 'call_3_lengths_behind', 79: 'call_4_lengths_behind'})
columnDict.update({80: 'stretch_call_lengths_behind', 81: 'finish_lengths_behind', 82: 'comment_line'})
columnDict.update({83: 'trainer_first_name', 84: 'trainer_middle_name', 85: 'trainer_last_name'})
columnDict.update({86: 'jockey_first_name', 87: 'jockey_middle_name', 88: 'jockey_last_name'})
columnDict.update({89: 'apprentice_weight_allowance', 90: 'owner_full_name'})
columnDict.update({91: 'is_claimed', 92: 'claiming_trainer_first_name', 93: 'claiming_trainer_middle_name', 94: 'claiming_trainer_last_name', 95: 'claiming_owner_full_name'})
columnDict.update({96: 'is_scratch', 97: 'scratch_code'})
columnDict.update({98: 'last_race_track', 99: 'last_race_date', 100: 'last_race_num', 101: 'last_race_finish'})
columnDict.update({103: 'long_comment'})

# only for columns that require explicit dtype assignment e.g. wager_type_n: sometimes csv col starts with a number and then thinks entire column is int
dtype = {130: str, 136: str, 142: str, 148: str, 154: str, 160: str, 166: str, 172: str, 178: str, 184: str, 190: str, 196: str} # wager_type_n
dtype.update({18: str}) # grade_code
dtype.update({58: str}) # program_number because 1A possible values

map_exotic_wager = {'0': 'Choose6', '1': 'Roulette', '2': 'TwointheMoney', '3': 'Pick3', '4': 'Pick4',
                    '5': 'Pick5', '6': 'Pick6', '6J': 'Pick6Jackpot', '7J': 'Pick7Jackpot', '7': 'Pick7',
                    '8': 'Countdown', '9': 'Pick9', '13': '123Racing', 'A': 'Triactor', 'B': 'SuperTri',
                    'BQ': 'BracketQuinella', 'C': 'Classix', 'D': 'DailyDouble', 'E': 'Exacta', 'F': 'Exacta',
                    'G': 'Perfector', 'H': 'Head2Head', 'HE': 'HalfwayExacta', 'HW': 'HalfwayWin', 'I': 'Pick10',
                    'ID': 'InstantDailyDouble', 'J': 'Exactor', 'T': 'Trifecta', 'S': 'Superfecta', 'S5': 'SuperHighFive',
                    'P': 'JockeyChallenge', 'M': 'ConsolationPick3', 'O': 'Omni', 'Z': 'ConsolationDouble', 'WN': 'Win',
                    'PL': 'Place', 'SH': 'Show'}

map_wager_axis = {'Exacta': 'vertical',
                  'Trifecta': 'vertical',
                  'Superfecta': 'vertical',
                  'DailyDouble': 'horizontal',
                  'Pick3': 'horizontal',
                  'Pick4': 'horizontal',
                  'Pick5': 'horizontal',
                  'Pick6': 'horizontal'}

# this is for normalizing bet type that watchandwager / amtote uses and merging with pnl.
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
                '6': 'P6',
                'J': 'EX',
                'A': 'TR'}

# single leg, multi-runner and multi-race pool symbols
pool_single = ['WN', 'PL', 'SH']
pool_multi_runner = ['EX', 'QU', 'TR', 'SU']
pool_multi_race = ['DB', 'P3', 'P4', 'P5', 'P6']
minus_num_races = {'DB': 1, 'P3': 2, 'P4': 3, 'P5': 4, 'P6': 5}

# add the wager_type_1, etc columns up to _12
wager_list = ['wager_type_', 'wager_unit_', 'wager_numright_', 'wager_payout_', 'wager_winningPGM_', 'wager_pool_']
for num in range(1, 13):
    index_start = 130 + 6 * (num - 1)
    keys = [index_start + x for x in [0, 1, 2, 3, 4, 5]]
    values = ['%s%d' % (x, num) for x in wager_list]
    columnDict.update(dict(zip(keys, values)))


class JCapper:
    """ JCapper wraps the results files from JCapper
        Attributes:
        df: DataFrame of processed Past Performance data (use for calculations)
        dfraw: DataFrame of raw JCapper data
    """
    def __init__(self, verbose=False):
        self.s3 = S3FileSystem(anon=False)
        self.df = DataFrame()
        self.dfraw = DataFrame()
        self.df_scratch = DataFrame()
        self.df_result_matrix = DataFrame()
        self.df_payout = DataFrame()
        # now getting track detail exclusively from git file (horse/betsim/data/track_detail.csv) instead of relative path from where data is being loaded
        track_detail = os.path.join(data.__path__._path[0], 'track_detail.csv')
        self.dftrack = read_csv(track_detail)
        self.map_track_chart_to_x8 = self.dftrack.set_index('chart_file_sym')['x8_track_sym'].to_dict()
        self.map_track_x8_to_chart = self.dftrack.set_index('x8_track_sym')['chart_file_sym'].to_dict()
        self.verbose = verbose

    def load(self, datelist, tracklist=[], path=''):
        """
        load all JCapper Results/Charts files for the specified dates
        :param datelist: list of dates
        :param tracklist: list of x8 track symbols e.g. DMR, etc
        :param path: str, if given, load data from given directory, otherwise, load data directly from s3
        """

        # check that tracklist are x8 track symbols
        not_x8_symbol = set(tracklist) - set(self.dftrack['x8_track_sym'])
        if not_x8_symbol:
            raise ValueError('tracklist must be list of x8 track symbols in track_detail. The following symbols are not'
                             'x8 track symbols or they are not in track_detail. \n%s' % not_x8_symbol)

        # check if any of tracks in tracklist are not available from JCapper
        df_missing = self.dftrack[self.dftrack['x8_track_sym'].isin(tracklist)]
        if df_missing['chart_file_sym'].isnull().any():
            raise ValueError('The following x8 track symbols in tracklist are not available in JCapper Data. '
                             'Check WatchandWager Results.\n%s' % df_missing[df_missing['chart_file_sym'].isnull()]['x8_track_sym'])

        # convert tracklist symbols to chart symbols
        tracklist = Series(tracklist).map(self.map_track_x8_to_chart)

        # raw data
        self.dfraw = DataFrame()

        if not path:
            # load each date and concat to the master raw df
            for d in datelist:
                # load the DataFrame for this date
                year = d.strftime('%Y')
                month = d.strftime('%m')
                day = d.strftime('%d')
                # bug: 2017-12-24 races exist
                # skip Christmas, no jcapper file
                if month == '12' and day in ['24', '25']:
                    continue
                if self.verbose:
                    print('jcapper.load(%s)' % d.strftime('%Y-%m-%d'))
                key = 'x8-bucket/jcapper/%s/%s/%s' % (year, month, day)
                s3_files = self.s3.ls(key)  # list of all files in a given direcetory - in this case, all files for a single day
                # filter: Chart Files have 'F.ZIP' suffix
                s3_files = [fp for fp in s3_files if fp[-5:] == 'F.ZIP']
                # filter tracks
                if not tracklist.empty:
                    s3_files = [fp for fp in s3_files if os.path.basename(fp)[:-9] in list(tracklist)]
                for fp in s3_files:
                    df = read_csv(self.s3.open(fp, mode='rb'), header=None, compression='zip', dtype=dtype)
                    # concat in the master df
                    self.dfraw = concat([self.dfraw, df])
        else:
            # load each date and concat to the master raw df
            for d in datelist:
                # load the DataFrame for this date
                year = d.strftime('%Y')
                month = d.strftime('%m')
                day = d.strftime('%d')
                # skip Christmas, no jcapper file
                if month == '12' and day in ['24', '25']:
                    continue
                if self.verbose:
                    print('jcapper.load(%s)' % d.strftime('%Y-%m-%d'))
                path_day = os.path.join(path, 'jcapper', year, month, day)
                files = os.listdir(path_day)  # list of all files in a given direcetory - in this case, all files for a single day
                # filter: Chart Files have 'F.ZIP' suffix
                files = [fp for fp in files if fp[-5:] == 'F.ZIP']
                # filter tracks
                if not tracklist.empty:
                    files = [fp for fp in files if os.path.basename(fp)[:-9] in list(tracklist)]
                for fp in files:
                    fp = os.path.join(path_day, fp)
                    df = read_csv(fp, header=None, compression='zip', dtype=dtype)
                    # concat in the master df
                    self.dfraw = concat([self.dfraw, df])

        # copy a subset of columns and replace the header names (numbers to text)
        cols = list(columnDict.keys())
        try:
            self.df = self.dfraw[cols].copy()
        except KeyError:
            raise FileNotFoundError('\nThere are no Result Files for tracks: %s dates %s to %s in S3' % (tracklist, datelist[0].date(), datelist[-1].date()))
        self.df.rename(columns=columnDict, inplace=True)

        # add mapped columns
        self._add_mapped_columns()

        # convert dates and validate
        self.df['date'] = to_datetime(self.df['date'], format='%m/%d/%Y')

        # additional time index data and day of week for seasonality
        self.df['month'] = self.df['date'].map(lambda x: x.month)
        self.df['weekday'] = self.df['date'].map(lambda x: x.strftime('%A'))
        self.df['year'] = self.df['date'].map(lambda x: x.year)
        self.df['weeknum'] = self.df['date'].map(lambda x: x.strftime('%w'))

        # convert position and finish data to integers
        # normalize horse name
        self.df['x8name'] = self.df['horse_name'].map(self._normalize_name)
        self.df['x8country'] = self.df['horse_name'].map(self._country_from_name)

        # normalize track sym and make race_id
        self.df['x8_track_sym'] = self.df['chart_file_sym'].map(self.map_track_chart_to_x8)
        self.df['race_id'] = self.df['x8_track_sym'] + '_' + self.df['date'].dt.strftime('%Y%m%d') + '_' + self.df['race_number'].astype(str)
        self.df['runner_id'] = self.df['race_id'] + '_' + self.df['program_number'].map(str)

        # drop rows where we are missing chart symbol mapping in track detail if any
        x8_isnull = self.df['x8_track_sym'].isnull()
        if x8_isnull.any():
            missing_chart_symbols = self.df[x8_isnull]['chart_file_sym'].unique()
            warn('jcp.load() track_detail.csv is missing jcp symbols: %s\nDropping all rows with missing symbols' % missing_chart_symbols)
            self.df = self.df[~x8_isnull]
            print('jcp.load() dropping %s rows' % x8_isnull.sum())

        # assign num_starters_pre  before dropping scratches
        self.df['num_starters_pre'] = self.df.groupby('race_id')['num_starters_post'].transform('count')

        # filter rows where the horse scratched
        n = len(self.df)
        self.df_scratch = self.df[self.df['is_scratch'] > 0]
        self.df = self.df[self.df['is_scratch'] != 1]

        if self.verbose and (n != len(self.df)):
            print('JCapper.load(): filtering scratched horses reduced from %d to %d' % (n, len(self.df)))

        # drop race from df if the whole race is missing morning line odds
        cancelled_races = self.df[self.df['official_finish_position'].isnull()]['race_id'].unique()
        if cancelled_races:
            print('pp dropping cancelled races: %s' % cancelled_races)
            self.df = self.df[~self.df['race_id'].isin(cancelled_races)]

        # dtype conversions that have to happen after NA vaues are filtered by scratch filter (just above this line)
        self.df['post_position'] = self.df['post_position'].map(int)
        self.df['official_finish_position'] = self.df['official_finish_position'].map(int)

        # compressed dataframe for payouts by wagertype only
        # one row per race and pool - effectively a pool_id
        self.df_payout = self._make_df_payout(self.df)

        # Matrix of diagonal 1s by race_id for vector computations
        self.df_result_matrix = self.get_result_matrix(self.df)

        # validate the df removed 20171128
        self._validate(datelist)

    def _make_df_payout(self, df):
        dfs = []
        str_condition = 'wager_type_|wager_unit_|wager_numright_|wager_payout_|wager_winningPGM_|wager_pool_'
        wager_cols = df.columns[df.columns.str.contains(str_condition)]

        df_wagers = df.set_index('race_id')[wager_cols].copy()

        for i in range(1, 13):
            col_bynum = [name + str(i) for name in wager_list]
            df_bynum = df_wagers[col_bynum].drop_duplicates()
            df_bynum.reset_index(inplace=True)
            df_bynum.columns = ['race_id', 'wager_type', 'wager_unit', 'wager_numright', 'wager_payout', 'wager_winningPGM', 'wager_pool']
            df_bynum['wager_number'] = str(i)
            dfs.append(df_bynum)

        df_payout = concat(dfs, ignore_index=False)
        df_payout.dropna(how='all', inplace=True)

        # this block of code is for adding the win/place/show pools to df_payout
        # the win/place/show pools are recorded in a different format and are being transformed to df_payout format
        wps = self.df[['race_id', 'program_number', 'payout_win', 'payout_place', 'payout_show', 'wps_combined_pool']].dropna(thresh=4)
        wps = wps.melt(id_vars=['race_id','program_number','wps_combined_pool'], value_vars=['payout_win', 'payout_place', 'payout_show'], var_name='wager_type', value_name='wager_payout')
        wps = wps[wps['wager_payout'].notnull()]
        wps['wager_type'] = wps['wager_type'].map({'payout_win': 'WN', 'payout_place': 'PL', 'payout_show': 'SH'})
        wps['wager_numright'] = nan
        wps['wager_unit'] = 2.0
        wps.rename(columns={'program_number': 'wager_winningPGM', 'wps_combined_pool': 'wager_pool'}, inplace=True)
        df_payout = concat([df_payout, wps], sort=True)
        df_payout = df_payout.reset_index(drop=True)

        # normalize columns
        df_payout['wager_name'] = df_payout['wager_type'].map(map_exotic_wager)
        df_payout['wager_axis'] = df_payout['wager_name'].map(lambda x: map_wager_axis.get(x))
        df_payout['bet_type'] = df_payout['wager_type'].map(map_bet_type)
        df_payout['payout_norm'] = df_payout['wager_payout'] / df_payout['wager_unit']
        df_payout['wager_winningPGM'] = df_payout['wager_winningPGM'].map(lambda x: str(x).replace('-', ','))
        df_payout.rename(columns={'wager_winningPGM': 'winning_pgm', 'wager_pool': 'pool_total'}, inplace=True)

        # pool signal columns
        df_payout['is_single_leg'] = df_payout['bet_type'].map(lambda x: True if x in pool_single else False)
        df_payout['is_multi_runner'] = df_payout['bet_type'].map(lambda x: True if x in pool_multi_runner else False)
        df_payout['is_multi_race'] = df_payout['bet_type'].map(lambda x: True if x in pool_multi_race else False)

        # for multi race pools, change race_id from ending race_id to starting race_id so it can merge w/ df_bets
        df_payout['race_id'] = df_payout.apply(lambda x: '_'.join(
            x['race_id'].split('_')[:2] + [str(int(x['race_id'].split('_')[-1]) - minus_num_races[x['bet_type']])]) if x[
            'is_multi_race'] else x['race_id'], axis=1)

        # total amount bet on winning combination
        # TODO wrong for WPS and other
        df_payout['correct_money'] = df_payout['pool_total'] / df_payout['payout_norm']

        # for multi-race bets where a 1 or more runners scratches in any of the races
        def parse_slash(df_slash):
            dfs = []
            for i in df_slash.index:
                s = df_slash.loc[i].winning_pgm # '5,2/10,5/7/8/10/12,3/11'

                list_of_lists = [r.split('/') for r in s.split(',')]
                prod_tuples = list(product(*list_of_lists))
                output_strings = [','.join(t) for t in prod_tuples]

                df = DataFrame(df_slash.loc[i].to_dict(),index=[i])
                df = concat([df] * len(output_strings))
                df['winning_pgm'] = output_strings

                dfs.append(df)
 
            return concat(dfs)

        df_payout['is_slash'] = df_payout['winning_pgm'].str.count('/') > 0
        df_slash = df_payout[df_payout['is_slash'] * df_payout['is_multi_race']]
        df_payout = df_payout.drop(index=df_slash.index.values)

        df_payout = concat([df_payout, parse_slash(df_slash)])

        return df_payout

    def _get_scratches(self):
        """
        use ac.runner table for mapping program_number to df_scratch on x8name
        'program_number' and 'post_position' columns have null values for scratched runners in jcp files
        """
        raise Exception('DEPRECATED: This function has been deprecated because animal crackers was taken down.')

        print('reading engine_ac for scratches runner_ids')

        races = self.df_scratch['race_id'].unique()
        meta_data = MetaData()

        # sql
        runner = Table('runner', meta_data, autoload=True, autoload_with=engine_ac)
        stmt_runner = select([runner]).where(runner.columns.race_id.in_(races))
        sqlresults_runner = engine_ac.execute(stmt_runner).fetchall()
        df_runner = DataFrame(sqlresults_runner, columns=sqlresults_runner[0].keys())

        # columns for merging
        df_runner['x8name'] = df_runner['name'].map(self._normalize_name)

        # merge
        df_runner = df_runner[['x8name', 'race_id', 'program_number']]
        self.df_scratch = merge(self.df_scratch, df_runner, on=['race_id', 'x8name'], how='left')

        # correct column names so df_scratch is standardized
        self.df_scratch.rename(columns={'program_number_x': 'program_number',
                                        'program_number_y': 'program_number_scratch'},
                               inplace=True)

        # re make runner_id columns with program numbers
        self.df_scratch['runner_id'] = self.df_scratch['race_id'] + '_' + self.df_scratch['program_number_scratch'].map(str)

    def _normalize_name(self, name):
        # this used to be called make_canonical_name in main, and is unchanged
        canonical_name = name.upper().replace("'", "").strip()
        ending = canonical_name.find("(")
        if ending > -1:
            canonical_name = canonical_name[0:ending].strip().replace(" ", "")
        return canonical_name.replace(" ", "")

    def _country_from_name(self, name):
        '''IF name has "(XXX)" we  use Ireland e.g.'''
        split_name = name.split("(")
        if len(split_name) > 1:
            return split_name[1].replace(")", "")
        else:
            return "USA"

    def add_computed_columns(self):

        # probs and entropy
        self.df['prob_final_tote_odds'] = self.df.groupby('race_id')['final_tote_odds'].transform(compute_probs_from_odds)

        self.df['entropy_final_tote_odds'] = self.df.groupby('race_id')['prob_final_tote_odds'].transform(lambda x: entropy(x, base =len(x)))
        self.df['entropy_final_tote_odds'] = self.df['entropy_final_tote_odds'].map(lambda x: nan if isneginf(x) else x)

        self.df['prob_morning_line_odds'] = self.df.groupby('race_id')['morning_line'].transform(compute_probs_from_odds)
        self.df['rank_prob_morning_line_odds'] = self.df.groupby('race_id')['prob_morning_line_odds'].rank(ascending=False)

        self.df['entropy_morning_line_odds'] = self.df.groupby('race_id')['prob_morning_line_odds'].transform(lambda x: entropy(x, base=len(x)))
        self.df['entropy_morning_line_odds'] = self.df['entropy_morning_line_odds'].map(lambda x: nan if isneginf(x) else x)

        self.df['num_effective_starters_morning_line'] = self.df.entropy_morning_line_odds * self.df.num_starters_post
        self.df['num_effective_starters_final_tote_odds'] = self.df.entropy_final_tote_odds * self.df.num_starters_post
        #self.df['drop_morning_line_odds'] = (self.df['num_starters_post'] - self.df['num_effective_starters_morning_line']).map(round)
        self.df['diff_logprob_final_tote_morning_line'] = self.df['prob_final_tote_odds'].map(lambda x:math.log(x)) - self.df['prob_morning_line_odds'].map(lambda x:math.log(x))

        self.df['rank_prob_final_tote_odds'] = self.df.groupby('race_id')['prob_final_tote_odds'].rank(ascending=False)
        self.df['rank_diff_logprob_final_tote_morning_line'] = self.df.groupby('race_id')['diff_logprob_final_tote_morning_line'].transform(lambda x:x.rank(ascending=False))

        # sprint if 1759 yards (1 mile) or less, route if more
        self.df['is_route'] = self.df['distance'].map(lambda x: int(x > 1759))
        self.df['num_starters_post'] = self.df['num_starters_post'].map(lambda x: int(x))
        self.df['cost_exacta_from_win_show'] = self.df['num_starters_post'].map(lambda x: (x - 1) * 1)
        self.df['cost_trifecta_from_place_wc'] = self.df['num_starters_post'].map(lambda x: (x - 1) * (x - 2) * 2)
        self.df['cost_superfecta_from_show_a1'] = self.df['num_starters_post'].map(lambda x: (x - 1) * (x - 2) * (x-3) * 3)
        self.df['cost_synth_place_tri'] = self.df['num_starters_post'].map(lambda x: (x - 1) * (x - 2) * 2)
        #self.df['cost_synth_'] = self.df['num_starters_post'].map(lambda x: (x - 1) * (x - 2) * 2)
        self.df['log_ratio_effectivestarters_morningline'] = -1.0 * log(self.df.num_effective_starters_morning_line / self.df.num_starters_post)

        self.df['max_prob_morning_line_odds'] = self.df.groupby('race_id')['prob_morning_line_odds'].transform(lambda x: x.max())
        self.df['max_prob_final_tote_odds'] = self.df.groupby('race_id')['prob_final_tote_odds'].transform(lambda x: x.max())
        self.df['underperformance_weighted'] = (self.df['rank_prob_final_tote_odds'] - self.df['official_finish_position']) * self.df['prob_final_tote_odds']

        #self.df['log_ratio_effectivestarters_morningline'] = -1.0 * log(self.df

    def _add_mapped_columns(self):
        # deal with columns that have a distinct set codes and respective meanings
        path_map = map.__path__._path[0]
        for filename in os.listdir(path_map):
            fp = os.path.join(path_map, filename)
            _dict = read_csv(fp, header=None, dtype=str).set_index(0)[1].to_dict()
            col = filename.split('.')[1] # 'scratch_code' in 'map.scratch_code.csv'
            self.df['map_%s' % col] = self.df[col].map(_dict)

        # deal with True/False columns which are coded as 'Y' or nan, which will be mapped to 0 or 1
        cols_signal = self.df.columns[self.df.columns.str.startswith('is_')]
        for col in cols_signal:
            self.df[col] = self.df[col].map({nan: 0, 'Y': 1})

    def match_wom_dates(self, today, start_year, end_year, filter_month=True):
        """
        uses week of month to find historical dates
        :param today: datetime.date(2018,4,13)
        :param start_year: int ie. 2017
        :param end_year: int ie. 2018
        :param filter_month: bool, whether or not function should return dates in same month as today or not
        :return: dates
        """
        # imply week-of-month from given dates
        wom = get_freq_wom(today) #
        #wom.replace('5','4') #only handles 4 weeks

        # all dates in start/end year range that fall on the same week of month as today
        wom_dates = date_range(datetime(start_year, 1, 1), datetime(end_year, 12, 31), freq=wom)

        if filter_month:
            # return dates only in today's month
            return [r for r in wom_dates if r.month == today.month]
        else:
            # return dates in all months
            return [r for r in wom_dates]

    def make_result_matrix_race(self, _df, max_finish_pos):
        result_matrix = zeros((len(_df.x8name), max_finish_pos))
        fill_diagonal(result_matrix, 1)
        return DataFrame(index=_df.x8name, data=result_matrix, columns=range(1, max_finish_pos + 1))

    def get_result_matrix(self, index_cols = ['race_id','x8name']):
        '''Returns a matrix of 1,0 for each finish position from 1 to max_pos
        This results in an N by max_pos matrix of 1,0 indexed by index_cols'''
        max_finish_pos = self.df.official_finish_position.max()
        grp_raceid = self.df.groupby('race_id')
        return grp_raceid.apply(lambda dfrace: self.make_result_matrix_race(dfrace, max_finish_pos))

    def generate_winning_ticket(self, race_id, wager_name):
        '''Returns the string for the winning ticket for this wager_name'''
        df_raceid_wager = self.df_payout.groupby(['race_id','wager_name']).get_group((race_id, wager_name))

    def _validate(self, datelist):
        """check that the resulting DataFrame values are as expected"""
        # wager type
        # wagerCodes = list(map_exotic_wager.keys())
        # if not self.df['wager_type_1'].fillna('').isin(wagerCodes).all():
        #     difference = list(set(self.df['wager_type_1'].unique()) - set(wagerCodes))
        #     raise Exception('JCapper file wager types expected %s, but received %s, difference %s' % (wagerCodes, list(self.df['wager_type_1'].unique()), difference))
        #
        # # breed type
        # breedCodes = ['TB'] # should only be TB because we are only getting Thoroughbred data
        # if not self.df['breed_code'].isin(breedCodes).all():
        #     raise Exception('JCapper breed code expected %s, but received %s' % (breedCodes, list(self.df['breed_code'].unique())))
        #
        # # surface type
        # surfaceCodes = ['D', 'T', 'P']
        # if not self.df['surface_code'].isin(surfaceCodes).all():
        #     raise Exception('JCapper surface codes expected %s, but received %s' % (surfaceCodes, list(self.df['surface_code'].unique())))
        #
        # # course type
        # coursetypeCodes = ['M', 'I', 'O', 'H', 'S']
        # if not self.df['course_type_code'].isin(coursetypeCodes).all():
        #     raise Exception('JCapper course type codes expected %s, but received %s' % (coursetypeCodes, list(self.df['course_type_code'].unique())))
        #
        # # # race type
        # # racetypeCode = ['ALW', 'AOC', 'CLH', 'CLM', 'CST', 'HCP', 'MAT', 'MCL', 'MOC', 'MST', 'MSW', 'OCH', 'OCL', 'OCS', 'SHP', 'SST', 'STK', 'STR', 'TRL']
        # # if not self.df['race_type_code'].isin(racetypeCode).all():
        # #     raise Exception('JCapper race type codes expected %s, but received %s' % (racetypeCode, list(self.df['race_type_code'].unique())))
        #
        # # sex restriction
        # sexrestrictionCode = ['', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'M', 'R']
        # if not self.df['sex_restriction_code'].fillna('').isin(sexrestrictionCode).all():
        #     raise Exception('JCapper sex restriction codes expected %s, but received %s' % (sexrestrictionCode, list(self.df['sex_restriction_code'].unique())))
        #
        # # age restriction
        # agerestrictionCode = ['', '02', '03', '04', '05', '06', '07', '08', '09', 2, '23', '24', '2U', '34', '35', '36', '3U', '45', '46', '47', '4U', '56', '57', '58', '59', '5U', '67', '68', '69', '6U', '78', '79', '7U', '8U', '9U']
        # if not self.df['age_restriction_code'].fillna('').isin(agerestrictionCode).all():
        #     raise Exception('JCapper age restriction codes expected %s, but received %s' % (agerestrictionCode, list(self.df['age_restriction_code'].unique())))
        #
        # # graded stake race code
        # gradeCode = ['', '1', '2', '3']
        # if not self.df['grade_code'].fillna('').isin(gradeCode).all():
        #     raise Exception('JCapper grade codes expected %s, but received %s' % (gradeCode, list(self.df['grade_code'].unique())))
        #
        # # track condition code
        # trackconCode = ['FST', 'WF', 'GD', 'SLY', 'MY', 'SL', 'HY', 'FR', 'HD', 'FM', 'YL', 'SF']
        # if not self.df['track_condition_code'].isin(trackconCode).all():
        #     raise Exception('JCapper track condition codes expected %s, but received %s' % (trackconCode, list(self.df['track_condition_code'].unique())))
        #
        # # weather
        # weatherCode = ['A', 'B', 'C', 'E', 'F', 'H', 'L', 'O', 'R', 'S', 'T', 'W']
        # if not self.df['weather_code'].isin(weatherCode).all():
        #     raise Exception('JCapper weather codes expected %s, but received %s' % (weatherCode, list(self.df['weather_code'].unique())))

        # # horse sex
        # sexCode = ['F', 'M', 'C', 'H', 'G', 'R'] # DAZZLINGPIONEER does not have data
        # if not self.df['sex_code'].isin(sexCode).all():
        #     raise Exception('JCapper horse sex codes expected %s, but received %s' % (sexCode, list(self.df['sex_code'].unique())))

        # # medication used
        # medicationCode = ['', 'L', 'B', 'A', 'LA', 'LB', 'BL', 'nan'] # LA and LB are not in the charts schema - assuming they are combinations of both medications
        # if not self.df['medication_code'].fillna('').isin(medicationCode).all():
        #     raise Exception('JCapper medication codes expected %s, but received %s' % (medicationCode, list(self.df['medication_code'].unique())))
        #
        # # equipment used
        # equipmentCode = ['A', 'B', 'C', 'F', 'J', 'L', 'R', 'T', 'Z'] # probably need an updated list of equipment codes
        # if not self.df['equipment_code'].isin(equipmentCode).all():
        #     raise Exception('JCapper equipment codes expected %s, but received %s' % (equipmentCode, list(self.df['equipment_code'].unique())))

        wrong_dates = set(self.df.date.dt.date) - set([d.date() for d in datelist])
        if wrong_dates:
            raise Exception('s3 is housing .TXT files with wrong date in target date folders..\n'
                            'wrong_dates = %s' % wrong_dates)
