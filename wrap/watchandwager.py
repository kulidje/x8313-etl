from pandas import DataFrame, Series, Timestamp, read_csv, concat, to_datetime, merge
from os.path import join, splitext, basename
from s3fs.core import S3FileSystem
from horse.betsim import data
import gzip
from math import isclose
from warnings import warn
from datetime import datetime
from collections import defaultdict
from horse.betsim.math import compute_probs_from_odds
from numpy import nan

entries_columns = {0: 'betting_interest', 1: 'program_number', 2: 'horse_name', 3: 'birth_place', 4: 'birth_year',
                   5: 'jockey_name', 6: 'trainer_name', 7: 'scratched'}
results_columns = {0: 'bet_type', 1: 'pool_currency', 2: 'winning_pgm', 3: 'payout', 4: 'base_stake'}

# single leg, multi-runner and multi-race pool symbols
pool_single = ['WN', 'PL', 'SH']
pool_multi_runner = ['EX', 'QU', 'TR', 'SU']
pool_multi_race = ['DB', 'P3', 'P4', 'P5', 'P6']
minus_num_races = {'DB': 1, 'P3': 2, 'P4': 3, 'P5': 4, 'P6': 5}


class WatchandWager:
    """
    WatchandWager deals with data we recieve from watch and wager via AWS S3
    odds: win, place, show, exacta, quinella, quinella place, trifecta, superfecta, Pick N
    changes: changes (includes trainer, jockey, race changes and scratches)
    entries: only entries files
    results: results (payouts), finishers (finish positions)
    added feature: re-loading does not load from scratch when re-loading data, checks for new files and add instead of replaces
    """
    def __init__(self, verbose=False):
        # data
        self.df_entries = DataFrame()
        self.df_results = DataFrame()
        self.df_results_raw = DataFrame()
        self.df_scratches = DataFrame()
        self.df_win_odds = DataFrame()
        self.df_final_win_odds = DataFrame()
        self.df_place_odds = DataFrame()
        self.df_final_place_odds = DataFrame()
        self.df_show_odds = DataFrame()
        self.df_final_show_odds = DataFrame()
        self.df_wps = DataFrame()
        # dictionary of dictionaries - first level is race_id then race level meta data, second level is timestamp
        self.exacta = defaultdict(lambda: defaultdict(dict))
        self.trifecta = defaultdict(lambda: defaultdict(dict))
        self.final_trifecta = defaultdict(lambda: defaultdict(dict))
        self.superfecta = defaultdict(lambda: defaultdict(dict))
        # list of files we have already read so that we can filter files we have already read when re-loading
        self.files_read = []
        # now getting track detail exclusively from git file (horse/betsim/data/track_detail.csv) instead of relative path from where data is being loaded
        # this is because we are now loading data from s3 directly
        track_detail = join(data.__path__._path[0], 'track_detail.csv')
        dftrack = read_csv(track_detail)
        # mapping dictionaries
        self.map_itsp_to_x8 = dftrack.set_index('itsp_track_sym')['x8_track_sym'].to_dict()
        self.map_x8_to_itsp = dftrack.set_index('x8_track_sym')['itsp_track_sym'].to_dict()
        # verbose
        self.verbose = verbose

        # TODO add dfraw to each loader
        # TODO add more assertions to _validate()'s

    def load_win_odds(self, datelist, tracklist=[], race_id=''):
        """
        load all live win odds for a day, includes odds at 5 MTP, 3 MTP, 1 MTP
        :param datelist: list of date objects, e.g. list(datetime(), datetime())
        :return: DataFrame
        """
        # instantiate s3 file storage - must be inside functions so it's not static
        s3 = S3FileSystem(anon=False)

        # convert tracklist symbols to itsp track symbols
        tracklist = Series(tracklist).map(self.map_x8_to_itsp)
        if tracklist.isnull().any():
            raise ValueError('param tracklist must be list of x8 track symbols in track_detail: \n%s' % tracklist)

        for d in datelist:
            date_str = d.strftime('%Y-%m-%d')
            key = 'x8-bucket/odds/%s' % date_str
            try:
                s3_files = s3.ls(key) # list of all files in a given direcetory - in this case, all files for a single day
                # filter for only files with 'WN' in name - win odds (including final_odds)
                s3_files = [fp for fp in s3_files if basename(fp).find('_WN_') > -1]
                # filter for only odds files, not final_odds
                s3_files = [fp for fp in s3_files if basename(fp).startswith('odds')]
                # filter by race_id, if not given then filter by tracklist, if not given than load all for given dates
                if race_id:
                    s3_files = [fp for fp in s3_files if fp.split('_')[1] == self.map_x8_to_itsp[race_id.split('_')[0]]]
                    s3_files = [fp for fp in s3_files if fp.split('_')[-4] == race_id.split('_')[2]]
                elif not tracklist.empty:
                    s3_files = [fp for fp in s3_files if fp.split('_')[1] in list(tracklist)]
                # filter out files we have already read
                s3_files = list(set(s3_files) - set(self.files_read))
                self.files_read += s3_files
                if self.verbose:
                    print('ww.load_win_odds(%s) loading %s files..' % (date_str, len(s3_files)))
                for fp in s3_files:
                    # 'odds_AQD_THOROUGHBRED_USA_2018-02-24_1_WN_USD_1504215337'
                    f_name = splitext(splitext(basename(fp))[0])[0].split('_')
                    lines = [line.decode().replace('\n', '').split(',') for line in gzip.open(s3.open(fp, mode='rb'), 'rb')]
                    df = DataFrame([l for i, l in enumerate(lines) if i in [3,5,7]]).unstack().unstack()
                    try:
                        df.columns = ['odds', 'probables', 'combo_totals']
                        df['betting_interest'] = (df.index + 1).astype(str)
                        df['date'] = to_datetime(d.date())
                        df['itsp_track_sym'] = f_name[1]  # 'AQD'
                        df['country'] = f_name[3]  # 'USA'
                        df['race_num'] = f_name[-4]  # '1'
                        df['currency'] = f_name[-2]  # 'USD'
                        # time odds were posted
                        df['timestamp'] = to_datetime(f_name[-1], unit='s')
                        # parse second line, separately from rest of file
                        df['carry_in'] = float(lines[1][0])
                        df['pool_total'] = float(lines[1][1])
                        df['base_probable_stake'] = float(lines[1][2])
                    except ValueError:
                        # this try block and exception is for the case when odds files come out with nothing but '-' in the odds line
                        # this means no odds yet or very small pool total (under ~$10)
                        warn('Excepting ValueError, no data: %s' % fp)
                        # df still exists but with no data and [0,1] column names
                        df = DataFrame()

                    # concat to master win odds df
                    self.df_win_odds = concat([self.df_win_odds, df])

            except FileNotFoundError:
                warn('There are no win odds files for %s yet. Skipping this day.' % d.date()) # more specific message instead of generic FileNotFoundError

        # for case when there is no data for dates trying to load
        if not self.df_win_odds.empty:
            # track symbol conversion
            self.df_win_odds['x8_track_sym'] = self.df_win_odds['itsp_track_sym'].map(self.map_itsp_to_x8)

            # drop any rows where itsp track symbol mapping is missing from track detail if any
            x8_isnull = self.df_win_odds['x8_track_sym'].isnull()
            if x8_isnull.any():
                missing_itsp_symbols = self.df_win_odds[x8_isnull]['itsp_track_sym'].unique()
                warn('ww.load_win_odds() track_detail.csv is missing itsp symbols: %s\nDropping all rows with missing symbols' % missing_itsp_symbols)
                self.df_win_odds = self.df_win_odds[~x8_isnull]
                print('ww.load_win_odds() dropping %s rows\n' % x8_isnull.sum())

            self.df_win_odds['probables'] = self.df_win_odds['probables'].astype(float)
            self.df_win_odds['probables'] = self.df_win_odds['probables'].replace(0.0, nan)

            # race/runner identifiers
            self.df_win_odds['race_id'] = self.df_win_odds['x8_track_sym'] + '_' + self.df_win_odds['date'].dt.strftime('%Y%m%d') + '_' + self.df_win_odds['race_num']

            # make ordinal timestamp (order 5 MTP, 3 MTP, 1 MTP, Final)
            self.df_win_odds['ordinal_timestamp'] = self.df_win_odds.groupby('race_id')['timestamp'].transform(lambda x: x.rank(ascending=True, method='dense'))
            self.df_win_odds['max_ots'] = self.df_win_odds.groupby('race_id')['ordinal_timestamp'].transform('max')

            self.df_win_odds['market_odds'] = self.df_win_odds['probables'] / self.df_win_odds['base_probable_stake'] - 1
            self.df_win_odds['prob_market_odds'] = self.df_win_odds.groupby(['race_id', 'ordinal_timestamp'])['market_odds'].transform(compute_probs_from_odds)

            self._validate_win_odds()

    def load_final_win_odds(self, datelist, tracklist=[]):
        """
        load all final win odds for a day
        :param datelist: list of date objects, e.g. list(datetime(), datetime())
        :return: DataFrame
        """
        # instantiate s3 file storage - must be inside functions so it's not static
        s3 = S3FileSystem(anon=False)

        # convert tracklist symbols to itsp track symbols
        tracklist = Series(tracklist).map(self.map_x8_to_itsp)
        if tracklist.isnull().any():
            raise ValueError('param tracklist must be list of x8 track symbols in track_detail: \n%s' % tracklist)

        for d in datelist:
            date_str = d.strftime('%Y-%m-%d')
            key = 'x8-bucket/odds/%s' % date_str
            try:
                s3_files = s3.ls(key) # list of all files in a given direcetory - in this case, all files for a single day
                # filter for only files with 'WN' in name - win odds (including final_odds)
                s3_files = [fp for fp in s3_files if basename(fp).find('_WN_') > -1]
                # filter for only final_odds files
                s3_files = [fp for fp in s3_files if basename(fp).startswith('final')]
                if not tracklist.empty:
                    s3_files = [fp for fp in s3_files if fp.split('_')[2] in list(tracklist)]
                # filter out files we have already read
                s3_files = list(set(s3_files) - set(self.files_read))
                self.files_read += s3_files
                if self.verbose:
                    print('ww.load_final_win_odds(%s) loading %s files..' % (date_str, len(s3_files)))
                for fp in s3_files:
                    # 'final_odds_AQD_THOROUGHBRED_USA_2018-02-24_1_WN_USD'
                    f_name = splitext(splitext(basename(fp))[0])[0].split('_')
                    lines = [line.decode().replace('\n', '').split(',') for line in gzip.open(s3.open(fp, mode='rb'), 'rb')]
                    df = DataFrame([l for i, l in enumerate(lines) if i in [3,5,7]]).unstack().unstack()
                    try:
                        df.columns = ['odds', 'probables', 'combo_totals']
                        df['betting_interest'] = (df.index + 1).astype(str)
                        df['date'] = to_datetime(d.date())
                        df['itsp_track_sym'] = f_name[2]  # 'AQD'
                        df['country'] = f_name[4]  # 'USA'
                        df['race_num'] = f_name[-3]  # '1'
                        df['currency'] = f_name[-1]  # 'USD'
                        # parse second line, separately from rest of file
                        df['carry_in'] = float(lines[1][0])
                        df['pool_total'] = float(lines[1][1])
                        df['base_probable_stake'] = float(lines[1][2])
                    except ValueError:
                        # this try block and exception is for the case when odds files come out with nothing but '-' in the odds line
                        # this means no odds yet or very small pool total (under ~$10)
                        warn('Excepting ValueError, no data: %s' % fp)
                        # df still exists but with no data and [0,1] column names
                        df = DataFrame()

                    # concat to master final win odds df
                    self.df_final_win_odds = concat([self.df_final_win_odds, df])

            except FileNotFoundError:
                warn('There are no final win odds files for %s yet. Skipping this day.' % d.date()) # more specific message instead of generic FileNotFoundError

        # for case when there is no data for dates trying to load
        if not self.df_final_win_odds.empty:
            # symbol conversion
            self.df_final_win_odds['x8_track_sym'] = self.df_final_win_odds['itsp_track_sym'].map(self.map_itsp_to_x8)

            # drop any rows where itsp track symbol mapping is missing from track detail if any
            x8_isnull = self.df_final_win_odds['x8_track_sym'].isnull()
            if x8_isnull.any():
                missing_itsp_symbols = self.df_final_win_odds[x8_isnull]['itsp_track_sym'].unique()
                warn('ww.load_final_win_odds() track_detail.csv is missing itsp symbols: %s\nDropping all rows with missing symbols' % missing_itsp_symbols)
                self.df_final_win_odds = self.df_final_win_odds[~x8_isnull]
                print('ww.load_final_win_odds() dropping %s rows\n' % x8_isnull.sum())

            self.df_final_win_odds['probables'] = self.df_final_win_odds['probables'].astype(float)
            self.df_final_win_odds['probables'] = self.df_final_win_odds['probables'].replace(0.0, nan)

            # race/runner identifiers
            self.df_final_win_odds['race_id'] = self.df_final_win_odds['x8_track_sym'] + '_' + self.df_final_win_odds['date'].dt.strftime('%Y%m%d') + '_' + self.df_final_win_odds['race_num']

            self.df_final_win_odds['market_odds'] = self.df_final_win_odds['probables'] / self.df_final_win_odds['base_probable_stake'] - 1
            self.df_final_win_odds['prob_market_odds'] = self.df_final_win_odds.groupby('race_id')['market_odds'].transform(compute_probs_from_odds)

            self._validate_final_win_odds()

    def load_place_odds(self, datelist):
        """
        load all live place odds for a day, includes odds at 5 MTP, 3 MTP, 1 MTP
        :param datelist: list of date objects, e.g. list(datetime(), datetime())
        :return: DataFrame
        """
        # instantiate s3 file storage - must be inside functions so it's not static
        s3 = S3FileSystem(anon=False)

        for d in datelist:
            date_str = d.strftime('%Y-%m-%d')
            key = 'x8-bucket/odds/%s' % date_str
            try:
                s3_files = s3.ls(key) # list of all files in a given direcetory - in this case, all files for a single day
                # filter for only files with 'PL' in name - win odds (including final_odds)
                s3_files = [fp for fp in s3_files if basename(fp).find('_PL_') > -1]
                # filter for only odds files, not final_odds
                s3_files = [fp for fp in s3_files if basename(fp).startswith('odds')]
                # filter out files we have already read
                s3_files = list(set(s3_files) - set(self.files_read))
                self.files_read += s3_files
                if self.verbose:
                    print('ww.load_place_odds(%s) loading %s files..' % (date_str, len(s3_files)))
                for fp in s3_files:
                    # 'odds_AQD_THOROUGHBRED_USA_2018-04-08_6_PL_USD_1523218250'
                    f_name = splitext(splitext(basename(fp))[0])[0].split('_')
                    lines = [line.decode().replace('\n', '').split(',') for line in gzip.open(s3.open(fp, mode='rb'), 'rb')]
                    df = DataFrame([l for i, l in enumerate(lines) if i in [5, 6, 8]], dtype=float).unstack().unstack()
                    try:
                        # no 'odds' data just probables, combo totals and pool totals
                        df.columns = ['probables_1', 'probables_2', 'combo_totals']
                        df['program_number'] = (df.index + 1).astype(str)
                        df['date'] = to_datetime(d.date())
                        df['itsp_track_sym'] = f_name[1]  # 'AQD'
                        df['country'] = f_name[3]  # 'USA'
                        df['race_num'] = f_name[-4]  # '6'
                        df['currency'] = f_name[-2]  # 'USD'
                        # time odds were posted
                        df['timestamp'] = to_datetime(f_name[-1], unit='s')
                        df['carry_in'] = float(lines[1][0])
                        df['pool_total'] = float(lines[1][1])
                        df['base_probable_stake'] = float(lines[1][2])
                    except ValueError:
                        # this try block and exception is for the case when odds files come out with nothing but '-' in the odds line
                        # this means no odds yet or very small pool total (under ~$10)
                        warn('Excepting ValueError, no data: %s' % fp)
                        # df still exists but with no data and [0,1] column names
                        df = DataFrame()

                    # concat to master win odds df
                    self.df_place_odds = concat([self.df_place_odds, df])

            except FileNotFoundError:
                warn('There are no place odds files for %s yet. Skipping this day.' % d.date()) # more specific message instead of generic FileNotFoundError

        # for case when there is no data for dates trying to load
        if not self.df_place_odds.empty:
            # track symbol conversion
            self.df_place_odds['x8_track_sym'] = self.df_place_odds['itsp_track_sym'].map(self.map_itsp_to_x8)

            # drop any rows where itsp track symbol mapping is missing from track detail if any
            x8_isnull = self.df_place_odds['x8_track_sym'].isnull()
            if x8_isnull.any():
                missing_itsp_symbols = self.df_place_odds[x8_isnull]['itsp_track_sym'].unique()
                warn('ww.load_place_odds() track_detail.csv is missing itsp symbols: %s\nDropping all rows with missing symbols' % missing_itsp_symbols)
                self.df_place_odds = self.df_place_odds[~x8_isnull]
                print('ww.load_place_odds() dropping %s rows\n' % x8_isnull.sum())

            # get playable odds - first drop all rows w/ scratches runner (nans)
            self.df_place_odds = self.df_place_odds[self.df_place_odds['probables_1'] != 0.0]

            # race / runner identifiers
            self.df_place_odds['race_id'] = self.df_place_odds['x8_track_sym'] + '_' + self.df_place_odds['date'].dt.strftime('%Y%m%d') + '_' + self.df_place_odds['race_num']
            self.df_place_odds['runner_id'] = self.df_place_odds['race_id'] + '_' + self.df_place_odds['program_number']

            # make ordinal timestamp (order 5 MTP, 3 MTP, 1 MTP, Final)
            self.df_place_odds['ordinal_timestamp'] = self.df_place_odds.groupby('race_id')['timestamp'].transform(lambda x: x.rank(ascending=True, method='dense'))

            #self._validate_place_odds()

    def load_final_place_odds(self, datelist):
        """
        load all final place odds for a day
        :param datelist: list of date objects, e.g. list(datetime(), datetime())
        :return: DataFrame
        """
        # instantiate s3 file storage - must be inside functions so it's not static
        s3 = S3FileSystem(anon=False)

        for d in datelist:
            date_str = d.strftime('%Y-%m-%d')
            key = 'x8-bucket/odds/%s' % date_str
            try:
                s3_files = s3.ls(key) # list of all files in a given direcetory - in this case, all files for a single day
                # filter for only files with 'PL' in name - win odds (including final_odds)
                s3_files = [fp for fp in s3_files if basename(fp).find('_PL_') > -1]
                # filter for only odds files, not final_odds
                s3_files = [fp for fp in s3_files if basename(fp).startswith('final')]
                # filter out files we have already read
                s3_files = list(set(s3_files) - set(self.files_read))
                self.files_read += s3_files
                if self.verbose:
                    print('ww.load_final_place_odds(%s) loading %s files..' % (date_str, len(s3_files)))
                for fp in s3_files:
                    # 'final_odds_AQD_THOROUGHBRED_USA_2018-04-08_6_PL_USD'
                    f_name = splitext(splitext(basename(fp))[0])[0].split('_')
                    lines = [line.decode().replace('\n', '').split(',') for line in gzip.open(s3.open(fp, mode='rb'), 'rb')]
                    df = DataFrame([l for i, l in enumerate(lines) if i in [5, 6, 8]], dtype=float).unstack().unstack()
                    try:
                        # no 'odds' data just probables, combo totals and pool totals
                        df.columns = ['probables_1', 'probables_2', 'combo_totals']
                        df['program_number'] = (df.index + 1).astype(str)
                        df['date'] = to_datetime(d.date())
                        df['itsp_track_sym'] = f_name[2]  # 'AQD'
                        df['country'] = f_name[4]  # 'USA'
                        df['race_num'] = f_name[-3]  # '6'
                        df['currency'] = f_name[-1]  # 'USD'
                        # parse second line, separately from rest of file
                        df['carry_in'] = float(lines[1][0])
                        df['pool_total'] = float(lines[1][1])
                        df['base_probable_stake'] = float(lines[1][2])
                    except ValueError:
                        # this try block and exception is for the case when odds files come out with nothing but '-' in the odds line
                        # this means no odds yet or very small pool total (under ~$10)
                        warn('Excepting ValueError, no data: %s' % fp)
                        # df still exists but with no data and [0,1] column names
                        df = DataFrame()

                    # concat to master win odds df
                    self.df_final_place_odds = concat([self.df_final_place_odds, df])

            except FileNotFoundError:
                warn('There are no final place odds files for %s yet. Skipping this day.' % d.date()) # more specific message instead of generic FileNotFoundError

        # for case when there is no data for dates trying to load
        if not self.df_final_place_odds.empty:
            # track symbol conversion
            self.df_final_place_odds['x8_track_sym'] = self.df_final_place_odds['itsp_track_sym'].map(self.map_itsp_to_x8)

            # drop any rows where itsp track symbol mapping is missing from track detail if any
            x8_isnull = self.df_final_place_odds['x8_track_sym'].isnull()
            if x8_isnull.any():
                missing_itsp_symbols = self.df_final_place_odds[x8_isnull]['itsp_track_sym'].unique()
                warn('ww.load_final_place_odds() track_detail.csv is missing itsp symbols: %s\nDropping all rows with missing symbols' % missing_itsp_symbols)
                self.df_final_place_odds = self.df_final_place_odds[~x8_isnull]
                print('ww.load_final_place_odds() dropping %s rows\n' % x8_isnull.sum())

            # get playable odds - first drop all rows w/ scratches runner (nans)
            self.df_final_place_odds = self.df_final_place_odds[self.df_final_place_odds['probables_1'] != 0.0]

            # race / runner identifiers
            self.df_final_place_odds['race_id'] = self.df_final_place_odds['x8_track_sym'] + '_' + self.df_final_place_odds['date'].dt.strftime('%Y%m%d') + '_' + self.df_final_place_odds['race_num']
            self.df_final_place_odds['runner_id'] = self.df_final_place_odds['race_id'] + '_' + self.df_final_place_odds['program_number']

            #self._validate_final_place_odds()

    def load_show_odds(self, datelist):
        """
        load all live show odds for a day, includes odds at 5 MTP, 3 MTP, 1 MTP
        :param datelist: list of date objects, e.g. list(datetime(), datetime())
        :return: DataFrame
        """
        # instantiate s3 file storage - must be inside functions so it's not static
        s3 = S3FileSystem(anon=False)

        for d in datelist:
            date_str = d.strftime('%Y-%m-%d')
            key = 'x8-bucket/odds/%s' % date_str
            try:
                s3_files = s3.ls(key) # list of all files in a given direcetory - in this case, all files for a single day
                # filter for only files with 'SH' in name - show odds (including final_odds)
                s3_files = [fp for fp in s3_files if basename(fp).find('_SH_') > -1]
                # filter for only odds files, not final_odds
                s3_files = [fp for fp in s3_files if basename(fp).startswith('odds')]
                # filter out files we have already read
                s3_files = list(set(s3_files) - set(self.files_read))
                self.files_read += s3_files
                if self.verbose:
                    print('ww.load_show_odds(%s) loading %s files..' % (date_str, len(s3_files)))
                for fp in s3_files:
                    f_name = splitext(splitext(basename(fp))[0])[0].split('_')
                    lines = [line.decode().replace('\n', '').split(',') for line in gzip.open(s3.open(fp, mode='rb'), 'rb')]
                    df = DataFrame([l for i, l in enumerate(lines) if i in [5, 6, 8]], dtype=float).unstack().unstack()
                    try:
                        # no 'odds' data just probables, combo totals and pool totals
                        df.columns = ['probables_1', 'probables_2', 'combo_totals']
                        df['program_number'] = (df.index + 1).astype(str)
                        df['date'] = to_datetime(d.date())
                        df['itsp_track_sym'] = f_name[1]
                        df['country'] = f_name[3]
                        df['race_num'] = f_name[-4]
                        df['currency'] = f_name[-2]
                        # time odds were posted
                        df['timestamp'] = to_datetime(f_name[-1], unit='s')
                        # parse second line, separately from rest of file
                        df['carry_in'] = float(lines[1][0])
                        df['pool_total'] = float(lines[1][1])
                        df['base_probable_stake'] = float(lines[1][2])
                    except ValueError:
                        # this try block and exception is for the case when odds files come out with nothing but '-' in the odds line
                        # this means no odds yet or very small pool total (under ~$10)
                        warn('Excepting ValueError, no data: %s' % fp)
                        # df still exists but with no data and [0,1] column names
                        df = DataFrame()

                    # concat to master win odds df
                    self.df_show_odds = concat([self.df_show_odds, df])

            except FileNotFoundError:
                warn('There are no show odds files for %s yet. Skipping this day.' % d.date()) # more specific message instead of generic FileNotFoundError

        # for case when there is no data for dates trying to load
        if not self.df_show_odds.empty:
            # track symbol conversion
            self.df_show_odds['x8_track_sym'] = self.df_show_odds['itsp_track_sym'].map(self.map_itsp_to_x8)

            # drop any rows where itsp track symbol mapping is missing from track detail if any
            x8_isnull = self.df_show_odds['x8_track_sym'].isnull()
            if x8_isnull.any():
                missing_itsp_symbols = self.df_show_odds[x8_isnull]['itsp_track_sym'].unique()
                warn('ww.load_show_odds() track_detail.csv is missing itsp symbols: %s\nDropping all rows with missing symbols' % missing_itsp_symbols)
                self.df_show_odds = self.df_show_odds[~x8_isnull]
                print('ww.load_show_odds() dropping %s rows\n' % x8_isnull.sum())

            # get playable odds - first drop all rows w/ scratches runner (nans)
            self.df_show_odds = self.df_show_odds[self.df_show_odds['probables_1'] != 0.0]

            # race / runner identifiers
            self.df_show_odds['race_id'] = self.df_show_odds['x8_track_sym'] + '_' + self.df_show_odds['date'].dt.strftime('%Y%m%d') + '_' + self.df_show_odds['race_num']
            self.df_show_odds['runner_id'] = self.df_show_odds['race_id'] + '_' + self.df_show_odds['program_number']

            # make ordinal timestamp (order 5 MTP, 3 MTP, 1 MTP, Final)
            self.df_show_odds['ordinal_timestamp'] = self.df_show_odds.groupby('race_id')['timestamp'].transform(lambda x: x.rank(ascending=True, method='dense'))

            #self._validate_show_odds()

    def load_final_show_odds(self, datelist):
        """
        load all final show odds for a day
        :param datelist: list of date objects, e.g. list(datetime(), datetime())
        :return: DataFrame
        """
        # instantiate s3 file storage - must be inside functions so it's not static
        s3 = S3FileSystem(anon=False)

        for d in datelist:
            date_str = d.strftime('%Y-%m-%d')
            key = 'x8-bucket/odds/%s' % date_str
            try:
                s3_files = s3.ls(key) # list of all files in a given direcetory - in this case, all files for a single day
                # filter for only files with 'SH' in name - show odds (including final_odds)
                s3_files = [fp for fp in s3_files if basename(fp).find('SH') > -1]
                # filter for only odds files, not final_odds
                s3_files = [fp for fp in s3_files if basename(fp).startswith('final')]
                # filter out files we have already read
                s3_files = list(set(s3_files) - set(self.files_read))
                self.files_read += s3_files
                if self.verbose:
                    print('ww.load_final_show_odds(%s) loading %s files..' % (date_str, len(s3_files)))
                for fp in s3_files:
                    f_name = splitext(splitext(basename(fp))[0])[0].split('_')
                    lines = [line.decode().replace('\n', '').split(',') for line in gzip.open(s3.open(fp, mode='rb'), 'rb')]
                    df = DataFrame([l for i, l in enumerate(lines) if i in [5, 6, 8]], dtype=float).unstack().unstack()
                    try:
                        # no 'odds' data just probables, combo totals and pool totals
                        df.columns = ['probables_1', 'probables_2', 'combo_totals']
                        df['program_number'] = (df.index + 1).astype(str)
                        df['date'] = to_datetime(d.date())
                        df['itsp_track_sym'] = f_name[2]
                        df['country'] = f_name[4]
                        df['race_num'] = f_name[-3]
                        df['currency'] = f_name[-1]
                        # parse second line, separately from rest of file
                        df['carry_in'] = float(lines[1][0])
                        df['pool_total'] = float(lines[1][1])
                        df['base_probable_stake'] = float(lines[1][2])
                    except ValueError:
                        # this try block and exception is for the case when odds files come out with nothing but '-' in the odds line
                        # this means no odds yet or very small pool total (under ~$10)
                        warn('Excepting ValueError, no data: %s' % fp)
                        # df still exists but with no data and [0,1] column names
                        df = DataFrame()

                    # concat to master win odds df
                    self.df_final_show_odds = concat([self.df_final_show_odds, df])

            except FileNotFoundError:
                warn('There are no final show odds files for %s yet. Skipping this day.' % d.date()) # more specific message instead of generic FileNotFoundError

        # for case when there is no data for dates trying to load
        if not self.df_final_show_odds.empty:
            # track symbol conversion
            self.df_final_show_odds['x8_track_sym'] = self.df_final_show_odds['itsp_track_sym'].map(self.map_itsp_to_x8)

            # drop any rows where itsp track symbol mapping is missing from track detail if any
            x8_isnull = self.df_final_show_odds['x8_track_sym'].isnull()
            if x8_isnull.any():
                missing_itsp_symbols = self.df_final_show_odds[x8_isnull]['itsp_track_sym'].unique()
                warn('ww.load_final_show_odds() track_detail.csv is missing itsp symbols: %s\nDropping all rows with missing symbols' % missing_itsp_symbols)
                self.df_final_show_odds = self.df_final_show_odds[~x8_isnull]
                print('ww.load_final_show_odds() dropping %s rows\n' % x8_isnull.sum())

            # get playable odds - first drop all rows w/ scratches runner (nans)
            self.df_final_show_odds = self.df_final_show_odds[self.df_final_show_odds['probables_1'] != 0.0]

            # race / runner identifiers
            self.df_final_show_odds['race_id'] = self.df_final_show_odds['x8_track_sym'] + '_' + self.df_final_show_odds['date'].dt.strftime('%Y%m%d') + '_' + self.df_final_show_odds['race_num']
            self.df_final_show_odds['runner_id'] = self.df_final_show_odds['race_id'] + '_' + self.df_final_show_odds['program_number']

            #self._validate_final_show_odds()

    def load_wps(self, datelist):
        """
        - load wps odds
        - merge
        - column rename
        :param datelist: list of datetimes
        """

        self.load_win_odds(datelist)
        self.load_place_odds(datelist)
        self.load_show_odds(datelist)

        # remove shared columns from place and show df's
        shared = ['race_id', 'country', 'program_number', 'date', 'itsp_track_sym', 'race_num', 'currency',
                  'x8_track_sym']

        win = self.df_win_odds.copy()
        place = self.df_place_odds.drop(shared, axis=1)
        show = self.df_show_odds.drop(shared, axis=1)

        # TODO merge on race_id and betting interest (odds do not have true runner_id because they don't have program number data)
        # add suffixes
        win.columns = [str(col) + '_wn' if col not in shared + ['runner_id', 'ordinal_timestamp'] else col for col in win.columns]
        place.columns = [str(col) + '_pl' if col not in ['runner_id', 'ordinal_timestamp'] else col for col in place.columns]
        show.columns = [str(col) + '_sh' if col not in ['runner_id', 'ordinal_timestamp'] else col for col in show.columns]

        self.df_wps = merge(win, place, on=['runner_id', 'ordinal_timestamp'], how='inner')
        self.df_wps = merge(self.df_wps, show, on=['runner_id', 'ordinal_timestamp'], how='inner')

    def load_exacta(self, datelist):
        """
        load all exacta odds for a day - does not include probables or combo totals
        feature: re-loading does not load from scratch, it checks for new data and adds to data instead of replaces
        :param datelist: list of date objects, e.g. list(datetime(), datetime())
        :return: defaultdict of defauldicts
        """
        # instantiate s3 file storage - must be inside functions so it's not static
        s3 = S3FileSystem(anon=False)

        for d in datelist:
            date_str = d.strftime('%Y-%m-%d')
            key = 'x8-bucket/odds/%s' % date_str
            try:
                s3_files = s3.ls(key) # list of all files in a given direcetory - in this case, all files for a single day
                # keep only files with 'EX' in name - show odds (including final_odds)
                s3_files = [fp for fp in s3_files if basename(fp).find('_EX_') > -1]
                # keep only odds files, not final_odds
                s3_files = [fp for fp in s3_files if basename(fp).startswith('odds')]
                # filter out files we have already read
                s3_files = list(set(s3_files) - set(self.files_read))
                self.files_read += s3_files
                if self.verbose:
                    print('ww.load_exacta(%s) loading %s files..' % (date_str, len(s3_files)))
                for fp in s3_files:
                    # 'odds_PHD_THOROUGHBRED_USA_2018-04-09_8_EX_USD_1523304774'
                    f_name = splitext(splitext(basename(fp))[0])[0].split('_')
                    # list of lists of entire file
                    # using this method instead of read_csv because we cannot know which lines of an exacta file
                    # to skip before reading because they are dependant on number of runners in a race
                    lines = [line.decode().replace('\n', '').split(',') for line in gzip.open(s3.open(fp, mode='rb'), 'rb')]
                    # combo totals section of exacta file
                    df = DataFrame(lines[lines.index(['combo totals'])+1:])
                    date = to_datetime(d.date())
                    itsp_track_sym = f_name[1]
                    x8_track_sym = self.map_itsp_to_x8.get(itsp_track_sym)
                    # if itsp_track_sym is missing from track_detail.csv then skip
                    if not x8_track_sym:
                        continue
                    country = f_name[3]
                    race_num = f_name[-4]
                    currency = f_name[-2]
                    # time odds were posted
                    timestamp = to_datetime(f_name[-1], unit='s')
                    # second line of file
                    carry_in = float(lines[1][0])
                    pool_total = float(lines[1][1])
                    base_probable_stake = float(lines[1][2])
                    race_id = x8_track_sym + '_' + date.strftime('%Y%m%d') + '_' + race_num
                    # make runner_id along index and columns
                    df.index = race_id + '_' + (df.index + 1).astype(str)
                    df.columns = race_id + '_' + (df.columns + 1).astype(str)

                    # this section is for making exacta matrix into a df
                    df = df.applymap(float)
                    # pool total when you sum the actual exacta matrix
                    inferred_pool = df.sum().sum()
                    # divide all probables by inferred pool to make normalized probabilities
                    df = df.applymap(lambda x: x/inferred_pool)
                    # name index
                    df.index.name = 'pos_1'
                    df.columns.name = 'pos_2'
                    # make df
                    df = df.unstack().reset_index()
                    # make combination column - tuples of runner_id's
                    df['combination'] = df.apply(lambda x: (x['pos_1'], x['pos_2']), axis=1)
                    # drop combinations w/ same runner
                    df = df[df['pos_1'] != df['pos_2']]
                    # columns
                    df.drop(['pos_1', 'pos_2'], axis=1, inplace=True)
                    df.rename(columns={0: 'probs'}, inplace=True)
                    # columns
                    df['timestamp'] = timestamp
                    df['carry_in'] = carry_in
                    df['pool_total'] = pool_total
                    df['base_probable_stake'] = base_probable_stake

                    # dataframe
                    self.exacta[race_id][timestamp] = df

                    # race level meta data
                    self.exacta[race_id]['date'] = date
                    self.exacta[race_id]['itsp_track_sym'] = itsp_track_sym
                    self.exacta[race_id]['x8_track_sym'] = x8_track_sym
                    self.exacta[race_id]['race_num'] = race_num
                    self.exacta[race_id]['country'] = country
                    self.exacta[race_id]['currency'] = currency

                # rename file keys to ordinal timestamp instead of timestamp
                for race_id, race_dict in self.exacta.items():
                    ots = []
                    # create ots dict of timestamp: rank (order)
                    for key, value in race_dict.items():
                        if isinstance(key, Timestamp):
                            ots.append(key)
                    ots = Series(ots)
                    ots.index = ots.values
                    ots = ots.rank(ascending=True, method='dense')
                    ots = ots.to_dict()
                    # rename keys in master dict
                    for key, value in race_dict.items():
                        if isinstance(key, Timestamp):
                            self.exacta[race_id][ots[key]] = self.exacta[race_id][key]
                            del self.exacta[race_id][key]

            except FileNotFoundError:
                # more specific message instead of generic FileNotFoundError
                warn('There are no exacta odds files for %s yet. Skipping this day.' % d.date())

        self._validate_exacta()

    def load_trifecta(self, datelist):
        """
        load trifecta odds files - only contain pool size
        :param datelist: list of date objects, e.g. list(datetime(), datetime())
        :return: defaultdict of defaultdicts
        """
        # instantiate s3 file storage - must be inside functions so it's not static
        s3 = S3FileSystem(anon=False)

        for d in datelist:
            date_str = d.strftime('%Y-%m-%d')
            key = 'x8-bucket/odds/%s' % date_str
            try:
                s3_files = s3.ls(key) # list of all files in a given direcetory - in this case, all files for a single day
                # keep only files with 'TR' in name - show odds (this will include final_odds_* files)
                s3_files = [fp for fp in s3_files if basename(fp).find('_TR_') > -1]
                # keep only odds files, not final_odds
                s3_files = [fp for fp in s3_files if basename(fp).startswith('odds')]
                # filter out files we have already read
                s3_files = list(set(s3_files) - set(self.files_read))
                self.files_read += s3_files
                if self.verbose:
                    print('ww.load_trifecta(%s) loading %s files..' % (date_str, len(s3_files)))
                for fp in s3_files:
                    # odds_AQD_THOROUGHBRED_USA_2018-03-11_4_TR_USD_1520794552
                    f_name = splitext(splitext(basename(fp))[0])[0].split('_')
                    # list of lists of entire file
                    # using this method instead of read_csv because we cannot know which lines of an trifecta file
                    # to skip before reading because they are dependant on number of runners in a race
                    lines = [line.decode().replace('\n', '').split(',') for line in gzip.open(s3.open(fp, mode='rb'), 'rb')]
                    # meta data parsing
                    date = to_datetime(d.date())
                    itsp_track_sym = f_name[1]
                    x8_track_sym = self.map_itsp_to_x8.get(itsp_track_sym)
                    # if itsp_track_sym is missing from track_detail.csv then skip
                    if not x8_track_sym:
                        continue
                    country = f_name[3]
                    race_num = str(f_name[-4])
                    currency = f_name[-2]
                    # time odds were posted
                    timestamp = to_datetime(f_name[-1], unit='s')
                    # second line of file
                    carry_in = float(lines[1][0])
                    pool_total = float(lines[1][1])
                    base_probable_stake = float(lines[1][2])
                    race_id = x8_track_sym + '_' + date.strftime('%Y%m%d') + '_' + race_num

                    # concat to master defaultdict
                    self.trifecta[race_id][timestamp]['carry_in'] = carry_in
                    self.trifecta[race_id][timestamp]['pool_total'] = pool_total
                    self.trifecta[race_id][timestamp]['base_probable_stake'] = base_probable_stake
                    # race level meta data
                    self.trifecta[race_id]['date'] = date
                    self.trifecta[race_id]['itsp_track_sym'] = itsp_track_sym
                    self.trifecta[race_id]['country'] = country
                    self.trifecta[race_id]['race_num'] = race_num
                    self.trifecta[race_id]['currency'] = currency
                    self.trifecta[race_id]['x8_track_sym'] = x8_track_sym

            except FileNotFoundError:
                # more specific message instead of generic FileNotFoundError
                warn('There are no trifecta odds files for %s yet. Skipping this day.' % d.date())

    def load_final_trifecta(self, datelist):
        """
        load final trifecta odds files - only contain pool size
        :param datelist: list of date objects, e.g. list(datetime(), datetime())
        :return: defaultdict of defaultdicts
        """
        # instantiate s3 file storage - must be inside functions so it's not static
        s3 = S3FileSystem(anon=False)

        for d in datelist:
            date_str = d.strftime('%Y-%m-%d')
            key = 'x8-bucket/odds/%s' % date_str
            try:
                s3_files = s3.ls(key) # list of all files in a given direcetory - in this case, all files for a single day
                # keep only files with 'TR' in name - show odds (this will include final_odds_* files)
                s3_files = [fp for fp in s3_files if basename(fp).find('_TR_') > -1]
                # keep only odds files, not final_odds
                s3_files = [fp for fp in s3_files if basename(fp).startswith('final')]
                # filter out files we have already read
                s3_files = list(set(s3_files) - set(self.files_read))
                self.files_read += s3_files
                if self.verbose:
                    print('ww.load_final_trifecta(%s) loading %s files..' % (date_str, len(s3_files)))
                for fp in s3_files:
                    # final_odds_CHD_THOROUGHBRED_USA_2018-05-01_4_TR_USD
                    f_name = splitext(splitext(basename(fp))[0])[0].split('_')
                    # list of lists of entire file
                    # using this method instead of read_csv because we cannot know which lines of an trifecta file
                    # to skip before reading because they are dependant on number of runners in a race
                    lines = [line.decode().replace('\n', '').split(',') for line in gzip.open(s3.open(fp, mode='rb'), 'rb')]
                    # meta data parsing
                    date = to_datetime(d.date())
                    itsp_track_sym = f_name[2]
                    x8_track_sym = self.map_itsp_to_x8.get(itsp_track_sym)
                    # if itsp_track_sym is missing from track_detail.csv then skip
                    if not x8_track_sym:
                        continue
                    country = f_name[4]
                    race_num = str(f_name[-3])
                    currency = f_name[-1]
                    # second line of file
                    carry_in = float(lines[1][0])
                    pool_total = float(lines[1][1])
                    base_probable_stake = float(lines[1][2])
                    race_id = x8_track_sym + '_' + date.strftime('%Y%m%d') + '_' + race_num

                    # concat to master defaultdict
                    self.final_trifecta[race_id]['carry_in'] = carry_in
                    self.final_trifecta[race_id]['pool_total'] = pool_total
                    self.final_trifecta[race_id]['base_probable_stake'] = base_probable_stake
                    # race level meta data
                    self.final_trifecta[race_id]['date'] = date
                    self.final_trifecta[race_id]['itsp_track_sym'] = itsp_track_sym
                    self.final_trifecta[race_id]['country'] = country
                    self.final_trifecta[race_id]['race_num'] = race_num
                    self.final_trifecta[race_id]['currency'] = currency
                    self.final_trifecta[race_id]['x8_track_sym'] = x8_track_sym

            except FileNotFoundError:
                # more specific message instead of generic FileNotFoundError
                warn('There are no final trifecta odds files for %s yet. Skipping this day.' % d.date())

    def load_superfecta(self, datelist):
        """
        load superfecta odds files - only contain pool size (same as trifecta file format)
        :param datelist: list of date objects, e.g. list(datetime(), datetime())
        :return: defaultdict of defaultdicts
        """
        # instantiate s3 file storage - must be inside functions so it's not static
        s3 = S3FileSystem(anon=False)

        for d in datelist:
            date_str = d.strftime('%Y-%m-%d')
            key = 'x8-bucket/odds/%s' % date_str
            try:
                s3_files = s3.ls(key) # list of all files in a given direcetory - in this case, all files for a single day
                # keep only files with 'SU' in name - show odds (this will include final_odds_* files)
                s3_files = [fp for fp in s3_files if basename(fp).find('_SU_') > -1]
                # keep only odds files, not final_odds
                s3_files = [fp for fp in s3_files if basename(fp).startswith('odds')]
                # filter out files we have already read
                s3_files = list(set(s3_files) - set(self.files_read))
                self.files_read += s3_files
                if self.verbose:
                    print('ww.load_superfecta(%s) loading %s files..' % (date_str, len(s3_files)))
                for fp in s3_files:
                    # odds_HUN_THOROUGHBRED_USA_2018-04-09_4_SU_USD_1523298412
                    f_name = splitext(splitext(basename(fp))[0])[0].split('_')
                    # list of lists of entire file
                    # using this method instead of read_csv because we cannot know which lines of a superfecta file
                    # to skip before reading because they are dependant on number of runners in a race
                    lines = [line.decode().replace('\n', '').split(',') for line in gzip.open(s3.open(fp, mode='rb'), 'rb')]
                    # meta data parsing
                    date = to_datetime(d.date())
                    itsp_track_sym = f_name[1]
                    x8_track_sym = self.map_itsp_to_x8.get(itsp_track_sym)
                    # if itsp_track_sym is missing from track_detail.csv then skip
                    if not x8_track_sym:
                        continue
                    country = f_name[3]
                    race_num = f_name[-4]
                    currency = f_name[-2]
                    # time odds were posted
                    timestamp = to_datetime(f_name[-1], unit='s')
                    # second line of file
                    carry_in = float(lines[1][0])
                    pool_total = float(lines[1][1])
                    base_probable_stake = float(lines[1][2])
                    race_id = x8_track_sym + '_' + date.strftime('%Y%m%d') + '_' + race_num

                    # concat to master defaultdict
                    self.superfecta[race_id][timestamp]['carry_in'] = carry_in
                    self.superfecta[race_id][timestamp]['pool_total'] = pool_total
                    self.superfecta[race_id][timestamp]['base_probable_stake'] = base_probable_stake
                    # race level meta data
                    self.superfecta[race_id]['date'] = date
                    self.superfecta[race_id]['itsp_track_sym'] = itsp_track_sym
                    self.superfecta[race_id]['country'] = country
                    self.superfecta[race_id]['race_num'] = race_num
                    self.superfecta[race_id]['currency'] = currency
                    self.superfecta[race_id]['x8_track_sym'] = x8_track_sym

            except FileNotFoundError:
                # more specific message instead of generic FileNotFoundError
                warn('There are no superfecta odds files for %s yet. Skipping this day.' % d.date())

    def load_entries(self, datelist, tracklist=[]):
        """
        development stage - load all entries data for a day into df
        :param datelist: list of date objects, e.g. list(datetime(), datetime())
        """
        # instantiate s3 file storage - must be inside functions so it's not static
        s3 = S3FileSystem(anon=False)

        # normalize tracklist
        tracklist = Series(tracklist).map(self.map_x8_to_itsp)
        if tracklist.isnull().any():
            raise ValueError('param tracklist must be list of x8 track symbols in track_detail: \n%s' % tracklist)

        # only loading entries by dates, not tracks - loading all data per day
        for d in datelist:
            # for dealing with case where there's no races in a day
            date_str = d.strftime('%Y-%m-%d')
            key = 'x8-bucket/entries/%s' % date_str
            try:
                s3_files = s3.ls(key) # list of all files in a given direcetory - in this case, all files for a single day

                # filter s3_files so that only one file for each race is read, the most recently updated file
                filter = DataFrame({'fp': s3_files})
                filter['timestamp'] = filter['fp'].map(lambda x: to_datetime(splitext(splitext(x)[0])[0].split('_')[-1], unit='s'))
                filter['race'] = filter['fp'].map(lambda x: x[:x.rfind('_')])
                # filter tracks
                if tracklist.any():
                    filter['itsp_track_sym'] = filter['fp'].map(lambda x: x.split('_')[1])
                    filter = filter[filter['itsp_track_sym'].isin(tracklist)]
                newest = filter.groupby('race')['timestamp'].transform('max')
                filter = filter[filter['timestamp'] == newest]
                s3_files = list(filter['fp'])

                # filter out files we have already read
                s3_files = list(set(s3_files) - set(self.files_read))
                self.files_read += s3_files
                if self.verbose:
                    print('ww.load_entries(%s) loading %s files..' % (date_str, len(s3_files)))
                for fp in s3_files:
                    f_name = splitext(splitext(basename(fp))[0])[0] # 'entries_PHD_THOROUGHBRED_USA_2018-05-15_1_1526372378'
                    post_time = read_csv(s3.open(fp, mode='rb'), compression='gzip', skiprows=lambda x: x!=0, header=None).iloc[0][5] # only read 1st row
                    # skip first row of csv which includes same data in the file name except for new column 'post_time'
                    df = read_csv(s3.open(fp, mode='rb'), compression='gzip', skiprows=1, header=None) # skip 1st row
                    df['date'] = to_datetime(d.date())
                    df['post_time'] = to_datetime(str(post_time), unit='s') # convert post_time to string and then to timestamp
                    df['timestamp'] = to_datetime(f_name.split('_')[-1], unit='s')  # '1526372378' -> Timestamp('2018-05-15 08:19:38')
                    df['itsp_track_sym'] = f_name.split('_')[1] # 'PHD'
                    df['country'] = f_name.split('_')[3] # 'USA'
                    df['race_num'] = f_name.split('_')[5] # '1'
                    df['track_type'] = df.iloc[0][0] # 'DIRT'
                    df['distance'] = df.iloc[0][1] # '5 1/2F'
                    df['race_type'] = df.iloc[0][2] # 'MAIDEN CLAIMING'
                    df['conditions'] = df.iloc[0][3] # 'FOR MAIDENS, TWO YEARS OLD. Weight, 122 lbs. Claiming Price $10,000 [Ohio Registered Foals Preferred]. '
                    # extra field at iloc[0][4] that is unused
                    df['prize_currency'] = df.iloc[0][5] # 'USD'
                    df['prize_money'] = df.iloc[0][6] # '12000.00'
                    df = df.iloc[1:] # drop first row
                    # rename 0-7 column names
                    df.rename(columns=entries_columns, inplace=True)
                    # concat to master entries df
                    self.df_entries = concat([self.df_entries, df])
            except FileNotFoundError:
                # we think should only throw on 2017-12-25 and before 2017-09-01
                warn('There are no entries files for %s yet. Skipping this day.' % d.date())

        if self.df_entries.empty:
            return 0

        # track symbol identifiers
        if datetime(2018, 11, 3) not in datelist:
            self.map_itsp_to_x8['CHD'] = 'CDX'

        self.df_entries['x8_track_sym'] = self.df_entries['itsp_track_sym'].map(self.map_itsp_to_x8)

        # drop any rows where itsp track symbol mapping is missing from track detail if any
        x8_isnull = self.df_entries['x8_track_sym'].isnull()
        if x8_isnull.any():
            missing_itsp_symbols = self.df_entries[x8_isnull]['itsp_track_sym'].unique()
            warn('ww.load_entries() track_detail.csv is missing itsp symbols: %s\nDropping all rows with missing symbols' % missing_itsp_symbols)
            self.df_entries = self.df_entries[~x8_isnull]
            print('ww.load_entries() dropping %s rows\n' % x8_isnull.sum())

        # normalize horse name
        self.df_entries['x8name'] = self.df_entries['horse_name'].map(self._normalize_name)
        self.df_entries['x8jockey'] = self.df_entries['jockey_name'].map(self._normalize_name)
        self.df_entries['x8trainer'] = self.df_entries['trainer_name'].map(self._normalize_name)

        # race / runner identifiers
        self.df_entries['race_id'] = self.df_entries['x8_track_sym'] + '_' + self.df_entries['date'].dt.strftime('%Y%m%d') + '_' + self.df_entries['race_num']
        self.df_entries['runner_id'] = self.df_entries['race_id'] + '_' + self.df_entries['program_number']

        # dtype conversions
        self.df_entries['prize_money'] = self.df_entries['prize_money'].map(float)
        self.df_entries['birth_year'] = self.df_entries['birth_year'].fillna(0).map(int)

        # num_starters
        self.df_entries['num_starters'] = self.df_entries.groupby('race_id')['runner_id'].transform('size')
        self.df_entries['num_scratched_live'] = self.df_entries.groupby('race_id')['scratched'].transform('sum')
        self.df_entries['num_starters_live'] = self.df_entries['num_starters'] - self.df_entries['num_scratched_live']

        # this is to count the number of bettable runners which is the correct number when dealing with fair odds
        self.df_entries['is_bettable'] = (self.df_entries['betting_interest'] == self.df_entries['program_number']).map(int)

        # df_entries constraint: 1 row per runner
        # only keep newest data for each race
        newest = self.df_entries.groupby('race_id')['timestamp'].transform('max')
        self.df_entries = self.df_entries[self.df_entries['timestamp'] == newest]

        self._validate_entries()

    def load_results(self, datelist, tracklist=[], filter_cancellations=True):
        """
        development stage - load all results data for a day into df
        :param datelist: list of datetime objects, e.g. list(datetime(), datetime())
        """
        # instantiate s3 file storage - must be inside functions so it's not static
        s3 = S3FileSystem(anon=False)

        # convert tracklist symbols to itsp track symbols
        tracklist = Series(tracklist).map(self.map_x8_to_itsp)
        if tracklist.isnull().any():
            raise ValueError('param tracklist must be list of x8 track symbols in track_detail: \n%s' % tracklist)

        # only loading results by dates, not tracks - loading all data per day
        for d in datelist:
            date_str = d.strftime('%Y-%m-%d')
            key = 'x8-bucket/results/%s' % date_str
            try:
                # list of all files in a given direcetory - in this case, all files for a single day - this includes results_ and finishers_
                s3_files = s3.ls(key)
                # filter for only results (not finishers)
                s3_files = [fp for fp in s3_files if basename(fp).startswith('results')]
                # filter tracklist
                if not tracklist.empty:
                    s3_files = [fp for fp in s3_files if fp.split('_')[1] in list(tracklist)]
                # filter out files we have already read
                s3_files = list(set(s3_files) - set(self.files_read))
                self.files_read += s3_files
                if self.verbose:
                    print('ww.load_results(%s) loading %s files..' % (date_str, len(s3_files)))
                for fp in s3_files:
                    f_name = splitext(splitext(basename(fp))[0])[0] # 'results_TPD_THOROUGHBRED_USA_2018-01-13_8'
                    # skip first row of csv which includes same data in the file name
                    df = read_csv(s3.open(fp, mode='rb'), compression='gzip', skiprows=1, header=None)
                    df['date'] = to_datetime(d.date())
                    df['itsp_track_sym'] = f_name.split('_')[1] # 'TPD'
                    df['country'] = f_name.split('_')[3] # 'USA'
                    df['race_num'] = f_name.split('_')[-1] # '8'
                    # rename 0-6 column names
                    df.rename(columns=results_columns, inplace=True)
                    # concat to master results df
                    self.df_results = concat([self.df_results, df])
            except FileNotFoundError:
                if d == datetime(2018, 3, 6):
                    warn('Excepting FileNotFoundError for {0} - There is no WaW Feed Data for this day - see '
                         'https://mail.google.com/mail/u/0/#search/apegg%40watchandwager.com/1612eea4015c6ee3'.format(d.date()))
                elif d < datetime(2017, 10, 2):
                    warn('Excepting FileNotFoundError for %s - We did not start receiving results data until 2019-10-02' % d.date())
                elif datetime(2017, 10, 28) < d < datetime(2017, 11, 3):
                    warn('date hole')
                else:
                    raise FileNotFoundError('No results files for %s' % d.date())

        # make raw df
        self.df_results_raw = self.df_results.copy()

        if self.df_results_raw.empty:
            raise FileNotFoundError('No Files for dates: %s and tracks: %s' % (datelist, tracklist))

        # track symbol conversion
        self.df_results['x8_track_sym'] = self.df_results['itsp_track_sym'].map(self.map_itsp_to_x8)

        # drop any rows where itsp track symbol mapping is missing from track detail if any
        x8_isnull = self.df_results['x8_track_sym'].isnull()
        if x8_isnull.any():
            missing_itsp_symbols = self.df_results[x8_isnull]['itsp_track_sym'].unique()
            warn('ww.load_results() track_detail.csv is missing itsp symbols: %s\nDropping all rows with missing symbols' % missing_itsp_symbols)
            self.df_results = self.df_results[~x8_isnull]
            print('ww.load_results() dropping %s rows\n' % x8_isnull.sum())

        # race identifier
        self.df_results['race_id'] = self.df_results['x8_track_sym'] + '_' + self.df_results['date'].dt.strftime('%Y%m%d') + '_' + self.df_results['race_num']

        # payouts with base_stake == 1.0
        self.df_results['payout_norm'] = (self.df_results['payout'] / self.df_results['base_stake']).fillna(0.0)

        if self.verbose:
            print('ww.load_results(): Filtering out consolation bet rows from results. "BO" in winning_pgm col.')
        # filter out consolation bet rows
        self.df_results = self.df_results[self.df_results['winning_pgm'] != ' BO']

        if filter_cancellations:
            print('ww.load_results(): Filtering out cancellations (%s rows where winning_pgm is null)' % self.df_results['winning_pgm'].isnull().sum())
            self.df_results = self.df_results[~self.df_results['winning_pgm'].isnull()]

        # amtote expects multi-runner bets 'runners' to be formatted with ',' but ww.results formats winning_pgm runners with '/'
        self.df_results['winning_pgm'] = self.df_results['winning_pgm'].map(str)
        self.df_results['winning_pgm'] = self.df_results['winning_pgm'].map(lambda x: x.replace('/', ','))

        # pool signal columns
        self.df_results['is_single_leg'] = self.df_results['bet_type'].map(lambda x: 1 if x in pool_single else 0)
        self.df_results['is_multi_runner'] = self.df_results['bet_type'].map(lambda x: 1 if x in pool_multi_runner else 0)
        self.df_results['is_multi_race'] = self.df_results['bet_type'].map(lambda x: 1 if x in pool_multi_race else 0)

        # TODO test and activate multi-race race_id adapting code that was copied from jcapper.py
        # for multi race pools, change race_id from ending race_id to starting race_id so it can merge w/ df_bets
        self.df_results['race_id'] = self.df_results.apply(lambda x: '_'.join(
            x['race_id'].split('_')[:2] + [str(int(x['race_id'].split('_')[-1]) - minus_num_races[x['bet_type']])]) if x[
            'is_multi_race'] else x['race_id'], axis=1)

        self._validate_results()

    def load_scratches(self, datelist, tracklist=[], filter_newest=False):
        """
        load scratches data for a day into df
        concerning 's3://x8-bucket/changes/changes_*' files
        3 parts to changes files (see README) changes, going, scratches - for now only reading scratches
        :param datelist: list of datetime objects, e.g. list(datetime(), datetime())
        """
        # instantiate s3 file storage - must be inside functions so it's not static
        s3 = S3FileSystem(anon=False)

        # normalize tracklist
        tracklist = Series(tracklist).map(self.map_x8_to_itsp)
        if tracklist.isnull().any():
            raise ValueError('param tracklist must be list of x8 track symbols in track_detail: \n%s' % tracklist)

        # only loading scratches by dates, not tracks - loading all data per day
        for d in datelist:
            date_str = d.strftime('%Y-%m-%d')
            key = 'x8-bucket/changes/%s' % date_str
            try:
                s3_files = s3.ls(key)  # list of all files in a given directory

                # TODO KEEP filter_newest=False until "re-entered" ww bug fixed. see https://mail.google.com/mail/u/0/#sent/KtbxLthVZwxdKMRTkDLHchzvdPgFBplqdq?compose=DmwnWrRsnxDTvgTBhnfmRPSrrkLhmKtRKWMSvJGhRjmSqRTKgsdlwGLZBQVScrNNbZDQBsqGbzqG
                # this is mainly a speed issue right now because if we don't filter and keep newest (which is slow) then we get all the scratches correctly
                if filter_newest:
                    # filter s3_files so that only one file for each race is read, the most recently updated file
                    filter = DataFrame({'fp': s3_files})
                    filter['timestamp'] = filter['fp'].map(lambda x: to_datetime(splitext(splitext(x)[0])[0].split('_')[-1], unit='s'))
                    filter['race'] = filter['fp'].map(lambda x: x[:x.rfind('_')])
                    # filter tracks
                    if tracklist.any():
                        filter['itsp_track_sym'] = filter['fp'].map(lambda x: x.split('_')[1])
                        filter = filter[filter['itsp_track_sym'].isin(tracklist)]
                    newest = filter.groupby('race')['timestamp'].transform('max')
                    filter = filter[filter['timestamp'] == newest]
                    s3_files = list(filter['fp'])

                # filter out files we have already read
                s3_files = list(set(s3_files) - set(self.files_read))
                self.files_read += s3_files
                if self.verbose:
                    print('ww.load_scratches(%s) loading %s files..' % (date_str, len(s3_files)))
                for fp in s3_files:
                    # 'changes_OPM_THOROUGHBRED_USA_2018-01-13_6_1515878857'
                    f_name = splitext(splitext(basename(fp))[0])[0]
                    file = [line for line in gzip.open(s3.open(fp, mode='rb'), 'rb')][-1]
                    # check if scratches exist in changes file - if no scratches (last line) in file there is b'\n'
                    if file != b'\n':
                        df = DataFrame({'program_number_scratch': file.decode()[:-1].split(',')})
                        df['date'] = to_datetime(d.date())
                        df['itsp_track_sym'] = f_name.split('_')[1]  # 'OPM'
                        df['country'] = f_name.split('_')[3]  # 'USA'
                        df['race_num'] = f_name.split('_')[-2]  # '6'
                        df['timestamp'] = to_datetime(f_name.split('_')[-1], unit='s')  # '1515878857' -> Timestamp('2018-01-13 21:27:37')
                        # concat to master scratches df
                        self.df_scratches = concat([self.df_scratches, df])
            except FileNotFoundError:
                warn('There are no changes files for %s yet. Skipping this day.' % d.date())

        # for case when there is no data for dates trying to load, this could be because of no changes files or we have changes files but no scratches in them
        if not self.df_scratches.empty:
            # track symbol conversion
            self.df_scratches['x8_track_sym'] = self.df_scratches['itsp_track_sym'].map(self.map_itsp_to_x8)

            # drop any rows where itsp track symbol mapping is missing from track detail if any
            x8_isnull = self.df_scratches['x8_track_sym'].isnull()
            if x8_isnull.any():
                missing_itsp_symbols = self.df_scratches[x8_isnull]['itsp_track_sym'].unique()
                warn('ww.load_scratches() track_detail.csv is missing itsp symbols: %s\nDropping all rows with missing symbols' % missing_itsp_symbols)
                self.df_scratches = self.df_scratches[~x8_isnull]
                print('ww.load_scratches() dropping %s rows\n' % x8_isnull.sum())

            # race / runner identifiers
            self.df_scratches['race_id'] = self.df_scratches['x8_track_sym'] + '_' + self.df_scratches['date'].dt.strftime('%Y%m%d') + '_' + self.df_scratches['race_num']
            # TODO validate this program_number aligns with program_number / post_position on other data sources
            self.df_scratches['runner_id'] = self.df_scratches['race_id'] + '_' + self.df_scratches['program_number_scratch']

            # df_scratches constraint: 1 row per runner
            # keep rows with earliest timestamp so it is clear when the scratch occured
            if not filter_newest:
                earliest = self.df_scratches.groupby('runner_id')['timestamp'].transform('min')
                self.df_scratches = self.df_scratches[self.df_scratches['timestamp'] == earliest]

            self._validate_scratches()

    def _normalize_name(self, name):
        # this used to be called make_canonical_name in main, and is unchanged
        canonical_name = str(name).upper().replace("'", "").strip()
        ending = canonical_name.find("(")
        if ending > -1:
            canonical_name = canonical_name[0:ending].strip().replace(" ", "")
        return canonical_name.replace(" ", "")

    def _validate_win_odds(self):
        """check that the win odds DataFrame values are as expected"""
        # assert that the sum of probs_odds_norm by race group is equal to close to 1.0
        sum_probs_odds_norm = self.df_win_odds.groupby(['race_id', 'ordinal_timestamp'])['prob_market_odds'].sum()
        if not sum_probs_odds_norm.map(lambda x: isclose(x, 1.0, rel_tol=.0001)).all():
            raise Exception('Sum of normalized probability odds by race is not equal to close to 1.0')

    def _validate_final_win_odds(self):
        """check that the final win odds DataFrame values are as expected"""
        sum_probs_odds_norm = self.df_final_win_odds.groupby('race_id')['prob_market_odds'].sum()
        if not sum_probs_odds_norm.map(lambda x: isclose(x, 1.0, rel_tol=.0001)).all():
            raise Exception('Sum of normalized probability odds by race is not equal to close to 1.0')

    def _validate_exacta(self):
        """check that the exacta default dict values are as expected"""
        # check probs are normalized
        for race_id in self.exacta.keys():
            assert isclose(self.exacta[race_id][1]['probs'].sum(), 1)
            if 2 in self.exacta[race_id].keys():
                assert isclose(self.exacta[race_id][2]['probs'].sum(), 1)
            if 3 in self.exacta[race_id].keys():
                assert isclose(self.exacta[race_id][3]['probs'].sum(), 1)

        for race_id in self.exacta.keys():
            ots1 = self.exacta[race_id][1]['timestamp'].unique()[0]
            if 2 in self.exacta[race_id].keys():
                ots2 = self.exacta[race_id][2]['timestamp'].unique()[0]
                assert ots2 > ots1
            if 3 in self.exacta[race_id].keys():
                ots3 = self.exacta[race_id][3]['timestamp'].unique()[0]
                assert ots3 > ots2 > ots1

    def _validate_entries(self):
        """check that the entries DataFrame values are as expected"""
        # entries files should only have 1 row per runner unlike odds
        if len(self.df_entries) != len(self.df_entries.runner_id.unique()):
            raise Exception('WatchandWager.load_entries(): len(df_entries) != len(df_entries.runner_id.unique())')

        # check if timestamp col was parsed properly - greater than first day we started receiving real-time data
        assert (self.df_entries['timestamp'] > datetime(2017, 9, 1)).all(), 'timestamps are being parsed incorrectly'

    def _validate_results(self):
        """check that the results DataFrame values are as expected"""
        pass

    def _validate_scratches(self):
        """check that the scratches DataFrame values are as expected"""
        if len(self.df_scratches) != len(self.df_scratches.runner_id.unique()):
            raise Exception('WatchandWager.load_scratches(): len(df_scratches) != len(df_scratches.runner_id.unique())')
