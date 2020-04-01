# wrapper for a consolidated view of all the race data

from pandas import DataFrame, Series, MultiIndex, merge, read_csv, pivot_table, concat, to_datetime
from horse.betsim.wrap.guaranteedtipsheet import GuaranteedTipSheet
from horse.betsim.wrap.jcapper import JCapper
from horse.betsim.wrap.pastperformance import PastPerformance
from horse.betsim.wrap.watchandwager import WatchandWager
from horse.betsim.wrap.factorfactory import FactorFactory, cols_x8race_class
from horse.betsim.wrap.bets import Bets
from horse.betsim.math import mean_best_N_of_K, compute_probs_from_odds, add_probsT4
from horse.betsim import data
from horse.betsim.data import attr_map
from numpy import multiply, nan
from scipy.stats import entropy
from horse.betsim.models.probability import add_win_probs_from_score
from time import time
import pickle
import os
import string


translator = str.maketrans('','',string.punctuation)


def clean_text_column(df, colname):
    """removes punctuation and white space from a column"""
    df[colname] = df.loc[:, colname].apply(lambda s: s.translate(translator))
    df[colname] = df.loc[:, colname].apply(lambda s: s.lower())
    df[colname] = df.loc[:, colname].apply(lambda s: s.lstrip())
    return df


def extract_datelist(dailyraces, attr_date='date_entries'):
    '''Returns a datelist from a DailyRaces using the exact dates'''
    return list(dailyraces.df[attr_date].map(lambda x: x.date()).unique())


class DailyRaces(object):
    """
    DailyRaces has evolved into an agnostic data consolidator.
        3 states:
        'pre' (before races start): [pp, gts]
        'play' (intra day / during races): pre + [ww.df_entries, ww.df_scratches, ww.df_win_odds]
        'post' (after races finish): pre + play + [jcp, ww.df_final_win_odds, ww.df_results]

    By designing the class this way, we can use this class for simulating strategies as if you are in-play or as if you are pre-race
    df: DataFrame containing a merge of various sources: gts, jcapper, scratches, odds, final_odds, etc
    """
    def __init__(self, state, verbose=False, trading=False):
        """

        :param state: pre/play/post/historical
        :param verbose:
        :param trading: if True, do not drop pp tracks that are not in entries tracklist. include all tracks in pp. this is
                 used for trading.py files. Can only be trading=True if state=pre
        """
        # processed df - for this class the dfraw is accessible inside each of the classes that it is loading e.g. self.jcp.dfraw
        self.df = DataFrame()
        self.dfX = DataFrame()

        # bets object
        self.bets = Bets()

        # dict of Race() objects by race_id {race_id: Race()}
        self.races = {}

        # wrappers for raw data
        self.gts = GuaranteedTipSheet()
        self.jcp = JCapper(verbose=verbose)
        self.pp = PastPerformance(verbose=verbose)
        self.ww = WatchandWager(verbose=verbose)
        self.ff = FactorFactory()

        # datelist needs to be here because update()
        self.datelist = []
        self.tracklist = []

        # state
        self.state = state
        # possible values for state are 'pre', 'in' and 'post'
        if self.state not in ['pre', 'play', 'post', 'historical']:
            raise ValueError('"%s" is not a usable state. Possible states are: ["pre", "play", "post"]' % state)
        self.trading = trading
        if self.trading:
            print('trading=True - using pp tracks (ignoring waw tracks)')
            if self.state != 'pre':
                print('if trading=True then state must equal "pre" - setting state="pre"')
                self.state = 'pre'

        # attribute mapping stuff
        fp_attr_map = os.path.join(attr_map.__path__._path[0], '%s.csv' % state)
        self.default_attr_map = read_csv(fp_attr_map).set_index('new_name')['old_name'].to_dict()

        # moving this to wrapper units
        # now getting track detail exclusively from git file (horse/betsim/data/track_detail.csv) instead of relative path from where data is being loaded
        track_detail = os.path.join(data.__path__._path[0], 'track_detail.csv')
        self.dftrack = read_csv(track_detail)

        # for normalizing output df - always produces same tracks (entries)
        self.bettable_tracks = []

        # print statements
        self.verbose = verbose

    def load(self, datelist, tracklist=[]):
        """
        load the required dataframes for given state
        """

        # make datelist and tracklist global for update() method
        self.datelist = datelist
        self.tracklist = tracklist

        if self.state == 'historical':
            self._load_history()
            return 0

        # load 'pre' df's
        if self.verbose:
            print('loading gts and pp..')
        # load past performance
        self.pp.load(datelist, tracklist)

        # always load entries because we will use it for filtering tracks
        self.ww.load_entries(datelist, tracklist)
        if self.ww.df_entries.empty or self.trading:
            self.bettable_tracks = self.pp.df['x8_track_sym'].unique()
        else:
            self.bettable_tracks = self.ww.df_entries['x8_track_sym'].unique()

        # we only have gts data (on s3) since 2017-08-31
        try:
            # load gts
            self.gts.load(datelist)
        except FileNotFoundError:
            print('\nNo GTS data for the dates %s to %s. Continuing without GTS.' % (datelist[0].date(), datelist[-1].date()))

        # if state is 'play' or 'post' then load 'play' df's
        if self.state in ['play', 'post']:
            if self.verbose:
                print('loading ww.df_scratches and ww.df_win_odds..')
            # load waw scratches
            self.ww.load_scratches(datelist)
            # load waw win odds
            self.ww.load_win_odds(datelist, tracklist)
            # if state is 'post' then load 'post' df's
            if self.state == 'post':
                if self.verbose:
                    print('loading jcp, ww.df_results and ww.df_final_win_odds..')
                # load jcapper / charts
                self.jcp.load(datelist, tracklist)
                # load waw results
                #self.ww.load_results(datelist)
                # load waw final win odds
                self.ww.load_final_win_odds(datelist)

        # computed columns
        self.pp.add_computed_columns()
        if not self.jcp.df.empty:
            self.jcp.add_computed_columns()

        # merge all into master df
        self._merge_data()

        # factors
        self.add_computed_columns()

        # label and make Race objects
        self.label()

        # validate
        self._validate()

    def _load_history(self):
        """
        load historical pp/jcp data
        """

        # TODO pre-select tracklist from s3

        # load data
        self.pp.load(self.datelist, self.tracklist)
        self.pp.add_computed_columns()
        self.jcp.load(self.datelist, self.tracklist)
        self.jcp.add_computed_columns()

        target_races = set(self.pp.df.race_id.unique()).intersection(set(self.jcp.df.race_id.unique()))

        on_cols = ['runner_id', 'race_id']

        # pp.df key: has every runner_id per race
        self.df = self._rename(self.pp.df, '_pp')
        self.df = self.df[self.df['race_id'].isin(target_races)].copy()

        # jcp.df key: has every runner_id per race
        jcp = self._rename(self.jcp.df, '_jcp')
        jcp = jcp[jcp['race_id'].isin(target_races)].copy()

        self.df = merge(self.df, jcp, on=on_cols, how='left')

        self.add_computed_columns()

        self.label()

        # compute pre/post probs i.e. probs with all runners and then re-normalized
        # self.dfX = self.dfX.groupby('race_id').apply(lambda dfrace: add_win_probs_from_score(dfrace, 'runner_HDWPSRRating'))
        # self.dfX['entropy_prob_runner_HDWPSRRating'] = self.dfX.groupby('race_id')['prob_runner_HDWPSRRating'].transform(lambda x: entropy(x, base=len(x)))
        # self.dfX['entropy_prob_runner_HDWPSRRating'] = self.dfX['entropy_prob_runner_HDWPSRRating'].map(lambda x: nan if isneginf(x) else x)
        # self.dfX['rank_prob_runner_HDWPSRRating'] = self.dfX.groupby('race_id')['prob_runner_HDWPSRRating'].rank(ascending=False)

    def update(self, prob_model='all'):
        # re-load all non-static data, namely ww odds/changes/entries
        # re-merge the new data w/ static data (pp) to produce new updated master df

        if self.state == 'play':
            if self.verbose:
                print('updating ww.df_entries, ww.df_scratches, ww.df_win_odds..')
            # load waw entries
            self.ww.load_entries(self.datelist, self.tracklist)
            # load waw scratches
            self.ww.load_scratches(self.datelist)
            # load waw win odds
            self.ww.load_win_odds(self.datelist, self.tracklist)
            # re-merge to new master df

        # columns
        self.add_computed_columns()

        # label dfX
        self.label()

        # filter scratches
        num = len(self.dfX)
        if self.trading:
            scratched = self.ww.df_entries[self.ww.df_entries['scratched']]['runner_id']
            self.dfX = self.dfX[~self.dfX['runner_id'].isin(scratched)]
        elif self.state == 'historical':
            self.dfX = self.dfX[self.dfX['final_tote_odds'].notnull()]
        else:
            self.dfX = self.dfX[self.dfX['scratch'].isnull()]
        print('filtered %s runners from dr.dfX' % (num - len(self.dfX)))

        # re-normalize probabilities after filtering scratches
        self._normalize_probs()

        # probs
        self.compute_probs(prob_model=prob_model)

        # validate
        self._validate()

    def _rename(self, df, suffix):
        """
        internal function to help with this objective:

        rename all columns from each data source w/ respective suffix that indicates which source the data came from
        (e.g. '_pp' for PastPerformance) so that when the data is merged to create the master df the same columns
        are output every time and the master df can be handled in the same way by external code, regardless of the
        chosen state

        note - 'runner_id' does not get renamed because it is being used as key for merging data

        this method works by passing a df param and the accompanied suffix param which should indicate the data source
        e.g. _rename(jcp.df, '_jcp')

        :return df with added suffixes to columns
        """
        df = df.copy()
        df.columns = [col + suffix if col not in ['runner_id', 'race_id'] else col for col in df.columns]
        return df

    def _merge_data(self):
        """
        internal function to merge columns from various DataFrames into a single df

        must be merged on runner_id w/ program number only - otherwise df's won't align e.g. '1A'
        """

        if self.verbose:
            print('merging dataframes..')

        on_cols = ['runner_id', 'race_id']

        # pp.df key: has every runner_id per race
        df = self.pp.df.copy()
        df = self._rename(df, '_pp')

        # filter out non-bettable tracks (all states of dr produce same tracks)
        df = df[df['x8_track_sym_pp'].isin(self.bettable_tracks)]

        # possible that gts.df is empty and not broken, see self.load()
        if not self.gts.df.empty:
            # gts key: has runner_id but only max 6 picks (runners/rows) per race
            # using keys from pp.df
            gts = self.gts.df.dropna(subset=['program_number']).copy() # get rid of rows in gts that are empty
            gts = self._rename(gts, '_gts')
            df = merge(df, gts, on=on_cols, how='left')

        # ww.df_entries key: has every runner_id per race
        # using keys from ww.df_entries, which will act as filter of races we can bet on because ww.df_entries
        # using 'inner' merge because the intersection of runers/tracks between entries and pp are tracks that we actually want to bet..
        # entries includes all tracks that we can bet on but sometimes includes QH tracks that we do not have in pp - we only want to bet on TB tracks
        if self.ww.df_entries.empty or self.trading:
            self.df = df
            self.df = self.df.reset_index(drop=True)
            return 0
        entries = self.ww.df_entries.copy()
        entries = self._rename(entries, '_entries')
        df = merge(df, entries, on=on_cols, how='inner')

        # if state is 'play' or 'post' then merge 'play' df's to master
        if self.state in ['play', 'post']:

            # for case when it is loading today data and win odds haven't come in yet
            if not self.ww.df_win_odds.empty:
                # ww.df_win_odds key: has every runner_id per race w/ scratched runners nan values
                # entries will almost always have more unique runners than win_odds during state 'play' because we get odds
                # at mtp 5,3,1 so until 5 mins before race we will have no win_odds for runners
                win_odds = self.ww.df_win_odds.copy()
                # this line produces most recent odds profile, so that there is only 1 runner per row in all states
                win_odds = win_odds[win_odds['max_ots'] == win_odds['ordinal_timestamp']]
                win_odds = self._rename(win_odds, '_win')
                # ww odds do not supply extra row for a 1A runner because they are the same odds as the 1
                # therefore, we do not want to merge on runner_id w/ 1A at the end but betting_interest_entries
                # which is consistent between 1A and 1 runner_id
                df = merge(df, win_odds,
                           left_on=['race_id', 'betting_interest_entries'],
                           right_on=['race_id', 'betting_interest_win'],
                           how='left')

                # re-normalize market probs after assigning 1A runners equal probs as 1
                df['prob_market_odds_win'] = df.groupby('race_id')['market_odds_win'].transform(compute_probs_from_odds)

            # for case when it is loading today data and scratches haven't come in yet
            if self.ww.df_scratches.empty:
                df['ww_scratched'] = nan
            else:
                # ww.df_scratches key: has runner_id but only includes runners that are scratched
                map_ww_scratched = self.ww.df_scratches.set_index('runner_id')['timestamp'].to_dict()
                df['ww_scratched'] = df['runner_id'].map(map_ww_scratched)

            # if state is 'post' then load 'post' df's
            if self.state == 'post':
                # jcp.df key: has every runner_id per race
                jcp = self._rename(self.jcp.df, '_jcp')
                df = merge(df, jcp, on=on_cols, how='left')

                if not self.ww.df_final_win_odds.empty:
                    # ww.df_final_win_odds key: has every runner_id per race
                    win_final = self._rename(self.ww.df_final_win_odds, '_win_final')
                    # ww odds do not supply extra row for a 1A runner because they are the same odds as the 1
                    # therefore, we do not want to merge on runner_id w/ 1A at the end but betting_interest_entries
                    # which is consistent between 1A and 1 runner_id
                    df = merge(df, win_final,
                               left_on=['race_id', 'betting_interest_entries'],
                               right_on=['race_id', 'betting_interest_win_final'],
                               how='left')

                    # re-normalize market probs after assigning 1A runners equal probs as 1
                    df['prob_market_odds_win_final'] = df.groupby('race_id')['market_odds_win_final'].transform(compute_probs_from_odds)

        self.df = df

    def add_computed_columns(self):
        """
        add pp group factors per runner
        """
        MAP_FINISH_PCTPURSE = {1: .6,
                               2: .25,
                               3: .10,
                               4: .035,
                               5: .015}

        factors_pp_speed_HDW = [c for c in self.df.columns if c.find('speed_HDW') > -1]
        factors_pp_speed_DRF = [c for c in self.df.columns if c.find('speed_DRF') > -1]

        x8race_class_cols = ['x8_track_sym_pp',
                             'race_race_type_pp',
                             'race_distance_pp',
                             'race_surface_pp',
                             'race_age_sex_restriction_pp']

        factors_pp_call_first = [c for c in self.df.columns if c.find('pp_call_first_pos') > -1]
        factors_pp_call_second = [c for c in self.df.columns if c.find('pp_call_second_pos') > -1]
        factors_pp_call_stretch = [c for c in self.df.columns if c.find('pp_call_stretch_pos') > -1]
        factors_pp_call_finish = [c for c in self.df.columns if c.find('pp_call_finish_pos') > -1]

        self.df.loc[:, factors_pp_call_first].fillna(0, inplace=True)
        self.df.loc[:, factors_pp_call_second].fillna(0, inplace=True)
        self.df.loc[:, factors_pp_call_stretch].fillna(0, inplace=True)
        self.df.loc[:, factors_pp_call_finish].fillna(0, inplace=True)

        self.df[['is_lead_finish' + f for f in factors_pp_call_finish]] = self.df[factors_pp_call_finish].applymap(lambda x: int(x < 2))
        self.df[['is_lead_first' + f for f in factors_pp_call_first]] = self.df[factors_pp_call_first].applymap(lambda x: int(x < 2))
        self.df[['is_lead_second' + f for f in factors_pp_call_second]] = self.df[factors_pp_call_second].applymap(lambda x: int(x < 2))
        self.df[['is_lead_stretch' + f for f in factors_pp_call_stretch]] = self.df[factors_pp_call_stretch].applymap(lambda x: int(x < 2))

        self.df[factors_pp_call_first] = self.df[factors_pp_call_first].fillna(0)
        self.df[factors_pp_call_second] = self.df[factors_pp_call_second].fillna(0)
        self.df[factors_pp_call_stretch] = self.df[factors_pp_call_stretch].fillna(0)
        self.df[factors_pp_call_finish] = self.df[factors_pp_call_finish].fillna(0)

        x8race_class_dict = self.df.groupby('race_id').apply(lambda df: tuple(df[x8race_class_cols].iloc[0])).to_dict()
        self.df['x8race_class'] = self.df['race_id'].map(x8race_class_dict)

        def gen_ppnames_dr(factor='speed_HDW', prefix='pp', suffix='pp'):
            """
            generates a list of pp column names for reading or adding
            i.e. if I know the factor name I can generate the column names
            i.e. if I want to add columns I can generate the new names in a similar format
            :param factor: 'speed_HDW' (a factor_name from DailyRaces pp  ie. 'speed_HDW)
            :param prefix: 'pp'
            :param suffix: 'pp'
            :return: ['pp_speed_HDW_0_pp', 'pp_speed_HDW_1_pp', ..., pp_speed_HDW_9_pp']
            """

            namespp = [prefix + "_" + factor + "_" + str(pp) + "_" + suffix for pp in range(10)]
            return namespp

        self.df[gen_ppnames_dr('x8_pctofpurse')] = self.df[gen_ppnames_dr('call_finish_pos')].applymap(lambda fp: MAP_FINISH_PCTPURSE.get(fp, 0))

        self.df[gen_ppnames_dr('x8_prob_odds')] = self.df[gen_ppnames_dr('odds')].applymap(lambda x: 1.0 / (1 + x))
        self.df[gen_ppnames_dr('earnings')] = multiply(self.df[gen_ppnames_dr('purse')], self.df[gen_ppnames_dr('x8_pctofpurse')])

        self.df['x8_ratio_purse_0_claim'] = self.df['pp_purse_0_pp'] / self.df['horse_claiming_price_pp'] #last race purse over current claiming price
        self.df['x8_ratio_earnings_0_claim'] = self.df['pp_earnings_0_pp'] / self.df['horse_claiming_price_pp'] #last race money earned over current claiming price

        # HDW RACE PAR
        self.df['x8runner_HDWPSRRating_norm_par'] = self.df['runner_HDWPSRRating_pp'].fillna(0) / self.df['race_speed_HDW_par_class_level_pp']
        self.df['x8_is_HDWPSRRating_norm_par'] = self.df['x8runner_HDWPSRRating_norm_par'].map(lambda x: int(x > .975))

        # HDW RUNNER
        self.df['x8speed_HDW_2of3_mean'] = self.df.loc[:, ['pp_speed_HDW_0_pp', 'pp_speed_HDW_1_pp', 'pp_speed_HDW_2_pp']].apply(lambda row: mean_best_N_of_K(row, n=2, k=3), axis=1)
        self.df['x8_speed_HDW_best_mean_2of3_lag_1'] = self.df.loc[:, ['pp_speed_HDW_1_pp', 'pp_speed_HDW_2_pp', 'pp_speed_HDW_3_pp']].apply(lambda row:mean_best_N_of_K(row,2,3),axis=1)
        self.df['x8_speed_HDW_best_mean_2of3_lag_2'] = self.df.loc[:, ['pp_speed_HDW_2_pp', 'pp_speed_HDW_3_pp', 'pp_speed_HDW_4_pp']].apply(lambda row:mean_best_N_of_K(row,2,3),axis=1)
        self.df['x8speed_HDW_2of3_rank'] = self.df.groupby('race_id')['x8speed_HDW_2of3_mean'].transform(lambda x: x.rank(ascending=False))
        self.df['x8speed_HDW_2of3_norm_par'] = self.df['x8speed_HDW_2of3_mean'] / self.df.race_speed_HDW_par_class_level_pp

        # DRF RUNNER
        self.df['x8speed_DRF_2of3_mean'] = self.df.loc[:, ['pp_speed_DRF_0_pp', 'pp_speed_DRF_1_pp', 'pp_speed_DRF_2_pp']].apply(lambda row: mean_best_N_of_K(row, n=2, k=3), axis=1)
        self.df['x8speed_DRF_2of3_rank'] = self.df.groupby('race_id')['x8speed_DRF_2of3_mean'].transform(lambda x:x.rank(ascending=False))

        self.df['x8_is_secondtimestarter'] = self.df.runner_horse_lifetime_starts_pp.map(lambda x: int(x == 1))

        self.df['median_speed_HDW'] = self.df[factors_pp_speed_HDW].median(axis=1)
        self.df['median_speed_DRF'] = self.df[factors_pp_speed_DRF].median(axis=1)

        # speed sum
        self.df['x8diffspeed_HDWPSRRating__HDWPar'] = self.df['runner_HDWPSRRating_pp'] - self.df['race_speed_HDW_par_class_level_pp']
        self.df['x8diffspeed_x8speed_HDW_2of3_mean__HDWPar'] = self.df['x8speed_HDW_2of3_mean'] - self.df['race_speed_HDW_par_class_level_pp']
        self.df['x8diffspeed_x8max_speed__HDWPar'] = self.df['x8max_speed_HDW_pp'] - self.df['race_speed_HDW_par_class_level_pp']
        self.df['x8diffspeed_runner_speed_HDWBest_turf_pp__HDWPar'] = self.df['runner_speed_HDWBest_turf_pp'] - self.df['race_speed_HDW_par_class_level_pp']
        self.df['x8diffspeed_runner_runner_speed_HDWBest_distance_pp__HDWPar'] = self.df['runner_speed_HDWBest_distance_pp'] - self.df['race_speed_HDW_par_class_level_pp']
        speed_sum_cols = [c for c in self.df.columns if c.startswith('x8diffspeed')]
        self.df['x8speed_sum_par'] = self.df[speed_sum_cols].applymap(lambda x: int(x > 0)).sum(axis=1) + self.df['runner_morning_line_odds_pp']

        # trading cols
        self.df['x8t001_HDWSpeedRating'] = ((self.df['runner_HDWPSRRating_pp'] - self.df['x8min_speed_HDW_pp']) /
                                            (self.df['x8max_speed_HDW_pp'] - self.df['x8min_speed_HDW_pp']) * 2.0 - 1)

        # # add ppk scores from table
        # df_ppk_score = read_sql('paprika_scores', engine_factors, index_col='runner_id')
        # self.df['ppk_score'] = self.df['runner_id'].map(df_ppk_score['paprika_score'].to_dict())
        # # for now (until we have ppk scores in database)
        # if self.df['ppk_score'].isnull().all():
        #     self.df['ppk_score'] = self.df.groupby('race_id')['runner_id'].transform(lambda x: random.randint(60, 80, size=len(x)))

        # following columns were built for SPPK_WIN_mini:
        # sppk rank
        self.df.sort_values(['race_id', 'runner_program_post_position_pp'], ascending=True, inplace=True)
        self.df['rank_hdw_sppk'] = self.df.groupby('race_id')['runner_HDWPSRRating_pp'].rank(method='first', ascending=False)
        self.df['rank_ml_sppk'] = self.df.groupby('race_id')['prob_morning_line_pp'].rank(method='first', ascending=False)

        # identifier column for races with any null hdw runners
        self.df['isnull_hdw'] = self.df.runner_HDWPSRRating_pp.isnull()
        null_hdw_races = self.df[self.df['isnull_hdw']].race_id.unique()
        self.df['isnull_hdw_any'] = self.df.race_id.map(lambda x: True if x in null_hdw_races else False)
        print('%s races where there is at least 1 null hdw score' % len(null_hdw_races))
        self.df['num_isnull_hdw'] = self.df.groupby('race_id')['isnull_hdw'].transform('sum')
        self.df['isnull_hdw_all'] = self.df['num_isnull_hdw'] == self.df['x8_num_starters_pp']
        print('%s races where all runners have null hdw scores' % self.df[self.df['isnull_hdw_all']].race_id.nunique())
        # some but not all
        self.df['isnull_hdw_some'] = self.df.groupby('race_id').isnull_hdw_all.transform('sum') != self.df.groupby('race_id').isnull_hdw_any.transform('sum')
        print('%s races where some but not all runners have a null hdw score' % self.df[self.df['isnull_hdw_some']].race_id.nunique())
        # par null
        self.df['isnull_race_hdw_par'] = self.df['race_speed_HDW_par_class_level_pp'].isnull()
        print('%s races where are race hdw par is null' % self.df[self.df['isnull_race_hdw_par']].race_id.nunique())

    def compute_probs(self, prob_model='all', uniform=False, t4=[]):
        """
        compute probability columns (or columns will be different after filtering scratches) to dr.dfX
        """

        accepted_models = ['prob_runner_HDWPSRRating',
                           'prob_x8speed_HDW_2of3_mean',
                           'prob_median_speed_HDW',
                           'prob_morning_line']

        if prob_model != 'all':
            assert prob_model in accepted_models
            assert prob_model.split('_')[0] == 'prob'

        t = time()
        if prob_model == 'all':
            print('betsim.wrap.DailyRaces(): computing prob_runner_HDWPSRRating')
            self.dfX = self.dfX.groupby('race_id').apply(lambda dfrace: add_win_probs_from_score(dfrace, 'runner_HDWPSRRating'))
            print('betsim.wrap.DailyRaces(): computing prob_x8speed_HDW_2of3_mean')
            self.dfX = self.dfX.groupby('race_id', as_index=False).apply(lambda dfrace: add_win_probs_from_score(dfrace, 'x8speed_HDW_2of3_mean'))
            print('betsim.wrap.DailyRaces(): computing prob_median_speed_HDW')
            self.dfX = self.dfX.groupby('race_id', as_index=False).apply(lambda dfrace: add_win_probs_from_score(dfrace, 'median_speed_HDW'))

        elif prob_model == 'prob_morning_line':
            print('prob_morning_line is already computed..')

        else:
            attr = prob_model[prob_model.find('_') + 1:]
            self.dfX = self.dfX.groupby('race_id').apply(lambda dfrace: add_win_probs_from_score(dfrace, attr))

        if prob_model in ['all', 'prob_runner_HDWPSRRating'] and uniform:
            uniform_probs = 1 / self.dfX.num_starters
            # TODO only fillna where isnull_hdw_all is True
            self.dfX['prob_runner_HDWPSRRating'] = self.dfX.prob_runner_HDWPSRRating.fillna(uniform_probs)
            #self.dfX.loc[self.dfX.index[self.dfX[self.dfX['isnull_hdw_all']], 'prob_runner_HDWPSRRating']

            # TODO autofill 25% prob_runner_HDWPSRRating if is_first_time and not isnull_hdw_all
            #self.dfX.loc[self.dfX.index[self.dfX['is_first_time'].astype(bool) * ~self.dfX['isnull_hdw_all']], 'prob_runner_HDWPSRRating']
            #self.dfX['prob_runner_HDWPSRRating'] = self.dfX.prob_runner_HDWPSRRating.fillna(uniform_probs)

        s = round(time() - t)
        print('betsim.wrap.DailyRaces(): computed prob model(s) in %s seconds' % s)

        # add t4 probs
        if t4:
            for c in t4:
                self.dfX = add_probsT4(self.dfX, c, v='old')

        # TODO create flag column for probs that look bad i.e. Foreign Horses will have Nan in HDW Scores

        probs = [c for c in self.dfX.columns if c.startswith('prob_') or c.startswith('probT4_')]

        for col in probs:
            # sort by largest to smallest
            self.dfX = self.dfX.groupby('race_id', as_index=False).apply(lambda df: df.sort_values(col, ascending=False)).reset_index(drop=True)
            self.dfX['cumsum_%s' % col] = self.dfX.groupby('race_id', as_index=False)[col].cumsum().reset_index(drop=True)
            self.dfX['rank_%s' % col] = self.dfX.groupby('race_id', as_index=False)[col].rank(ascending=False).reset_index(drop=True)
            # entropy
            self.dfX['entropy_%s' % col] = self.dfX.groupby('race_id', as_index=False)[col].transform(lambda x: entropy(x, base=len(x))).reset_index(drop=True)
            self.dfX['entropy_drop_top_%s' % col] = self.dfX.groupby('race_id', as_index=False)[col].transform(lambda x: entropy(x.drop(x.idxmax()), base=len(x)) if x.notnull().all() else nan).reset_index(drop=True)

        # if prob_model == 'prob_runner_HDWPSRRating':
        #     # hdw t4 idx
        #     self.dfX = add_probsT4(self.dfX, prob_model, v='old')
        # self.dfX = add_probsT4(self.dfX, 'prob_morning_line_post_norm', v='old')

    def _normalize_probs(self):
        """
        re-normalize raw probabilities after filtering scratches
        """

        cols = [c for c in self.dfX.columns if c.startswith('prob_') and c.find('pseudo') < 0]

        for c in cols:
            self.dfX[c] = self.dfX[c] / self.dfX.groupby('race_id')[c].transform(sum)

    def add_factors(self):

        if self.ff.df_train_winners.empty:
            self.ff.load()

        dfs = []
        for race_id, df in self.dfX.set_index(cols_x8race_class).groupby('race_id'):
            df = self.ff.add_x8class_score_race(df, self.ff.grp_train)
            dfs.append(df)
        self.dfX = concat(dfs)
        self.dfX.reset_index(inplace=True)

    def label(self, attr_map=None):
        """
        label dr.df to make dr.dfX
        """

        if attr_map:
            if self.verbose:
                print('betsim.wrap.DailyRaces(): labelling dfX using custom attr_map for state=%s' % self.state)
        else:
            # use default attr_map chosen by state of race if attr_map not given
            attr_map = self.default_attr_map
            if self.verbose:
                print('betsim.wrap.DailyRaces(): labelling dfX using default attr_map for state=%s' % self.state)

        # attribute mapping: attribute/name/label/column
        for new_name, old_name in attr_map.items():
            if new_name == 'program_number' and self.ww.df_entries.empty:
                old_name = 'runner_program_number_pp'
            if old_name in self.df.columns:
                self.dfX[new_name] = self.df.loc[:, old_name]

    def generate_df_bets(self, bet_type, race_id_start='', bet_amount=1.0, filter_scratches=False, strategy_name='x8default'):
        """
        generate all combinations of all possible bets in given day for given bet type
        for race in dfX.groupby(race_id):
            1) generate combinations
            2) wrap with Bets() to make df_bets
        :param race_id_start: starting race_id for bet combinations, if none given, use all races in dr.dfX
        :param bet_type: single leg: 'WN', 'PL', 'SH'
                         multi-runner: 'EX', 'TR', 'SU'
                         multi-race: 'DB', 'P3', 'P4', 'P5', 'P6'
        :param bet_amount: float, bet amount for each combination
        :param filter_scratches: bool, filter scratches or not
        :param strategy_name: str, for Bets(strategy_name=?)
        """

        # time process / bet type classification / # of positions by bet
        t = time()
        single_leg = ['WN', 'PL', 'SH']
        multi_runner = ['EX', 'TR', 'SU']
        multi_race = ['DB', 'P3', 'P4', 'P5', 'P6']
        map_n_leg = {'WN': 1, 'PL': 1, 'SH': 1, 'EX': 2, 'TR': 3, 'SU': 4, 'DB': 2, 'P3': 3, 'P4': 4, 'P5': 5, 'P6': 6}

        # create race_list
        if bet_type in single_leg + multi_runner:
            if not race_id_start:
                race_list = self.dfX['race_id'].unique()
            else:
                race_list = [race_id_start]

        elif bet_type in multi_race:
            if not race_id_start:
                raise Exception('If generating multi-race bets, user must pass race_id_start param')
            else:
                track = race_id_start.split('_')[0]
                date = to_datetime(race_id_start.split('_')[1])
                race_num = int(race_id_start.split('_')[2])
                race_card = self.dfX[(self.dfX['x8_track_sym'] == track) * (self.dfX['date']==date)]
                race_order = race_card.groupby('race_id')['race_num'].max().sort_values()
                race_list_series = race_order[(race_order >= race_num) & (race_order < race_num + map_n_leg[bet_type])]
                race_list = list(race_list_series.index)
        else:
            raise ValueError('bet type not supported..')

        # create copy of self.dfX, groupby object for race_id's and df_bets
        dfX = self.dfX.copy()

        # filter scratches
        if filter_scratches:

            # filter_scratches param exceptions
            if self.state == 'pre':
                raise Exception('If filter_scratches is True, dr.state cannot be pre.')

            len_runners_before = dfX.index.nunique()
            dfX = dfX[dfX['scratch'].isnull()]
            len_runners_after = dfX.index.nunique()
            print('dr.generate_df_bets(): filtered scratches from %s runners to %s runners.' % (len_runners_before,
                                                                                                len_runners_after))

        dfX = dfX.set_index('runner_id')
        df_races = dfX[dfX['race_id'].isin(race_list)].groupby('race_id')
        df_bets = DataFrame()

        # single leg
        if bet_type in single_leg:
            df_bets = dfX[dfX['race_id'].isin(race_list)].add_suffix('_0')
            df_bets.index.name = 'pos_0'

        # multi-runner
        elif bet_type in multi_runner:
            for race_id, df in df_races:
                idx = MultiIndex.from_product(iterables=[df.index for i in range(map_n_leg[bet_type])],
                                              names=['pos_%s' % i for i in range(map_n_leg[bet_type])])

                if bet_type == 'EX':
                    l0 = idx.get_level_values(0)
                    l1 = idx.get_level_values(1)

                    idx = idx[l0 != l1]

                elif bet_type == 'TR':
                    l0 = idx.get_level_values(0)
                    l1 = idx.get_level_values(1)
                    l2 = idx.get_level_values(2)

                    idx = idx[(l0 != l1) * (l1 != l2) * (l0 != l2)]

                elif bet_type == 'SU':
                    l0 = idx.get_level_values(0)
                    l1 = idx.get_level_values(1)
                    l2 = idx.get_level_values(2)
                    l3 = idx.get_level_values(3)

                    idx = idx[(l0 != l1) * (l0 != l2) * (l0 != l3) * (l1 != l2) * (l1 != l3) * (l2 != l3)]

                pos_vals = [df.add_suffix('_%s' % i)
                              .reindex(idx, level='pos_%s' % i)
                            for i in range(map_n_leg[bet_type])]

                merge = concat(pos_vals, axis=1)

                # concatenate
                df_bets = concat([df_bets, merge])

        elif bet_type in multi_race:
            n_bets_day = {'DB': 1, 'P3': 2, 'P4': 3, 'P5': 4, 'P6': 5}

            # validating that given race_list are consecutively occurring races in the same race card
            a = Series(race_list).map(lambda x: x.split('_')[0]).nunique() != 1
            b = Series(race_list).map(lambda x: x.split('_')[1]).nunique() != 1
            c = Series(race_list).map(lambda x: x.split('_')[2]).nunique() != len(race_list)
            if a or b or c:
                raise ValueError('race_list must be list of consecutive races for multi-race bet types..')
            # validating that enough races were given for given bet_type i.e. at least 3 races for a Pick 3 bet
            elif n_bets_day[bet_type] >= len(race_list):
                raise ValueError('Cannot generate %s bets for %s races.' % (bet_type, len(race_list)))

            # sort race_list in order of race occurrence
            df_order = DataFrame({'race_id': race_list})
            df_order['race_num'] = df_order['race_id'].map(lambda x: int(x.split('_')[2]))
            race_list = df_order.sort_values('race_num', ascending=True)['race_id']

            # total number of multi-race bets for given races. i.e. n=2 where bet_type=DB and len(race_list)=3
            n = len(race_list) - n_bets_day[bet_type]

            # list of lists of race_id's
            # i.e. if bet_type = DB
            #         race_groups = [['WOX_20181014_1', 'WOX_20181014_2'], ['WOX_20181014_2', 'WOX_20181014_3'], etc. ]
            race_groups = [[race_list[y] for y in range(i, n_bets_day[bet_type] + 1 + i)] for i in range(n)]

            for races in race_groups:
                idx = MultiIndex.from_product([df_races.get_group(race_id).index for race_id in races],
                                              names=['pos_%s' % i for i in range(n_bets_day[bet_type] + 1)])

                df_merge = concat([df_races.get_group(race_id)
                                           .add_suffix('_%s' % i)
                                           .reindex(idx, level='pos_%s' % i)
                                   for i, race_id in enumerate(races)],
                                  axis=1)

                # concatenate
                df_bets = concat([df_bets, df_merge])

        # assign runners column
        if bet_type in ['WN', 'PL', 'SH']:
            df_bets['runners'] = df_bets['program_number_0']
        elif bet_type in ['EX', 'DB']:
            df_bets['runners'] = df_bets['program_number_0'].str.cat(df_bets['program_number_1'], sep=',')
        elif bet_type in ['TR', 'P3']:
            df_bets['runners'] = df_bets['program_number_0'].str.cat([df_bets['program_number_1'],
                                                                      df_bets['program_number_2']], sep=',')
        elif bet_type in ['SU', 'P4']:
            df_bets['runners'] = df_bets['program_number_0'].str.cat([df_bets['program_number_1'],
                                                                      df_bets['program_number_2'],
                                                                      df_bets['program_number_3']], sep=',')
        elif bet_type == 'P5':
            df_bets['runners'] = df_bets['program_number_0'].str.cat([df_bets['program_number_1'],
                                                                      df_bets['program_number_2'],
                                                                      df_bets['program_number_3'],
                                                                      df_bets['program_number_4']], sep=',')

        elif bet_type == 'P6':
            df_bets['runners'] = df_bets['program_number_0'].str.cat([df_bets['program_number_1'],
                                                                      df_bets['program_number_2'],
                                                                      df_bets['program_number_3'],
                                                                      df_bets['program_number_4'],
                                                                      df_bets['program_number_5']], sep=',')

        # assign other df_bets columns
        df_bets['bet_type'] = bet_type
        df_bets['bet_amount'] = bet_amount
        df_bets['race_id'] = df_bets['race_id_0'].values
        df_bets['itsp_track_sym'] = df_bets['itsp_track_sym_0'].values
        df_bets['x8_track_sym'] = df_bets['x8_track_sym_0'].values
        df_bets['date'] = df_bets['date_0'].dt.date
        df_bets['race_num'] = df_bets['race_num_0'].values

        # x8 race class cols
        df_bets['race_race_type'] = df_bets['race_race_type_0'].values
        df_bets['race_distance'] = df_bets['race_distance_0'].values
        df_bets['race_surface'] = df_bets['race_surface_0'].values
        df_bets['race_age_sex_restriction'] = df_bets['race_age_sex_restriction_0'].values

        # Bets()
        self.bets.load(df_bets, strategy_name=strategy_name)

        elapsed = time() - t
        print('generated %s bets in %s seconds' % (bet_type, elapsed))

    def load_df_bets_all(self, multirace=False):
        """
        generate every possible permutation for every bet type
        :return: df_bets_all
        """

        if self.state == 'pre':
            filter_scratches = False
        else:
            filter_scratches = True

        # TODO unit test

        # single and multi-runner
        df_bets_all = DataFrame()
        for bet_type in ['WN', 'PL', 'SH', 'EX', 'TR', 'SU']:
            self.generate_df_bets(bet_type=bet_type, strategy_name='df_bets_all', filter_scratches=filter_scratches)
            df_bets_all = concat([df_bets_all, self.bets.df_bets], sort=True)

        # multi-race
        num_races = {'DB': 2, 'P3': 3, 'P4': 4, 'P5': 5, 'P6': 6}
        for track, df_card in self.dfX.groupby('x8_track_sym'):
            races = df_card['race_id'].unique()
            for bet_type in ['DB', 'P3', 'P4', 'P5', 'P6']:
                if len(races) >= num_races[bet_type]:
                    for race_id in races:
                        try:
                            self.generate_df_bets(race_id_start=race_id, bet_type=bet_type, strategy_name='df_bets_all', filter_scratches=filter_scratches)
                            df_bets_all = concat([df_bets_all, self.bets.df_bets], sort=True)
                        except ValueError:
                            continue

        # Bets()
        self.bets.load(df_bets_all)

    def make_df_calibrate(self):

        if self.state not in ['post', 'historical']:
            raise Exception('must be in state=post to create df_calibrate')

        # this makes df with 1 race per row and each bet type for the columns with the values as the pool total
        df_pivot_pool = pivot_table(self.jcp.df_payout, index='race_id', columns='bet_type', values='pool_total')
        # this makes df with 1 race per row and each bet type for the columns with the values as the normalized payout
        df_pivot_payout = pivot_table(self.jcp.df_payout, index='race_id', columns='bet_type', values='payout_norm')

        self.df_calibrate = merge(self.dfX, df_pivot_pool, how='left', on='race_id')
        self.df_calibrate = merge(self.df_calibrate, df_pivot_payout, how='left', on='race_id', suffixes=['_pool', '_payout'])

        self.df_calibrate['first_place'] = self.df_calibrate['official_finish_position'] == 1.0
        self.df_calibrate['second_place'] = self.df_calibrate['official_finish_position'] == 2.0

        print('generated dr.df_calibrate..')

    def get_odds_profile(self):
        """
        TODO pivot table all odds columns by runner
        :return:
        """
        pass

    def add_runner_factors_from_dailyraces(self, df_inputs_runner, cols_factor):
        """
        Convenience function for merging input df and selected cols from master df
        :param df_inputs_runner: arbitrary df with index='runer_id'
        :param cols_factor: list of columns
        :return: df
        """

        if df_inputs_runner.index.name != 'runner_id':
            raise ValueError("df_inputs_runner must have index='runner_id'")

        df_factors = self.df.set_index('runner_id')[cols_factor].copy()
        df_factors = merge(df_inputs_runner, df_factors, left_index=True, right_index=True, how='left').drop_duplicates().fillna(0.0)

        return df_factors

    def add_runner_result_labels(self, df_inputs_runner, cols_result=['official_finish_position_jcp', 'payout_win_jcp', 'payout_place_jcp', 'payout_show_jcp']):
        """
        Adds runner_result labels to a dataframe of runner inputs
        runner_result_labels are things like:
        -official_finish_position_jcp

        :param df_inputs_runner:
        :param cols_result:
        :return:
        """
        if self.state != 'post':
            raise Exception("DailyRaces.state must equal 'post'")
        if self.jcp.df.empty:
            raise Exception("No charts files yet.. (jcp)")

        df_results = self.df.set_index('runner_id')[cols_result].copy()
        df_labeled = merge(df_inputs_runner, df_results, left_index=True, right_index=True, how='left').drop_duplicates().fillna(0.0)
        return df_labeled

    def _validate(self):
        """
        df is as expected
        """

        # ensure dailyraces outputs uniform column names regardless of state being handled
        if self.pp.df.empty:
            assert 'x8_num_starters_pp' in self.df.columns
            assert 'x8_num_starters' in self.pp.df.columns
        if not self.ww.df_win_odds.empty:
            assert 'market_odds_win' in self.df.columns
            assert 'market_odds' in self.ww.df_win_odds.columns
        if not self.jcp.df.empty:
            assert 'race_id' in self.jcp.df.columns

        if not self.ww.df_scratches.empty and not self.ww.df_win_odds.empty:
            # late (< 5 minute) scratches - scratched runners that we have odds for
            # where runner is scratched and has odds
            late_scratches = self.df[(~self.df['ww_scratched'].isnull()) & (~self.df['market_odds_win'].isnull())]

        if not self.ww.df_entries.empty: # play
            # tracks that are in entries that are not in pp
            set_bad = set(self.ww.df_entries['runner_id']) - set(self.pp.df['runner_id'])

        if not self.ww.df_final_win_odds.empty:
            # assert every final odds row was mapped correctly onto master df
            set_final_odds_not_used = set(self.df[~self.df['market_odds_win_final'].isnull()]['runner_id']) - set(self.ww.df_final_win_odds)
            # set_final_odds_not_used should be the collection of tracks/runners that are dropped with doing 'inner' merge with pp and entries

        if len(self.df) != self.df['runner_id'].nunique():
            raise Exception('dr.df does not have 1 runner per row')

    def dump_pickle(self, path):
        # TODO: dates for output
        fp = open(os.path.join(path,'dr.pickle'))
        pickle.dump(self, (open(fp,'wb')))#.format(start_date.strftime(format_date), end_date.strftime(format_date))))
        fp.close()

    def dump_csv(self, path=data):
        self.df.to_csv(os.path.join(path, 'dfdr.csv'))#.format(start_date.strftime(format_date),
                                                                     #   end_date.strftime(format_date))))

    def get_daily_profile_pp(self):
        df_daily_profile = pivot_table(self.df, index=['x8_track_sym_pp',
                                                       'race_time_pp',
                                                       'x8_num_starters_pp',
                                                       'race_race_conditions_pp',
                                                       'race_age_sex_restriction_pp',
                                                       'race_race_classification_pp',
                                                       'race_distance_pp',
                                                       'race_surface_pp',
                                                       'race_id'],
                                                columns=['race_race_num_pp'],
                                                values=['x8runner_HDWPSRRating_norm_par',
                                                        'x8_is_HDWPSRRating_norm_par'],
                                                aggfunc={'x8runner_HDWPSRRating_norm_par': lambda x: x.describe()['50%'],
                                                         'x8_is_HDWPSRRating_norm_par': lambda x: x.sum()}).fillna(0)
        return df_daily_profile

    def generate_paprika_runner(self):
        """ map to """

