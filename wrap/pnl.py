
from os.path import join
from pandas import DataFrame, Series, to_datetime, DatetimeIndex, merge, read_csv, concat
from horse.betsim.wrap.jcapper import JCapper
from horse.betsim.wrap.watchandwager import WatchandWager
from horse.betsim.wrap.bets import all_required_cols
from horse.betsim import data
from datetime import datetime, date

# for merging with rebates in track_detail
rebate_cols = ['x8_track_sym', 'WN', 'PL', 'SH', 'EX', 'QU', 'TR', 'SU', 'SH5', 'DB', 'P3', 'P4', 'P5', 'P6']

# single leg, multi-runner and multi-race pool symbols
pool_single = ['WN', 'PL', 'SH']
pool_multi_runner = ['EX', 'QU', 'TR', 'SU']
pool_multi_race = ['DB', 'P3', 'P4', 'P5', 'P6']


class PNL:
    """
    simulator code for calculating theoretical pnl from an arbitrary df_bets
    input: df_bets (betfile)
    output: pnl (df_bets merged with payouts)
    """
    def __init__(self, jcp=None, market_impact=True):
        # df_bets
        self.df_bets = DataFrame()
        self.df_error = DataFrame()

        # flags
        self.market_impact = market_impact

        # master df with bets and payouts merged intelligently and calculated columns
        self.performance = Series()

        # track detail
        track_detail = join(data.__path__._path[0], 'track_detail.csv')
        self.dftrack = read_csv(track_detail)

        # datelist & tracklist
        self.datelist = []
        self.tracklist = []

        # data wrappers
        if not jcp:
            self.jcp = JCapper()
        else:
            self.jcp = jcp
            assert not self.jcp.df.empty

        self.ww = WatchandWager()

        # data place holders
        self.df_payout = DataFrame()
        self.scratches = set([])
        self.not_cancelled = set([])
        self.run_races = []
        self.dict_finish = {}

    def load(self, df_bets):
        """
        takes bet file (df_bets), extracts dates and loads necessary data to calculate theoretical pnl
        :param df_bets: df_bets is defined as a DataFrame that includes at least columns necessary for amtote.submit_bets()
        :return: master df with bets and payouts merged with calculated pnl columns assigned
        """

        if df_bets.empty:
            raise ValueError('df_bets is empty')

        df_bets_cols = set(df_bets.columns)
        expected = set(all_required_cols)
        missing_cols = expected - df_bets_cols
        if expected - df_bets_cols:
            raise ValueError('Input df_bets must be betsim.wrap.Bets.df_bets object. missing: %s' % missing_cols)

        # format dtypes in case loaded as csv, instantiate self.df_bets and set index to a bet
        df_bets['runners'] = df_bets['runners'].astype(str)
        df_bets['race_num'] = df_bets['race_num'].astype(str)
        self.df_bets = df_bets.reset_index()
        self.df_bets = self.df_bets.set_index(['race_id', 'bet_type', 'runners'], drop=False)

        # if simulating bet_result, use error_code to cheat returns
        if 'error_code' in self.df_bets.columns:
            self.df_error = self.df_bets[self.df_bets['error_code'] != 0]

        # get datelist and tracklist from df_bets
        self.datelist = self._get_datelist(self.df_bets)
        self.tracklist = self._get_tracklist(self.df_bets)

        # load data
        try:
            self._get_results()
        except FileNotFoundError:
            print('No Results Files from JCapper or WaW for target date. Breaking early.')
            self.df = self.df_bets.copy()
            return 0

        # merge df_bets with payouts
        if 'error_code' in self.df_bets.columns:
            self.df = (merge(self.df_bets[self.df_bets['error_code']==0], self.df_payout, left_index=True, right_index=True, how='left')
                       .fillna({'payout_norm': 0}))
        else:
            self.df = (merge(self.df_bets, self.df_payout, left_index=True, right_index=True, how='left')
                       .fillna({'payout_norm': 0}))

        # signal cols used for making runner_id_list
        self.df['is_single_leg'] = self.df['bet_type'].isin(pool_single)
        self.df['is_multi_runner'] = self.df['bet_type'].isin(pool_multi_runner)
        self.df['is_multi_race'] = self.df['bet_type'].isin(pool_multi_race)

        # assign runner_id_list for every bet type
        self.df['runner_id_list'] = self.df.apply(self._get_runner_id_list, axis=1, result_type='reduce')

        # official finish position - only works for single runner bets
        self.df['official_finish_position_0'] = self.df['runner_id_list'].map(lambda x: self.dict_finish.get(x[0]))

        # scratches for refund column
        self.df['is_scratched'] = self.df['runner_id_list'].map(lambda x: int(bool([r_id for r_id in x if r_id in self.scratches])))

        # cancelled races for refund column
        self.df['is_cancelled'] = self.df['race_id'].map(lambda x: int(x not in self.not_cancelled))

        # refunds
        self.df['is_refund'] = ((self.df['is_scratched'].values + self.df['is_cancelled'].values) > 0).astype(int)
        self.df['refunds'] = self.df['bet_amount'].values * self.df['is_refund'].values

        # intra-day simulation - which races have not been run
        self.df['is_post'] = self.df['race_id'].isin(self.run_races)

        # TODO EDGE CASES 1) coupled entries 2) scratches with coupled entries

        self.df['bet_amount_accepted'] = self.df['bet_amount'] - self.df['refunds']

        # re-combine df_bets with error
        if not self.df_error.empty:
            self.df_error['bet_amount_accepted'] = 0
            self.df = concat([self.df, self.df_error])

        # calculate rebates
        self._calc_rebates()

        # pnl calcs
        self._calcs()

        # identifiers and analytics
        self._add_computed_cols()

        # performance
        self._performance()

        # set index to runners
        pos_cols = [c for c in self.df.columns if c.startswith('pos_')]
        if len(pos_cols) > 0:
            self.df.set_index(pos_cols, inplace=True)

        # validate df
        self._validate()

    def _get_results(self):
        """
        1. Loads payouts and results data from multiple data sources
           - JCapper Results
           - WaW Results (intra-day simulating)
           - WaW Scratches
        2. Chooses which data source to use based on the availability of each source
        """

        # if today date is in datelist, use WaW Results for intra-day simulation, otherwise use JCapper Results
        if (self.datelist.date == datetime.today().date()).any():
            if self.ww.df_results.empty:
                self.ww.load_results(self.datelist, self.tracklist)
                self.ww.load_scratches(self.datelist)
            self.df_payout = self.ww.df_results.copy()
            self.scratches = set(self.ww.df_scratches['runner_id'])
            self.not_cancelled = set(self.ww.df_results['race_id'])

            print('pnl.load() simulating intra-day pnl using ww (not calculating market impact)')
            self.df_payout = self.df_payout[['race_id', 'bet_type', 'winning_pgm', 'payout_norm']]

            # find which races have not been run yet
            self.run_races = self.df_payout['race_id'].unique()

        else:
            try:
                if self.jcp.df.empty:
                    self.jcp.load(self.datelist, self.tracklist)
            except FileNotFoundError:
                try:
                    self.ww.load_results(self.datelist, self.tracklist)
                except FileNotFoundError:
                    raise FileNotFoundError('we have no jcapper or waw results data for target date: %s, skipping.' % self.datelist[0])
            self.df_payout = self.jcp.df_payout.copy()
            self.not_cancelled = set(self.jcp.df_payout['race_id'])
            self.dict_finish = self.jcp.df.set_index('runner_id')['official_finish_position'].to_dict()

            print('pnl.load() simulating historical pnl using JCapper (calculating market impact if ON)')
            self.df_payout = self.df_payout[['race_id', 'bet_type', 'winning_pgm', 'payout_norm', 'correct_money', 'pool_total']]

            # if all dates are before 2018/04/05 use JCapper for scratches
            if (self.datelist.date < date(2018, 4, 5)).all():
                self.scratches = set(self.jcp.df_scratch['runner_id'])

            # otherwise use WaW for scratches
            else:
                self.ww.load_scratches(self.datelist)
                self.scratches = set(self.ww.df_scratches['runner_id'])

        # df_payout normalizations for merge
        self.df_payout.rename(columns={'winning_pgm': 'runners'}, inplace=True)
        self.df_payout = self.df_payout[self.df_payout['bet_type'].notnull()]
        self.df_payout.set_index(['race_id', 'bet_type', 'runners'], inplace=True)

    def _get_datelist(self, df_bets):
        """
        infer datelist from df_bets
        :return: list, list of datetime objects
        """

        # if you are loading betfile from csv (debug script) the date column will be string instead of datetime
        dates = df_bets['date'].map(to_datetime)

        # get datetimes from df_bets
        datelist = DatetimeIndex(dates.dt.date.unique())

        return datelist

    def _get_tracklist(self, df_bets):
        """
        infer tracklist from df_bets
        :return: list, list of x8_track_sym's
        """

        # get tracks from df_bets
        race_id_series = Series(df_bets['race_id'].unique())
        if race_id_series.isnull().any():
            print('null tracks - dropping and continuing' % race_id_series.isnull().sum())
            race_id_series = race_id_series[race_id_series.notnull()]
        tracklist = race_id_series.map(lambda x: x.split('_')[0])

        return tracklist

    def _get_runner_id_list(self, row):
        # TODO This is very slow
        # return tuple of runner_ids involved in bet
        # return tuple of 1 runner_id for single_leg bet
        # return tuple of n runner_ids for n sized multi_runner bet
        # return tuple of n runner_ids for n sized multi_race bet
        if row['is_single_leg']:
            runner_id_list = [row['race_id'] + '_' + row['runners']]
        elif row['is_multi_runner']:
            runner_id_list = [row['race_id'] + '_' + program_number for program_number in row['runners'].split(',')]
        elif row['is_multi_race']:
            runner_id_list = [row['race_id'][:row['race_id'].rfind('_')] + '_' + str(int(row['race_num'])+i) + '_' + program_number for i, program_number in enumerate(row['runners'].split(','))]
        else:
            runner_id_list = []
            print('pnl._get_runner_id_list() not working as expected..')

        return tuple(runner_id_list)

    def _calc_rebates(self):
        """
        calculate rebates
        """

        # rebates calcs
        len_df = len(self.df)
        self.df = self.df.set_index(['x8_track_sym', 'bet_type'])
        df_rebates = (self.dftrack[self.dftrack['x8_track_sym'].notnull()][rebate_cols]
                          .melt(id_vars='x8_track_sym', var_name='bet_type', value_name='rebate_pct')
                          .set_index(['x8_track_sym', 'bet_type']))
        df_rebates = df_rebates.fillna(0)
        self.df = merge(self.df, df_rebates, left_index=True, right_index=True, how='left').fillna({'rebate_pct': 0})
        assert len_df == len(self.df), 'duplicate tracks in track detail i.e. LDM'
        self.df['rebate'] = self.df['bet_amount_accepted'].values * self.df['rebate_pct'].values
        self.df = self.df.reset_index()

    def _calcs(self):
        """
        pnl calculations
        """

        # pnl calcs
        self.df['gross_payout'] = self.df['bet_amount_accepted'].values * self.df['payout_norm'].values
        self.df['tradingprofitloss'] = self.df['gross_payout'].values - self.df['bet_amount_accepted'].values
        self.df['net_return'] = self.df['tradingprofitloss'].values + self.df['rebate'].values

        # do not calculate market impact returns if set to False
        if not self.market_impact:
            return 0

        # market impact calcs
        # we only care about our impact on pools we won because if we lost than it doesn't matter how much we bet,
        # we just lose all the money we bet
        df_winning_bets = self.df[self.df['gross_payout'] > 0]
        if not df_winning_bets.empty and not self.jcp.df.empty:
            # pool totals for each pool bet in
            pool_total = df_winning_bets.groupby(['race_id', 'bet_type'])['pool_total'].max()
            # normalized payouts for each pool bet in
            payout_norm = df_winning_bets.groupby(['race_id', 'bet_type'])['payout_norm'].max()
            # total amount of winning $1 bets on each pool i.e. normalized correct money
            correct_money = df_winning_bets.groupby(['race_id', 'bet_type'])['correct_money'].max()

            # total amount executed in each pool
            total_executed = self.df.groupby(['race_id', 'bet_type'])['bet_amount_accepted'].sum()
            # total amount executed in each pool with winning bets
            total_executed_correct = df_winning_bets.groupby(['race_id', 'bet_type'])['bet_amount_accepted'].sum()

            # pool total plus our money
            new_pool_total = concat([pool_total, total_executed], axis=1).dropna().sum(axis=1)
            # correct money plus our correct money
            new_correct_money = concat([correct_money, total_executed_correct], axis=1).dropna().sum(axis=1)
            # payouts with impact (after accounting for our money)
            new_payout_norm = new_pool_total.divide(new_correct_money)

            index_finder = df_winning_bets.reset_index().groupby(['race_id', 'bet_type'])['index'].max()
            map_new_payouts = concat([index_finder, new_payout_norm], axis=1).set_index('index')[0].to_dict()
        else:
            map_new_payouts = {}

        self.df['payout_norm_impact'] = self.df.index.map(lambda x: map_new_payouts.get(x)).fillna(0)

        # pnl calcs w/ impact
        self.df['gross_payout_impact'] = self.df['bet_amount_accepted'].values * self.df['payout_norm_impact'].values
        self.df['tradingprofitloss_impact'] = self.df['gross_payout_impact'].values - self.df['bet_amount_accepted'].values
        self.df['net_return_impact'] = self.df['tradingprofitloss_impact'].values + self.df['rebate'].values

    def _add_computed_cols(self):
        """
        additional columns
        """
        self.df['is_bet'] = (self.df['bet_amount_accepted'] > 0).astype(int)
        self.df['is_hit'] = (self.df['tradingprofitloss'] > 0).astype(int)

        # only assign if using jcp
        if not self.jcp.df.empty:
            map_num_starters_pre = self.jcp.df.groupby('race_id')['num_starters_pre'].first().to_dict()
            map_num_starters_post = self.jcp.df.groupby('race_id')['num_starters_post'].first().to_dict()
            self.df['num_starters_pre'] = self.df['race_id'].map(map_num_starters_pre)
            self.df['num_starters_post'] = self.df['race_id'].map(map_num_starters_post)
            self.df['num_exacta_pre'] = self.df['num_starters_pre'].map(lambda x: x*(x-1))
            self.df['num_exacta_post'] = self.df['num_starters_post'].map(lambda x: x*(x-1))
            self.df['num_trifecta_pre'] = self.df['num_starters_pre'].map(lambda x: x*(x-1)*(x-2))
            self.df['num_trifecta_post'] = self.df['num_starters_post'].map(lambda x: x*(x-1)*(x-2))

    def _bet_modifiers(self, df_bets):
        """
        break up bets using bet modifiers so that each bet is 1 row so we can do pnl calcs
        :param df_bets: self.df_bets
        """

        # if input df_bets source is reports than bets are like this: '5+9+7' for win bet, '5+6/6+7' for exacta etc.

        pass

    def _performance(self):
        """
        get performance for df_bets
        """
        performance = dict()
        performance['num_races_bet'] = self.df.race_id.nunique()
        performance['num_races_executed'] = self.df[self.df.refunds == 0].race_id.nunique()
        performance['num_bets'] = len(self.df)
        performance['num_bets_executed'] = len(self.df[self.df.refunds == 0])
        performance['total_amount_sent'] = self.df.bet_amount.sum()
        performance['total_executed'] = self.df.bet_amount_accepted.sum()
        performance['gross_payout'] = self.df.gross_payout.sum()
        performance['gross_return'] = self.df.tradingprofitloss.sum()
        performance['net_return'] = self.df.net_return.sum()
        performance['total_rebates'] = self.df.rebate.sum()
        performance['gross_ROI'] = self.df.tradingprofitloss.sum() / self.df.bet_amount_accepted.sum()
        performance['net_ROI'] = self.df.net_return.sum() / self.df.bet_amount_accepted.sum()

        self.performance = Series(performance)

    def _validate(self):
        # make sure data is as expected

        # for making sure df_bets doesn't multiply because it should always be 1 row per bet
        if len(self.df) != len(self.df_bets):
            raise Exception('Number of bets/rows in master df should be equal to number of bets/rows in input df_bets')

        # DEPRECATED - if entire race card was cancelled then this validation is wrong
        # make sure that payout data includes all the tracks being simulated
        #if len(set(self.tracklist) - set(self.jcp.df['x8_track_sym'])) > 0:
        #    raise Exception('Not all races in self.df_bets or tracklist (if it was passed) are included in self.df_payout')

        # check if output sources are missing data against eachother
        # missing = self.df['payout_norm_ww'].isnull().astype(int) + self.df['payout_norm_jcp'].isnull().astype(int)
        # if (missing == 1).any():
        #     raise Exception('missing results data, look into this: %s' % self.df[missing == 1]['race_id'])
