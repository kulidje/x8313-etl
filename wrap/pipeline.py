# TODO Automatically check for scratches - dr.update() - before sending bets
# TODO real time Expected Value
# TODO target set of races instead of 1 race

from pandas import DataFrame, concat, date_range
from numpy import nan
from horse.betsim.math import get_freq_wom
from horse.betsim.wrap.bets import Bets
from datetime import datetime
from horse.betsim.models.probability import HarvilleTri as harville
from math import ceil


class Strategy(object):
    """
    named tuple
    """
    def __init__(self, name, criteria, prob_model, betFunction):
        self.name = name  # name for labelling
        self.criteria = criteria  # criteria is used to score horses using historical training data
        self.probModel = prob_model  # probability model (attribute name starting w/ 'prob_')
        self.betFunction = betFunction  #


class Factor:
    """
    Parent Class

    1) name (for labelling)
    2) top n () i.e. 1 / 2
    3) horse attribute i.e. 'jockey' / 'runner_HDWPSRRating'
    4) operator i.e. 'median' / 'in'
    5) score i.e. 1 / 2 / 3
    """
    # condition_wom = self.df_train['date'].isin(self.wom_dates)
    # condition_month = self.df_train['month'] == self.target_date.month
    # condition_track = self.df_train['x8_track_sym'] == self.track

    # narrow historical data for selection
    # if sample == 'track':
    #     df_sample = self.df_train[condition_track]
    # elif sample == 'month':
    #     df_sample = self.df_train[condition_month]
    # elif sample == 'wom':
    #     df_sample = self.df_train[condition_wom]
    # else:
    #     raise ValueError('param sample must be in [track, wom, month] - add slices when needed')

    # narrow further if possible

    # 1) train factor using historical or not
    # 2) race/runner/track


class Pipeline:
    """
    The Pipeline Class holds all the data required to execute any model

    1 Pipeline object is used for a target date.
    """
    def __init__(self, dr, hist):

        if dr.dfX.date.nunique() == 1:
            self.dr = dr
        else:
            print('Pipeline() using more than one date..')
            self.dr = dr

        if not hist.df_train.empty:
            # note: not copying the df, using same object in memory (because of computing time)
            self.df_train = hist.df_train
        else:
            raise Exception('History object must be fully loaded before passing to Pipeline.')

        # chosen bet type
        self.race_id = ''
        self.bet_type = ''

        # intermediate internal collection of df_bets (while strategy is being processed)
        self.internal_collection = []

        # output
        self.df_bets = DataFrame()
        self.df_bets_agg = DataFrame()
        self.bets = Bets()

    def format_bets(self, strategy_name):
        """
        call this when every targeted race for single strategy is complete, at the end of strategy function
        """

        # add df_bets from last iteration of pipe.run() sequence to internal collection so it is not lost in shuffle
        self.internal_collection.append(self.df_bets)

        # concatenate full df_bets with constituent race df_bets
        self.df_bets_agg = concat(self.internal_collection)

        # if there are no bets in agg return empty df instead of correct cols instead of empty df of no cols
        if self.df_bets_agg.empty:
            self.run(self.dr.dfX.race_id.iloc[0], 'WN')
            self.df_bets_agg = self.df_bets[self.df_bets['race_id'].isnull()]

        # round up
        self.df_bets_agg['bet_amount'] = self.df_bets_agg['bet_amount'].map(ceil)

        # load into Bets()
        self.bets.load(self.df_bets_agg, strategy_name)

    def run(self, race_id, bet_type):
        """
        run process on 1 race - assign score factors to df_bets
        :param race_id: same same
        :param bet_type: WN/PL/EX .. (no MultiRace yet)
        """

        self.race_id = race_id
        self.bet_type = bet_type

        if not self.df_bets.empty:
            self.internal_collection.append(self.df_bets)

        # make all bet combinations for pool id
        self.dr.generate_df_bets(bet_type, race_id)
        self.df_bets = self.dr.bets.df_bets.copy()

        # assign factors (by pos) to df_bets
        self.df_bets = self._factors(self.df_bets)

        self.validate()

    def _factors(self, df_bets):

        x8race_class = df_bets.x8race_class_0.iloc[0]

        try:
            df_sample = self.df_train.groupby('x8race_class').get_group(x8race_class)
            # Important: un-select historical data that is equal to or greater than target race date
            df_sample = df_sample[df_sample['date'] < df_bets.date.min()]
        except KeyError:
            # TODO generalize this
            # for now every time we add a factor we also have to had the column name to this block to ensure that when
            # there is not df_sample for given x8race_class, the pipeline wont break when pipe[factor] is called in
            # strategy function

            jockey_cols = ['iv_score_' + c for c in df_bets.columns if c.startswith('jockey_')]
            trainer_cols = ['iv_score_' + c for c in df_bets.columns if c.startswith('trainer_')]
            hdwpsr_cols = ['iv_score_median_' + c for c in df_bets.columns if c.startswith('runner_HDWPSRRating_')]
            n_levels = df_bets.index.nlevels
            total_cols = ['iv_score_total_' + str(pos) for pos in range(n_levels)]

            cols = jockey_cols + trainer_cols + hdwpsr_cols + total_cols
            for c in cols:
                df_bets[c] = nan

            df_bets['num_iv_score_total_0'] = nan
            df_bets['num_iv_score_total_1'] = nan
            df_bets['num_iv_score_total_2'] = nan
            df_bets['num_iv_score_total_3'] = nan

            return df_bets

        # winning horses
        df_sample = df_sample[df_sample['official_finish_position'] == 1]

        # factor: iv_score_jockey
        jockey = 'jockey'
        winning_jockey = df_sample[jockey].values  # winning attributes
        cols = [c for c in df_bets.columns if c.startswith(jockey + '_')]
        attach_factor = df_bets[cols].isin(winning_jockey).astype(int)
        attach_factor = attach_factor.rename(lambda x: 'iv_score_' + x, axis=1)
        df_bets = concat([df_bets, attach_factor], axis=1)

        # factor: iv_score_trainer
        trainer = 'trainer'
        winning_trainer = df_sample[trainer].values
        cols = [c for c in df_bets.columns if c.startswith(trainer + '_')]
        attach_factor = df_bets[cols].isin(winning_trainer).astype(int)
        attach_factor = attach_factor.rename(lambda x: 'iv_score_' + x, axis=1)
        df_bets = concat([df_bets, attach_factor], axis=1)

        # factor: iv_score_median_runner_HDWPSRRating
        runner_HDWPSRRating = 'runner_HDWPSRRating'
        winning_runner_HDWPSRRating = df_sample[runner_HDWPSRRating].median()
        cols = [c for c in df_bets.columns if c.startswith(runner_HDWPSRRating + '_')]
        attach_factor = (df_bets[cols] > winning_runner_HDWPSRRating).astype(int)
        attach_factor = attach_factor.rename(lambda x: 'iv_score_median_' + x, axis=1)
        df_bets = concat([df_bets, attach_factor], axis=1)

        # factor: iv_score_total
        n_levels = df_bets.index.nlevels
        for pos in range(n_levels):
            pos = str(pos)
            cols = [c for c in df_bets.columns if c.startswith('iv_score_') and c.endswith('_' + pos)]
            score_total = 'iv_score_total_' + pos
            # iv_score_total_
            df_bets[score_total] = df_bets[cols].sum(axis=1)

        df_bets['is_score_0'] = df_bets['iv_score_total_0'].map(lambda x: x == 0).astype(int)
        df_bets['is_score_1'] = df_bets['iv_score_total_0'].map(lambda x: x == 1).astype(int)
        df_bets['is_score_2'] = df_bets['iv_score_total_0'].map(lambda x: x == 2).astype(int)
        df_bets['is_score_3'] = df_bets['iv_score_total_0'].map(lambda x: x == 3).astype(int)

        df_bets['num_iv_score_total_0'] = df_bets.groupby('race_id')['is_score_0'].transform(sum)
        df_bets['num_iv_score_total_1'] = df_bets.groupby('race_id')['is_score_1'].transform(sum)
        df_bets['num_iv_score_total_2'] = df_bets.groupby('race_id')['is_score_2'].transform(sum)
        df_bets['num_iv_score_total_3'] = df_bets.groupby('race_id')['is_score_3'].transform(sum)

        return df_bets

    def calibrate(self, prob_model, runners):
        """

        :param prob_model:  name to generate prob_model with and subsequently calibrate bets
        :param runners: runners to be selected (filtered and then re-normalized)
        :return:
        """

        # TODO make functional
        beta = {'EX': 0, 'TR': 1, 'SU': 1}
        sigma = {'EX': 0, 'TR': 0, 'SU': 1}

        # get required prob model
        if prob_model not in self.dr.dfX.columns:
            self.dr.compute_probs(prob_model)

        # get probability model
        win_probs = self.dr.dfX[self.dr.dfX['race_id'] == self.race_id].set_index('runner_id')[prob_model]

        # filter picks
        if len(runners) > 0:
            assert runners[0].count('_') == 3, 'runners must be list of runner_ids'
            runners = set(runners)
            win_probs = win_probs.loc[runners]
        else:
            self.df_bets['harville_%s' % prob_model] = 0
            return None

        # normalize probs
        win_probs = win_probs / win_probs.sum()

        if self.bet_type not in ['WN', 'PL', 'SH']:

            harville_vector = harville(win_probs,
                                       alpha=1,
                                       beta=beta[self.bet_type],
                                       sigma=sigma[self.bet_type])

            # concatenate df_bets and harville probs for target race on index (pos_0, pos_1 ..)
            self.df_bets = concat([self.df_bets, harville_vector], axis=1)

            # rename harville probs
            self.df_bets = self.df_bets.rename({0: 'harville_%s' % prob_model}, axis=1)

        else:
            # concatenate df_bets and win probs for target race on index i.e. pos_0 (runner_id)
            self.df_bets = concat([self.df_bets, win_probs], axis=1)

            # rename harville probs
            self.df_bets = self.df_bets.rename({prob_model: 'harville_%s' % prob_model}, axis=1)

    def expected_value(self):
        """
        Use real-time odds data to calculate expected value of df_bets
        :return: int, Expected Value
        """
        pass

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

    def clear(self):
        """
        for clearing pipeline to run new strategy without clearing data (dr and hist)
        :return:
        """

        self.df_bets = DataFrame()
        self.df_bets_agg = DataFrame()
        self.internal_collection = []
        self.bets = Bets()

    def validate(self):
        if (self.df_bets['iv_score_total_0'] > 3).any():
            raise Exception('scores are greater than 3.. scoring in pipe._factor() is not working as expected. Try'
                            're-loading dr and re-instantiating pipe and re-running strategy.')
