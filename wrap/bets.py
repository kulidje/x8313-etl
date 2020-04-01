# wrapper for df_bets

import os
from pandas import DataFrame, Series, to_datetime, Index, read_csv, concat
from horse.betsim.data.db.engines import engine_factors
from horse.betsim.models.probability import HarvilleTri as harville
from datetime import datetime
import numpy as np
from horse.betsim import data
from s3fs.core import S3FileSystem
import functools

# amtote required cols
strat_cols = ['bet_amount', 'bet_type', 'runners']
required_cols = ['unique_id', 'itsp_track_sym', 'race_num', 'date']
# x8 required cols - for Controller() - eg. needs race_id so we can do risk calcs by race in Controller()
x8_required_cols = ['race_id']

condense_cols = ['bet_type', 'race_id', 'itsp_track_sym', 'date', 'race_num', 'bet_amount', 'is_modified']
all_required_cols = strat_cols + required_cols + x8_required_cols

pool_single = ['WN', 'PL', 'SH']


class Ticket:
    """
    for wrapping a row of a df_bets or a single wager
        - must include at least all required columns
    """
    def __init__(self):
        self.row = Series()


class Bets:
    """
    for wrapping df_bets object, which is defined as a df a bets, that is each row is a separate bet
    df_bets will be comprised of Ticket objects
    - must not include duplicate bets (1 row per bet)
    """
    def __init__(self):
        self.df_bets = DataFrame()
        self.df_bets_condensed = DataFrame()

        self.dt_format = '%Y%m%d.%H%M%S'

        # default strategy_name name
        self.strategy_name = 'x8default'

        # map
        track_detail = os.path.join(data.__path__._path[0], 'track_detail.csv')
        dftrack = read_csv(track_detail)
        self.map_track_x8_itsp = dftrack.set_index('x8_track_sym')['itsp_track_sym'].to_dict()

    def load(self, df_bets, strategy_name='x8default'):
        """
        assign required df_bets columns depending on which columns are missing
        :param df_bets: df with df_bets minimum columns or more
        :param strategy_name: optional argument, if given assigns strategy_name column to df or re-writes strategy_name column, if not given default strategy_name name is assigned
        """
        missing_cols = set(strat_cols + x8_required_cols) - set(df_bets.columns)
        if missing_cols:
            raise Exception("Minimum required cols not given: %s missing: %s" % (strat_cols, missing_cols))

        # instantiate object
        self.df_bets = df_bets.copy()

        # derive date, race_num, itsp from race_id
        if 'date' not in self.df_bets.columns:
            self.df_bets['date'] = self.df_bets['race_id'].map(lambda x: to_datetime(x.split('_')[1], format='%Y%m%d'))
        if 'race_num' not in self.df_bets.columns:
            self.df_bets['race_num'] = self.df_bets['race_id'].map(lambda x: x.split('_')[2])
        if 'itsp_track_sym' not in self.df_bets.columns:
            self.df_bets['itsp_track_sym'] = self.df_bets['race_id'].map(lambda x: self.map_track_x8_itsp[x.split('_')[0]])

        # strategy_name name pre-pend unique_id
        self.strategy_name = strategy_name
        if len(self.strategy_name) > 17:
            raise Exception('strategy_name name too long. 17 character limit. len(strategy_name)==%s' % len(strategy_name))

        self.df_bets = self.add_unique_id(self.df_bets, strategy_name)

        # validate
        self._validate_df_bets()

    def queue_bets_s3(self, filename):
        """
        write self.df_bets to s3://x8-bucket/bets/
        test mode is not supported for s3 bet queueing
        """

        # we want to see what automatically generated betfiles output regardless of being empty or not
        # if self.df_bets.empty:
        #     raise Exception('bets.df_bets is empty')

        s3 = S3FileSystem(anon=False)

        s3_path = 'x8-bucket/bets/%s' % filename
        print('writing bets.df_bets to %s' % s3_path)

        bytes = self.df_bets.to_csv(None, index=False).encode()

        with s3.open(s3_path, 'wb') as f:
            f.write(bytes)

    def queue_bets(self, test_mode):
        """
        write self.df_bets to x8_factors.queued_bets and x8_factors.queued_bets_sim if in testing mode
        """

        if test_mode:
            if self.df_bets.empty:
                raise Exception('bets.df_bets is empty')
            else:
                print('testing mode is on, queueing bets in x8_factors.queued_bets_sim..')
                df_queued_bets = self.df_bets[all_required_cols]
                df_queued_bets.to_sql('queued_bets_sim', engine_factors, if_exists='append', index=False)

        else:
            if self.df_bets.empty:
                raise Exception('bets.df_bets is empty')
            else:
                print('testing mode is off, queueing bets in x8_factors.queued_bets..')
                df_queued_bets = self.df_bets[all_required_cols]
                df_queued_bets.to_sql('queued_bets', engine_factors, if_exists='append', index=False)

    def bet_matrix(self):
        """
        construct bet matrix from df_bets - where index is x8name and columns are wagered finish position
        WN == 1
        PL == 1 and 2
        SH == 1 and 2 and 3
        EX == runner_A 1 and runner_B 2
        TR == runner_A 1 and runner_B 2 runner_C 3
        DB == TODO don't know how to do this yet
        :return: bet_matrix
        """

        if self.df_bets['bet_type'].unique() != 'WN':
            raise Exception('This method is only working with df_bets with only Win bets at the moment.')

        bet_matrix = self.df_bets.groupby('x8name')['bet_amount'].sum().rename({'bet_amount': 1}).to_frame()

        return bet_matrix

    def condense(self):
        """
        condense straight wager df_bets (1 row per bet) into condensed df_bets using bet modifiers
        """

        # TODO unt test test/test_bets_condense.py

        df_bets = self.df_bets.reset_index()
        df_bets['is_modified'] = df_bets['runners'].map(lambda x: x in ['AL', 'WT'])

        # this groupby will divide the df_bets into groups (bet_type, race_id, bet_amount, bet_modified)
        df_bets_group = df_bets.groupby(condense_cols)

        for idx, df in df_bets_group:
            # idx = ('P4', 'CTX_20181101_1', 'TWN', datetime.date(2018, 11, 1), 1, 1.0, 'x8default_20181101.2243', False)
            bet_type = idx[0]
            race_id = idx[1]
            itsp = idx[2]
            date = idx[3]
            race_num = idx[4]
            bet_amount = idx[5]
            strat = idx[6]
            modified = idx[7]

            if (idx[1] in pool_single) or modified:
                self.df_bets_condensed = concat([self.df_bets_condensed, df])
            else:
                n_positions = {'EX': 2, 'TR': 3, 'SU': 4, 'DB': 2, 'P3': 3, 'P4': 4, 'P5': 5, 'P6': 6}
                groups = []
                for pos in range(n_positions[bet_type]):
                    all_runner_programs = df['program_number_%s' % pos].unique()
                    groups.append(','.join(all_runner_programs))

                runners = ',WT,'.join(groups)

                df.loc[0, 'runners'] = runners
                df.loc[0, 'race_id'] = race_id
                df.loc[0, 'itsp_track_sym'] = itsp
                df.loc[0, 'date'] = date
                df.loc[0, 'race_num'] = race_num
                df.loc[0, 'bet_amount'] = bet_amount
                df.loc[0, 'strategy_name'] = strat

                self.df_bets_condensed = self.df_bets_condensed.append(df.loc[0])

    def add_unique_id(self, df_bets, strategy_name):
        """
        assign 'unique_id' column to df_bets
        convention: timestamp + '_' + (index + 1) eg. 20180509.182019_1 to 20180509.182019_n

        timestamp is time that bets are generated, not time when bets are sent.
        """
        timestamp = datetime.utcnow().strftime(self.dt_format)
        index = Index(range(1, len(df_bets) + 1)).astype(str)

        df_bets['unique_id'] = strategy_name + '_' + timestamp + '_' + index

        return df_bets

    def _check_unique_id(self):
        """
        check if unique_id has been correctly formatted
        :return: True if correct / False if wrong
        """

        # first part of unique_id
        uid_timestamp = self.df_bets['unique_id'].map(lambda x: x.split('_')[-2])
        try:
            to_datetime(uid_timestamp, format=self.dt_format)
            return True
        except ValueError:
            return False

    def _validate_df_bets(self):
        """
        validates the DataFrame input is of the correct format
        """
        # check that expected columns exist
        df_cols = self.df_bets.columns
        missing_cols = set(all_required_cols) - set(df_cols)
        if missing_cols:
            raise Exception('Bets._validate_df_bets() ERROR input DataFrame missing columns: %s' % missing_cols)

        # TODO wont work with df_bets from dr.generate_df_bets() because using betting_interest i.e. bet 1 twice because 1 and 1A
        # check that there are no duplicate rows
        #if len(self.df_bets) != len(self.df_bets[['race_id', 'runners']].drop_duplicates()):
        #    raise Exception('duplicate rows exist')

        # make sure unique_id column is as expected
        if not self._check_unique_id():
            raise ValueError('unique_id column is not correctly formatted')
