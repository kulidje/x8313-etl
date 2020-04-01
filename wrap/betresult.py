# wrapper for bet_result sql table in x8_factors

import os
from horse.betsim.data.db.engines import engine_factors
from pandas import DataFrame, DatetimeIndex, read_sql, read_csv, merge, to_datetime
from horse.betsim import data
from datetime import timedelta


map_bet_type_to_wager_name = {'WN': 'Win', 'PL': 'Place', 'EX': 'Exacta', 'SH': 'Show', 'TR': 'Trifecta',
                              'SU': 'Superfecta'}


class BetResult:
    """
    This class will pull all rows from  the betsim.data.db.engines.engine_factors and filter based on the datelist that
    is passed in.

    Bet States
    1) Generated
    2) Attempted: passed risk checks in Controller()
    2.5) Bye Bye: Could not send to Tote because of FileBet/VPN Error
    3) Response: Hit Tote, accepted or rejected (this file)
    """

    def __init__(self):
        self.df_bet_result = DataFrame()
        self.df = DataFrame()
        # accepted by Tote
        self.df_live = DataFrame()
        # rejected by Tote
        self.df_rejected = DataFrame()

        # datetime str format
        self.dt_format = '%Y%m%d.%H%M%S'

        # dict for normalizing track symbols
        track_detail = os.path.join(data.__path__._path[0], 'track_detail.csv')
        dftrack = read_csv(track_detail)
        self.map_itsp_to_x8 = dftrack.set_index('itsp_track_sym')['x8_track_sym'].to_dict()

        # load entire bet result table
        self.load_all()

    def load_all(self):
        # load all rows in engine_factors.bet_result
        self.df_bet_result = read_sql('bet_result', engine_factors)

        # dtype and normalizations
        self.df_bet_result['bet_count'] = self.df_bet_result['bet_count'].astype(int)
        self.df_bet_result['error_code'] = self.df_bet_result['error_code'].astype(int)
        self.df_bet_result['bet_amount'] = self.df_bet_result['bet_amount'].fillna(0.0).map(float)
        self.df_bet_result['total_bet_cost'] = self.df_bet_result['total_bet_cost'].fillna(0.0).map(float)
        self.df_bet_result['wager_name'] = self.df_bet_result['bet_type'].map(map_bet_type_to_wager_name)

    def load(self, datelist, strategy_name=None):
        """
        Loads all rows in bet_result in the database for the dates
        Make date column
        Filter by datelist: greater than min value in datelist and less than max value in datelist
        """

        datelist = [d.date() for d in datelist]

        # Filter bet_result by date
        self.df = self.df_bet_result[self.df_bet_result['date'].isin(datelist)]
        # Filter bet_result by strategy_name if given
        if strategy_name:
            self.df = self.df[self.df['strategy_name'] == strategy_name]

        # reset index after filtering
        self.df = self.df.reset_index(drop=True)
        self.df_live = self.df[self.df['total_bet_cost'] > 0]
        self.df_rejected = self.df[self.df['error_code'] != 0]

        self._validate()

    def _validate(self):
        # make sure everything is as expected
        if len(self.df_rejected) + len(self.df_live) != len(self.df):
            raise Exception('We are losing bets/rows somewhere in the filtering of accepted/rejected bets.\n'
                            'df_live + df_rejected should equal df_bet_result.')
