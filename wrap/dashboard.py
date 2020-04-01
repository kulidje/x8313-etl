# wrapper for X8 Dashboard - Front End

from horse.betsim.wrap.daily import DailyRaces
from pandas import read_csv, DataFrame, concat
import numpy as np
import os
from horse.betsim import data

class Dashboard:
    """
    spec
    """
    def __init__(self, verbose=False):
        # data
        self.dr = DailyRaces(state='play')

        # track detail
        track_detail = os.path.join(data.__path__._path[0], 'track_detail.csv')
        self.dftrack = read_csv(track_detail)
        self.map_track_chart_to_x8 = self.dftrack.set_index('chart_file_sym')['x8_track_sym'].to_dict()
        self.map_track_x8_to_chart = self.dftrack.set_index('x8_track_sym')['chart_file_sym'].to_dict()
        self.verbose = verbose
    def load(self, dfbets):

        date = dfbets.iloc[0]
        self.dr.load([date], dfbets.x8_track_sym.unique())
    def make_empty_risk_matrix(self, df_race):
        """
        return empty risk matrix for df_race
        :param df_race: dataframe containing single race (every runner in a race)
        :return: empty risk matrix
        """
        race_range = len(df_race)
        risk_matrix = DataFrame(index=df_race.x8name,
                                data=np.zeros((race_range, 4)),
                                columns=range(1, 5))

        return risk_matrix

    def make_risk_matrix(self, df_bets):
        """
        given df_bets, return risk matrix
        :param df_bets:
        :return: risk matrix
        """
        empty_risk_matrix = self.make_empty_risk_matrix()

        # return empty risk matrix if df_bets is empty
        if df_bets.empty:
            return empty_risk_matrix

        def win(df_bets):
            risk_matrix = df_bets.groupby('x8name_0')['bet_amount'].sum().to_frame().rename({'bet_amount': 1}, axis=1)
            risk_matrix[2] = 0
            risk_matrix[3] = 0
            risk_matrix[4] = 0
            return risk_matrix

        def place(df_bets):
            risk_matrix = df_bets.groupby('x8name_0')['bet_amount'].sum().to_frame().rename({'bet_amount': 1}, axis=1)
            risk_matrix[2] = risk_matrix[1].copy()
            risk_matrix[3] = 0
            risk_matrix[4] = 0

            return risk_matrix

        def show(df_bets):
            risk_matrix = df_bets.groupby('x8name_0')['bet_amount'].sum().to_frame().rename({'bet_amount': 1}, axis=1)
            risk_matrix[2] = risk_matrix[1].copy()
            risk_matrix[3] = risk_matrix[1].copy()
            risk_matrix[4] = 0
            return risk_matrix

        def exacta(df_bets):
            risk_matrix = df_bets.groupby('x8name_0')['bet_amount'].sum().to_frame().rename({'bet_amount': 1}, axis=1)
            risk_matrix[2] = df_bets.groupby('x8name_1')['bet_amount'].sum().to_frame()['bet_amount']
            risk_matrix[3] = 0
            risk_matrix[4] = 0
            return risk_matrix

        def trifecta(df_bets):
            risk_matrix = df_bets.groupby('x8name_0')['bet_amount'].sum().to_frame().rename({'bet_amount': 1}, axis=1)
            risk_matrix[2] = df_bets.groupby('x8name_1')['bet_amount'].sum().to_frame()['bet_amount']
            risk_matrix[3] = df_bets.groupby('x8name_2')['bet_amount'].sum().to_frame()['bet_amount']
            risk_matrix[4] = 0
            return risk_matrix

        def superfecta(df_bets):
            risk_matrix = df_bets.groupby('x8name_0')['bet_amount'].sum().to_frame().rename({'bet_amount': 1}, axis=1)
            risk_matrix[2] = df_bets.groupby('x8name_1')['bet_amount'].sum().to_frame()['bet_amount']
            risk_matrix[3] = df_bets.groupby('x8name_2')['bet_amount'].sum().to_frame()['bet_amount']
            risk_matrix[4] = df_bets.groupby('x8name_3')['bet_amount'].sum().to_frame()['bet_amount']
            return risk_matrix

        bet_function = {'WN': win, 'PL': place, 'SH': show, 'EX': exacta, 'TR': trifecta, 'SU': superfecta}

        matricies = []
        for bet_type, df in df_bets.groupby('bet_type'):
            df_matrix = bet_function[bet_type](df)
            matricies.append(df_matrix)

        risk_matrix = sum(matricies)
        risk_matrix.index.names = ['x8name']

        return empty_risk_matrix + risk_matrix

    def get_df_bets(self, race_id):
        """
        get arbitrary df_bets
        """

        # get test df_bets with x8name
        self.dr.generate_df_bets(bet_amount=2.0, bet_type='WN', race_id_start=race_id)
        df_bets = self.dr.bets.df_bets.copy()
        self.dr.generate_df_bets(bet_amount=2.0, bet_type='EX', race_id_start=race_id)
        df_bets = concat([df_bets, self.dr.bets.df_bets])

        return df_bets

    def get_json(self, race_id):
        """
        given race_id, return data for front-end risk viewer in json format
        :param race_id: ABC_20180101_1
        :return: json
        """
        symbol = race_id.split('_')[0]
        name = self.dr.dftrack.set_index('x8_track_sym')['jcp_track_name'].to_dict()[symbol]

        df_bets = self.get_df_bets(race_id)
        bets_matrix = self.make_risk_matrix(df_bets)

        json = bets_matrix.to_json()

        return json


# group = dr.ww.df_entries.groupby('race_id')
# empty_risk = group.apply(make_empty_risk_matrix)  # TODO make attribute of daily races or entries (decide where it goes)
#
#
# target_race_id = dr.dfX.race_id.unique()[0]
# spec = get_json(target_race_id)

# TODO request1(all races of day) => all races of day
# TODO request2(race_id) => wager_matrix and relevant info
