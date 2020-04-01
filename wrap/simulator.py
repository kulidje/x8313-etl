
from horse.betsim.data.db.engines import engine_factors
from pandas import DataFrame, ExcelWriter, read_pickle, MultiIndex, concat, merge, read_csv, to_datetime
import sqlalchemy as db
import os
import horse
from time import time
from datetime import datetime as dt
from horse.bin.debug.simulate_vector import assign_harville_probs, assign_log_spread
from horse.betsim.wrap.factorfactory import FactorFactory
from horse.betsim import data
from matplotlib import pyplot as plt
from horse.betsim.math import add_probsT4


class Simulator:
    """
    SQL Alchemy Tutorial: https://towardsdatascience.com/sqlalchemy-python-tutorial-79a577141a91

    1. read dates from target dates
    2. filter using sql query
    3. add signal 0/1 columns
    4. output file in excel w/ PivotTable

    # TODO iteratively plot strategies using Plotly
    """
    def __init__(self) -> object:
        self.df = DataFrame()
        self.result = []
        self.historical = None

        self.df_bets = DataFrame()
        self.df_payouts = DataFrame()

        # factors
        self.ff = FactorFactory()

        # caching directory for storing pickled dataframe i.e. /x8313/sim_cache/
        horse_path = os.path.dirname(horse.__path__[0])

        # make caching directory if it does not already exist
        self.cache_path = os.path.join(horse_path, 'sim_cache')
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        # make output directory if it does not already exist
        self.output_path = os.path.join(horse_path, 'sim_output')
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # rebates
        track_detail = os.path.join(data.__path__._path[0], 'track_detail.csv')
        dftrack = read_csv(track_detail)
        self.map_x8track_rebate_pct = dftrack.set_index('x8_track_sym').to_dict()

        self.date_min = ''
        self.date_max = ''

        self.fp_output = ''

    def simulate(self, datelist, strategy={}, bet_type='WN', output=False):

        self.date_min = datelist.min().strftime('%Y%m%d')
        self.date_max = datelist.max().strftime('%Y%m%d')

        # TODO do this last: smart caching i.e. if only some of targeted data is stored, use it and load rest from db
        # TODO before doing above: compare time to load pickle vs db for varying data set sizes
        df_path = os.path.join(self.cache_path, '%s_%s.pkl' % (self.date_min, self.date_max))

        if os.path.exists(df_path):
            self.df = read_pickle(df_path, compression='gzip')

        else:
            t = time()
            connection = engine_factors.connect()
            metadata = db.MetaData()
            self.historical = db.Table('dr.dfX.historical',
                                       metadata,
                                       autoload=True,
                                       autoload_with=engine_factors)

            pos_params = {x: y for x, y in strategy.items() if '_pos_' in x}
            [strategy.pop(key, None) for key in pos_params]
            harville_params = {x: y for x, y in strategy.items() if 'harville' in x}
            [strategy.pop(key, None) for key in harville_params]
            filter_params = {x: y for x, y in strategy.items() if x.startswith('filter') and '_pos_' not in x}
            view_params = {x: y for x, y in strategy.items() if x.startswith('view') and '_pos_' not in x}

            # This is where we construct a SQL statement:

            # if there were no filter params in strategy then only filter datelist from SQL
            if not filter_params:
                statement = db.sql.schema.Column('date').in_(datelist)
            # otherwise make SQL WHERE AND statement using datelist and all filter params
            else:
                filters = []
                for param, vals in filter_params.items():
                    attr = '_'.join(param.split('_')[2:])  # target column name
                    filter_type = param.split('_')[1]  # 'isin' or 'between'
                    if filter_type == 'between':
                        filters.append(db.sql.expression.between(db.sql.schema.Column(attr), vals[0], vals[1]))
                    elif filter_type == 'isin':
                        filters.append(db.sql.schema.Column(attr).in_(vals))
                    else:
                        raise Exception('broken')

                statement = db.and_(db.sql.schema.Column('date').in_(datelist), *filters)

            query = db.select([self.historical]).where(statement)
            # query = query.select_from(census.join(state_fact, census.columns.state == state_fact.columns.name))
            result_proxy = connection.execute(query)
            self.result = result_proxy.fetchall()

            self.df = DataFrame(self.result, columns=[c.key for c in self.historical.columns])
            print('Simulator.simulate() read db in %s seconds' % round(time() - t, 4))

            # computing view parameter columns
            t = time()
            for param, vals in view_params.items():
                attr = '_'.join(param.split('_')[2:])  # target column name
                filter_type = param.split('_')[1]  # 'isin' or 'between'
                if filter_type == 'between':
                    self.df[param] = self.df[attr].between(vals[0], vals[1]).astype(int)
                elif filter_type == 'isin':
                    self.df[param] = self.df[attr].isin(vals).astype(int)
                else:
                    raise Exception('broken')

            print('Simulator.simulate() computed view columns in %s seconds' % round(time() - t, 4))

            # smart caching
            # self.df.to_pickle(df_path, compression='gzip')  # cache results

        # TODO remove: drop scratches and re-normalize
        #self.df = self.df[self.df.prob_final_tote_odds.notnull()]
        #self.normalize_probs()

        # extra factors that are not in database
        self._compute_extra_factors()

        # generate all possible permutation for target races from self.df
        if bet_type in ['EX', 'TR']:
            print('generating all %s permutations for %s target races in sim.df..' %
                  (bet_type, self.df.race_id.nunique()))
            prob_models = [key[key.find('prob_'):] for key in harville_params.keys()]
            map_n_leg = {'EX': 2, 'TR': 3}
            # iterate through all races in self.df
            for race_id, df in self.df.set_index('runner_id').groupby('race_id'):
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

                # TODO do not create positional columns for race level factors - from hard coded list of race factors I think is easiest way for now
                # TODO only create race level factors for target factors in strategy
                pos_vals = [df.add_suffix('_%s' % i)
                              .reindex(idx, level='pos_%s' % i)
                            for i in range(map_n_leg[bet_type])]

                df_merge = concat(pos_vals, axis=1)

                # delete this eventually
                df_merge['race_id'] = df_merge['race_id_0'].copy()
                df_merge['x8_track_sym'] = df_merge['x8_track_sym_0'].copy()
                # do harville stuff
                df_merge = assign_harville_probs(df_merge, prob_models, bet_type)

                # concatenate
                self.df_bets = concat([self.df_bets, df_merge])

            # TODO filter self.df_bets if necessary (same as filtering self.df)

            # load and merge with payouts
            connection = engine_factors.connect()
            metadata = db.MetaData()
            payouts = db.Table('exotics_payout',
                               metadata,
                               autoload=True,
                               autoload_with=engine_factors)

            races = self.df_bets.race_id.unique()
            statement = db.and_(db.sql.schema.Column('race_id').in_(races),
                                db.sql.schema.Column('bet_type').in_([bet_type]))
            query = db.select([payouts]).where(statement)
            result_proxy = connection.execute(query)
            result = result_proxy.fetchall()

            self.df_payouts = DataFrame(result, columns=[c.key for c in payouts.columns])
            self.df_payouts = self.df_payouts[~self.df_payouts['0'].isnull()]  # delete bad rows
            self.df_payouts = self.df_payouts.drop_duplicates(['race_id', 'winning_pgm'])

            # merge payout data with df_bets
            d = {'EX': 2, 'TR': 3, 'SU': 4}
            self.df_payouts.set_index([str(i) for i in list(range(d[bet_type]))], inplace=True)
            self.df_payouts.index.names = ['pos_' + name for name in self.df_payouts.index.names]
            self.df_bets = merge(self.df_bets, self.df_payouts.drop(columns='race_id'), left_index=True, right_index=True, how='left').fillna({'payout_norm': 0})

            # compute log_spread factors with all given prob_models
            self.df_bets = assign_log_spread(self.df_bets, prob_models)

        single_leg = ['WN', 'PL', 'SH']
        if bet_type in single_leg:
            for bet_type in single_leg:
                self._compute_pnl(bet_type)
        else:
            self._compute_pnl(bet_type)

        if output:
            now = dt.utcnow().strftime('%Y%m%d')
            self.fp_output = os.path.join(self.output_path, 'dr.dfX.historical.%s_%s_%s.xlsx' % (self.date_min, self.date_max, now))
            self.output(self.fp_output)

        self._validate()

    def get_ww_odds(self):
        t = time()
        connection = engine_factors.connect()
        metadata = db.MetaData()
        odds = db.Table('ww.df_win_odds',
                        metadata,
                        autoload=True,
                        autoload_with=engine_factors)

        statement = db.and_(db.sql.schema.Column('race_id').in_(self.df.race_id.unique()))

        query = db.select([odds]).where(statement)
        result_proxy = connection.execute(query)
        result = result_proxy.fetchall()

        df = DataFrame(result, columns=[c.key for c in odds.columns])
        print('Simulator.simulate() computed view columns in %s seconds' % round(time() - t, 4))

        return df

    def _compute_extra_factors(self):
            """
            - for adding factors to self.df that are not in the database.
            - use this sparingly because it increase run time every computed factors
            # TODO use this method as a function of 'strategy_dict' param in self.simulate()
            """

            self.df['ml_entropy_in_range'] = self.df['entropy_prob_morning_line_post'].between(0.85, 0.92).values.astype(int)
            self.df['sum_coupled_entries'] = self.df['coupled_type'].fillna('X').map({'A': 1, 'B': 1, 'X': 0})
            self.df['exclude'] = self.df.groupby('race_id')['sum_coupled_entries'].transform(lambda x: x.sum() > 0).astype(int)

            self.df['rank_prob_morning_line_post'] = self.df.groupby('race_id')['prob_morning_line_post'].rank(ascending=False, method='first')

            # self.df = add_probsT4(self.df, 'prob_runner_HDWPSRRating')
            # self.df = add_probsT4(self.df, 'prob_final_tote_odds')
            # self.df = add_probsT4(self.df, 'prob_morning_line_post')

            #self.df = self.ff.iv_stakes_factors(self.df)

    def normalize_probs(self):
        """
        re-normalize raw probabilities AFTER filtering scratches
        and re-normalize ranks
        """

        cols = [c for c in self.df.columns if c.startswith('prob_')]

        for c in cols:
            self.df[c + '_norm'] = self.df[c] / self.df.groupby('race_id')[c].transform(sum)

        # self.df = add_probsT4(self.df, 'prob_runner_HDWPSRRating_norm')
        # self.df = add_probsT4(self.df, 'prob_final_tote_odds_norm')
        # self.df = add_probsT4(self.df, 'prob_morning_line_post_norm')

    def re_populate(self):
        """
        re-build sim.df if deleted rows and you want original data back
        """

        self.df = DataFrame(self.result, columns=[c.key for c in self.historical.columns])

    def _compute_pnl(self, bet_type):
        """
        - payout vector operations
        - rebates

        eventually add risk_vector param i.e. risk_vector=(df_sample['harville_prob_final_tote_odds'] * 100).fillna(0).map(lambda x: ceil(x)).values
        """
        if bet_type in ['WN', 'PL', 'SH']:
            self.df['bet_amount'] = 1  # TODO should be risk vector parameter

            pos = {'WN': 1, 'PL': 2, 'SH': 3}
            col = {'WN': 'payout_win', 'PL': 'payout_place', 'SH': 'payout_show'}

            self.df['is_hit_%s' % bet_type] = (self.df['official_finish_position'].values < pos[bet_type] + 1).astype(int)
            self.df['pnl_%s' % bet_type] = ((self.df[col[bet_type]].fillna(0).values / 2) * self.df['bet_amount']) - self.df.bet_amount.values

            self.df['rebate_pct_%s' % bet_type] = self.df['x8_track_sym'].map(self.map_x8track_rebate_pct[bet_type]).fillna(0)
            self.df['rebate_%s' % bet_type] = self.df['bet_amount'] * self.df['rebate_pct_%s' % bet_type]

            # final calc
            self.df['net_return_%s' % bet_type] = self.df['pnl_%s' % bet_type].values + self.df['rebate_%s' % bet_type]

        else:
            self.df_bets['bet_amount'] = 1

            self.df_bets['gross_payout'] = self.df_bets['payout_norm'] * self.df_bets['bet_amount']
            self.df_bets['pnl'] = self.df_bets['gross_payout'].values - self.df_bets['bet_amount']

            self.df_bets['rebate_pct'] = self.df_bets['x8_track_sym'].map(self.map_x8track_rebate_pct[bet_type]).fillna(0)
            self.df_bets['rebate'] = self.df_bets['bet_amount'] * self.df_bets['rebate_pct']

            # final calc
            self.df_bets['net_return'] = self.df_bets['pnl'].values + self.df_bets['rebate']

    def output(self, fp):
        """
        output to excel for exploring
        """
        # TODO automatically make pivot table w/ signal columns
        t = time()

        # fp = '/Users/andrewkulidjian/projects/x8313/sim_output/test.xlsx'  # for testing

        front = ['race_id', 'date', 'runner_id']
        re_order = front + list(self.df.columns.drop(front))
        sheet_name = 'space'

        writer = ExcelWriter(fp, engine='xlsxwriter')
        self.df[re_order].to_excel(writer, sheet_name=sheet_name, index=False)
        writer.sheets[sheet_name].freeze_panes(1, 1)
        writer.save()
        print('Simulator.simulate() wrote dataset as excel file in %s seconds' % round(time() - t, 4))

    def standard_report(self, df, strategy_column):
        """M.E.D. reporting and plotting"""
        df['is_top4'] = (df['official_finish_position'].values < 5).astype(int)

        df_plot = df[df[strategy_column]].groupby('date').agg({'runner_id': 'size',
                                                               'pnl_WN': sum,
                                                               'pnl_PL': sum,
                                                               'pnl_SH': sum,
                                                               'is_hit_WN': sum,
                                                               'is_hit_PL': sum,
                                                               'is_hit_SH': sum,
                                                               'is_top4': sum})

        df_plot.rename(columns={'runner_id': 'Plays'}, inplace=True)
        df_plot['WN'] = round(df_plot['is_hit_WN'] / df_plot['Plays'], 2)
        df_plot['PL'] = round(df_plot['is_hit_PL'] / df_plot['Plays'], 2)
        df_plot['SH'] = round(df_plot['is_hit_SH'] / df_plot['Plays'], 2)
        df_plot['ITM4'] = round(df_plot['is_top4'] / df_plot['Plays'], 2)
        df_plot['WN_ROI'] = round(df_plot['pnl_WN'] / (df_plot['Plays'] * 2), 2)
        df_plot['PL_ROI'] = round(df_plot['pnl_PL'] / (df_plot['Plays'] * 2), 2)
        df_plot['W+P_ROI'] = round((df_plot['pnl_WN'] + df_plot['pnl_PL']) / df_plot['Plays'], 2)
        df_plot['SH_ROI'] = round(df_plot['pnl_SH'] / (df_plot['Plays'] * 2), 2)
        df_plot['PNL_WN'] = df_plot['pnl_WN']
        df_plot['PNL_PL'] = df_plot['pnl_PL']
        df_plot['PNL_SH'] = df_plot['pnl_SH']
        df_plot['day'] = to_datetime(df_plot.index).weekday_name

        path_df_plot = 'x8_Sreporting_%s_%s_%s' % (strategy_column, self.date_min, self.date_max)
        print('writing %s.csv' % path_df_plot)
        df_plot.to_csv('%s.csv' % path_df_plot, index=True)

        df[df[strategy_column]].groupby('date')[['net_return_WN', 'net_return_PL', 'net_return_SH']].sum().cumsum().plot()
        plt.show()

        path_raw_data = '%s_raw_data.csv' % path_df_plot
        print('printing raw data: %s' % path_raw_data)
        df.to_csv('%s.zip' % path_raw_data, index=False, compression='zip')

    def _validate(self):

        is_duplicates = self.df[self.df.runner_id.duplicated()].runner_id
        if not is_duplicates.empty:
            raise Exception('duplicate runners were loaded: %s' % is_duplicates)
