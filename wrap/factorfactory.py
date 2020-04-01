# factor wrapper for generating iv_scores locally

from s3fs.core import S3FileSystem
from pandas import DataFrame, read_csv, date_range, concat
from time import time
from numpy import nan
from horse.betsim.data.db.engines import engine_factors
import sqlalchemy as db
from horse.betsim.strategy.factors import *
import calendar
from datetime import datetime as dt
import inspect

cols_x8race_class = ['x8_track_sym', 'race_race_type', 'race_distance', 'race_surface', 'race_age_sex_restriction']
cols_x8race_class_loosen = ['x8_track_sym', 'race_race_type', 'is_sprint', 'race_surface', 'race_age_sex_restriction']
# TODO incorporate wrap/schema/df_train.csv - 2019/09/20 not sure if what I would use it for
dtype_lean = {'year': int,
              'month': int,
              'weekofyear': int,
              'weekday': str,
              'date': str,
              'x8_track_sym': str,
              'race_id': str,
              'race_race_type': 'category',
              'race_distance': int,
              'race_surface': 'category',
              'race_age_sex_restriction': 'category',
              'x8name': str,
              'jockey': str,
              'trainer': str,
              'runner_HDWPSRRating': float,
              'underperformance_weighted': float,
              'rank_morning_line': float,
              'payout_win': float,
              'payout_place': float,
              'payout_show': float,
              'prob_morning_line': float,
              'entropy_morning_line': float,
              'prob_final_tote_odds': float,
              'entropy_final_tote_odds': float,
              'num_starters_pre': float,
              'num_starters_post': float
              }


class FactorFactory:
    """

    """
    def __init__(self):
        self.df_train = DataFrame()
        self.df_train_winners = DataFrame()
        self.grp_train = DataFrame()

    def load(self, winners=True, lean=True):

        s3 = S3FileSystem(anon=False)
        if lean:
            cols = dtype_lean.keys()
            dtype = dtype_lean
        else:
            cols = None
            dtype = None

        if winners:
            print('betsim.wrap.FactorFactory(): reading df_train_winners from s3')
            t = time()
            s3_fp = 's3://x8-system/df_train_winners.csv'
            self.df_train_winners = read_csv(s3.open(s3_fp, mode='rb'),
                                             usecols=cols,
                                             parse_dates=['date'],
                                             dtype=dtype)
            s = round(time() - t)
            print('ff.load(): loaded %s days of df_train_winners in %s seconds' % (self.df_train_winners['date'].nunique(), s))

            self.grp_train = self.df_train_winners.groupby(cols_x8race_class)

        else:
            print('betsim.wrap.FactorFactory(): reading df_train from s3')
            t = time()
            s3_fp = 's3://x8-system/hist.df_train.csv.gz'
            self.df_train = read_csv(s3.open(s3_fp, mode='rb'),
                                     compression='gzip',
                                     usecols=cols,
                                     parse_dates=['date'],
                                     dtype=dtype)
            s = round(time() - t)
            print('ff.load(): loaded %s days of df_train in %s seconds' % (self.df_train['date'].nunique(), s))

    def add_x8class_score_race(self, dfXrace, grp_train, attr=None, func=None):
        '''
        Uses the grp_train which has races grouped by x8race_class columns to match x8race_class for this race
        dfXrace: dr.dfX.groupby('race_id').get_group(race_id)
        grp_train: df_train_winners.groupby(cols_x8race_class)
        attr, func: can be used at some point for custom functions and attrs, funcs
        '''

        this_x8class = dfXrace.index.values[0]
        this_date = dfXrace['date'].iloc[0]

        try:
            dftrain = grp_train.get_group(this_x8class)
            dftrain = dftrain[dftrain.date < this_date]
            dfXrace['iv_score_jockey'] = dfXrace['jockey'].isin(dftrain.jockey).astype(int)
            dfXrace['iv_score_trainer'] = dfXrace['trainer'].isin(dftrain.trainer).astype(int)
            dfXrace['iv_score_median_runner_HDWPSRRating'] = dfXrace['runner_HDWPSRRating'].map(
                lambda x: int(x > dftrain.runner_HDWPSRRating.median()))
        except KeyError:
            dfXrace['iv_score_jockey'] = 0
            dfXrace['iv_score_trainer'] = 0
            dfXrace['iv_score_median_runner_HDWPSRRating'] = 0

        cols_iv = [c for c in dfXrace.columns if c.startswith('iv_score_')]
        dfXrace['iv_score_total'] = dfXrace[cols_iv].sum(axis=1)

        dfXrace['is_score_0'] = dfXrace['iv_score_total'].map(lambda x: x == 0).astype(int)
        dfXrace['is_score_1'] = dfXrace['iv_score_total'].map(lambda x: x == 1).astype(int)
        dfXrace['is_score_2'] = dfXrace['iv_score_total'].map(lambda x: x == 2).astype(int)
        dfXrace['is_score_3'] = dfXrace['iv_score_total'].map(lambda x: x == 3).astype(int)

        dfXrace['num_iv_score_total_0'] = dfXrace.groupby('race_id')['is_score_0'].transform(sum)
        dfXrace['num_iv_score_total_1'] = dfXrace.groupby('race_id')['is_score_1'].transform(sum)
        dfXrace['num_iv_score_total_2'] = dfXrace.groupby('race_id')['is_score_2'].transform(sum)
        dfXrace['num_iv_score_total_3'] = dfXrace.groupby('race_id')['is_score_3'].transform(sum)

        return dfXrace

    def add_factors(self, df, factors=()):
        """
        add factors that need df_train to compute
        :return:
        """

        self.load(winners=False)

        grp_x8name = self.df_train.groupby('x8name')

        if 'HDWPSRRating_num_consec_non_null' in factors:
            df['HDWPSRRating_num_consec_non_null'] = nan
            df_notnull = df[df.x8name.notnull()].set_index(['date', 'x8name'])
            df = df.set_index(['date', 'x8name'])
            for date, x8name in df_notnull.index:
                this_x8name = grp_x8name.get_group(x8name)
                this_x8name = this_x8name[this_x8name['date'] < date]
                helper = this_x8name.set_index('date').sort_index(ascending=False)['runner_HDWPSRRating'].isnull()
                if helper.all():
                    loc = 0
                else:
                    loc = helper.index.drop_duplicates().get_loc(helper.idxmax())
                df.loc[(date, x8name), 'HDWPSRRating_num_consec_non_null'] = loc
            df = df.reset_index()

        return df

    def iv_stakes_factors(self, dfX):
        t = time()

        connection = engine_factors.connect()
        metadata = db.MetaData()
        historical = db.Table('dr.dfX.historical',
                              metadata,
                              autoload=True,
                              autoload_with=engine_factors)

        statement = db.and_(db.sql.expression.between(db.sql.schema.Column('official_finish_position'), 1, 1))

        query = db.select([historical]).where(statement)
        result_proxy = connection.execute(query)
        result = result_proxy.fetchall()

        sample = DataFrame(result, columns=[c.key for c in historical.columns])
        sample['is_sprint'] = (abs(sample['race_distance']) < 1760).astype(int)
        grp_train = sample.groupby(cols_x8race_class_loosen)

        dfX['is_sprint'] = (abs(dfX['race_distance']) < 1760).astype(int)

        dfs = []
        for race_id, df_race in dfX.set_index(cols_x8race_class_loosen).groupby('race_id'):

            this_x8class = df_race.index.values[0]

            try:
                year = df_race.date.dt.year.iloc[0]
                month = df_race.date.dt.month.iloc[0]
                target_dates = [date_range(dt(year, month, 1), dt(year, month, calendar.monthrange(year, month)[1])) for year in range(2010, year)]
                target_dates = [date for month_dates in target_dates for date in month_dates]

                dftrain = grp_train.get_group(this_x8class)
                dftrain = dftrain[dftrain.date.isin(target_dates)]

                df_race['iv_score_jockey'] = df_race['jockey'].isin(dftrain.jockey).astype(int)
                df_race['iv_score_trainer'] = df_race['trainer'].isin(dftrain.trainer).astype(int)
                df_race['iv_score_median_runner_HDWPSRRating'] = df_race['runner_HDWPSRRating'].map(lambda x: int(x > dftrain.runner_HDWPSRRating.median()))

            except KeyError:
                df_race['iv_score_jockey'] = 0
                df_race['iv_score_trainer'] = 0
                df_race['iv_score_median_runner_HDWPSRRating'] = 0

            cols_iv = [c for c in df_race.columns if c.startswith('iv_score_')]
            df_race['iv_score_total'] = df_race[cols_iv].sum(axis=1)

            df_race['is_score_0'] = df_race['iv_score_total'].map(lambda x: x == 0).astype(int)
            df_race['is_score_1'] = df_race['iv_score_total'].map(lambda x: x == 1).astype(int)
            df_race['is_score_2'] = df_race['iv_score_total'].map(lambda x: x == 2).astype(int)
            df_race['is_score_3'] = df_race['iv_score_total'].map(lambda x: x == 3).astype(int)

            df_race['num_iv_score_total_0'] = df_race.groupby('race_id')['is_score_0'].transform(sum)
            df_race['num_iv_score_total_1'] = df_race.groupby('race_id')['is_score_1'].transform(sum)
            df_race['num_iv_score_total_2'] = df_race.groupby('race_id')['is_score_2'].transform(sum)
            df_race['num_iv_score_total_3'] = df_race.groupby('race_id')['is_score_3'].transform(sum)

            dfs.append(df_race)

        print('%s() took %s seconds' % (inspect.stack()[0][3], time() - t))

        return concat(dfs)

    def iv_stakes_factors_mini(self, dfX):
        # this implementation is faster for less races
        # TODO figure out what the race number threshold where this is faster

        raise Exception('not working. see code..')

        connection = engine_factors.connect()
        metadata = db.MetaData()
        historical = db.Table('dr.dfX.historical',
                              metadata,
                              autoload=True,
                              autoload_with=engine_factors)

        dfX['is_sprint'] = (abs(dfX['race_distance']) < 1760).astype(int)

        t = time()
        dfs = []
        for race_id, df_race in dfX.groupby('race_id'):

            # target dates
            year = df_race.date.dt.year.iloc[0]
            month = df_race.date.dt.month.iloc[0]
            target_dates = [date_range(dt(year, month, 1), dt(year, month, calendar.monthrange(year, month)[1])) for year in range(2010, year)]
            target_dates = [date for month_dates in target_dates for date in month_dates]

            params = merge_dicts(isin('x8_track_sym', [df_race.x8_track_sym.iloc[0]]),
                                 isin('race_race_type', [df_race.race_race_type.iloc[0]]),
                                 isin('race_surface', [df_race.race_surface.iloc[0]]),
                                 isin('race_age_sex_restriction', [df_race.race_age_sex_restriction.iloc[0]]),
                                 between('official_finish_position', 1, 1))

            # took this code from Simulator()
            filters = []
            for param, vals in params.items():
                attr = '_'.join(param.split('_')[2:])
                filter_type = param.split('_')[1]
                if filter_type == 'between':
                    filters.append(db.sql.expression.between(db.sql.schema.Column(attr), vals[0], vals[1]))
                elif filter_type == 'isin':
                    filters.append(db.sql.schema.Column(attr).in_(vals))
                else:
                    raise Exception('broken')

            statement = db.and_(db.sql.schema.Column('date').in_(target_dates), *filters)

            query = db.select([historical]).where(statement)
            result_proxy = connection.execute(query)
            result = result_proxy.fetchall()

            sample = DataFrame(result, columns=[c.key for c in historical.columns])

            # distance
            sample['is_sprint'] = (abs(sample['race_distance']) < 1760).astype(int)
            sample = sample[sample['is_sprint'] == df_race['is_sprint'].iloc[0]]

            df_race['iv_score_jockey'] = df_race['jockey'].isin(sample.jockey).astype(int)
            df_race['iv_score_trainer'] = df_race['trainer'].isin(sample.trainer).astype(int)
            df_race['iv_score_median_runner_HDWPSRRating'] = df_race['runner_HDWPSRRating'].map(lambda x: int(x > sample.runner_HDWPSRRating.median()))

            dfs.append(df_race)

        print('%s() took %s seconds' % (inspect.stack()[0][3], time() - t))

        return concat(dfs)
