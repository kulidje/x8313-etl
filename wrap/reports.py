# wrapper for PNL Report Files - profit and loss statement
# daily 'WatchandWager Business Trading' email from info@wawbusinesstrading.com w/ CSV Report attached

import os
from pandas import DataFrame, read_csv, concat, to_datetime
from horse.betsim import data
from s3fs.core import S3FileSystem
from warnings import warn
from horse.betsim.wrap.betresult import BetResult

# rename dict
column_rename = {'accountnumber': 'account_number', 'meetingdate': 'meeting_date', 'venuename': 'venue_name',
                 'venuecountry': 'venue_country', 'race': 'race_num', 'bettype': 'bet_type', 'grosssales': 'gross_sales',
                 'rebaterate': 'rebate_rate', 'turnover': 'bet_amount', 'code': 'itsp_track_sym', 'tsn': 'serial_number'}

map_bet_type = {'WN': 'WN',
                'PL': 'PL',
                'SH': 'SH',
                'EX': 'EX',
                'QN': 'QN',
                'TR': 'TR',
                'SF': 'SU',
                'P3': 'P3',
                'p4': 'P4',
                'p5': 'P5',
                'p6': 'P6',
                'WPS': 'WPS',
                'WP': 'WP',
                'DB': 'DB'}

map_dtype = {'tsn': str, 'race': str, 'turnover': float}

# single leg, multi-runner and multi-race pool symbols
pool_single = ['WN', 'PL', 'SH']
pool_multi_runner = ['EX', 'QU', 'TR', 'SU']
pool_multi_race = ['DB', 'P3', 'P4', 'P5', 'P6']


class Reports:
    """
    parse
    """
    def __init__(self, verbose=False):
        self.s3 = S3FileSystem(anon=False)
        self.br = BetResult()

        self.df = DataFrame()
        self.df_electronic = DataFrame()
        self.dfraw = DataFrame()
        # for normalizing track symbols
        track_detail = os.path.join(data.__path__._path[0], 'track_detail.csv')
        self.dftrack = read_csv(track_detail)
        self.verbose = verbose

    def load(self, datelist):
        """
        load single waw pnl report by day date
        :param date: datetime like object
        """

        self.dfraw = DataFrame()
        for d in datelist:
            # load the DataFrame for this date
            date_str = d.strftime('%Y-%m-%d')
            fp = 'x8-bucket/reports/800679740 CSV Report %s.zip' % date_str

            try:
                df = read_csv(self.s3.open(fp, mode='rb'), compression='zip', dtype=map_dtype)
            except FileNotFoundError:
                warn('repots file is missing for this day %s. check ftp on EC2_Production')

            # validate the df for this date and then concat in the master df
            if df.columns[0] == 'No data for this date':
                df = DataFrame()
                warn('%s No data for this date. We did not place any bets.' % date_str)

            self.dfraw = concat([self.dfraw, df])

        # do not preform operations on df if all dates loaded have no data
        if not self.dfraw.empty:
            # raw df == df in this case
            self.df = self.dfraw.copy()

            # renaming columns using our naming conventions
            self.df.rename(columns=column_rename, inplace=True)

            # convert dates
            self.df['date'] = to_datetime(self.df['meeting_date'], format='%Y/%m/%d')

            # normalize track sym and make race_id
            map_itsp_to_x8 = self.dftrack.set_index('itsp_track_sym')['x8_track_sym'].to_dict()
            self.df['x8_track_sym'] = self.df['itsp_track_sym'].map(map_itsp_to_x8)
            self.df['race_id'] = self.df['x8_track_sym'] + '_' + self.df['date'].dt.strftime('%Y%m%d') + '_' + self.df['race_num'].astype(str)

            # normalize bet_type column
            self.df['bet_type'] = self.df['bet_type'].map(map_bet_type)
            self.df['runners'] = self.df['runners'].astype(str)

            # load bet result data from database
            self._add_computed_columns()

            # signal cols
            self.df['is_single_leg'] = self.df['bet_type'].map(lambda x: True if x in pool_single else False)
            self.df['is_multi_runner'] = self.df['bet_type'].map(lambda x: True if x in pool_multi_runner else False)
            self.df['is_multi_race'] = self.df['bet_type'].map(lambda x: True if x in pool_multi_race else False)
            self.df['is_bet_modifier'] = self.df['runners'].map(lambda x: '+' in x)

            self.br.load(datelist)
            self.df_electronic = self.df.merge(self.br.df[['serial_number', 'unique_id', 'strategy_name']],
                                               on='serial_number',
                                               how='inner')

            # validate
            self._validate()

    def _add_computed_columns(self):
        """
        add columns for computed metrics
        """
        self.df['net_return'] = self.df['tradingprofitloss'] + self.df['rebate']
        self.df['gross_roi'] = self.df['tradingprofitloss'] / self.df['bet_amount']
        self.df['net_roi'] = self.df['net_return'] / self.df['bet_amount']
        self.df['is_paid'] = (self.df['tradingprofitloss'] > 0).astype(int)
        self.df['is_unbet'] = (self.df['bet_amount'] < .0001).astype(int)

    def _validate(self):
        """check that the resulting DataFrame's values are as expected"""
        if self.df['bet_type'].isnull().any():
            raise Exception('Missing mappings in map_bet_type at top of reports.py file. Please add missing bet type.')


