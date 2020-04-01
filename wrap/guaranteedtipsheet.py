# wrapper for Guaranteed Tip Sheet (GTS) files

import os
from pandas import concat, DataFrame, read_csv, melt, to_datetime
from horse.betsim import data
from s3fs.core import S3FileSystem

# list of column names
column_names = ['gts_track_name', 'date', 'race_num', 'win', 'place', 'show', 'wildcard', 'a1', 'a2']

# dictionary of column types - read these fields as strings b/c the place could be 1A instead of just a number
dtypestr = {'race_num': str, 'win': str, 'place': str, 'show': str, 'wildcard': str, 'a1': str, 'a2': str}


class GuaranteedTipSheet:
    """ GuaranteedTipSheet wraps the results files from guaranteedtipsheet.com
        Attributes:
        df: DataFrame of processed GTS data (use this data for calculations)
        dfraw: DataFrame of raw GTS data
    """
    def __init__(self):
        #self.datapath = datapath
        self.s3 = S3FileSystem(anon=False)
        self.df = DataFrame()
        self.dfraw = DataFrame()
        track_detail = os.path.join(data.__path__._path[0], 'track_detail.csv')
        self.dftrack = read_csv(track_detail)
        # TODO 'Los Alamitos QH' not in track_detail - see gts_picks_20180309.csv

    def load(self, datelist):
        """ load all GuaranteedTipSheet (GTS) files for the specified dates
            :param datelist: list of dates
        """
        # TODO: validate input is list of dates

        # load each date
        self.dfraw = DataFrame()
        for d in datelist:

            # load the DataFrame for this date
            date_str = d.strftime('%Y%m%d')
            fp = 'x8-bucket/gts/gts_picks_%s.csv' % date_str
            try:
                df = read_csv(self.s3.open(fp, mode='rb'), header=None, names=column_names, dtype=dtypestr)
            except FileNotFoundError:
                raise FileNotFoundError('File not found. We do not have GTS data for %s on S3' % d.date())

            # validate the df for this date and then concat in the master df
            self.dfraw = concat([self.dfraw, df])

        # raw df == df in this case
        self.df = self.dfraw.copy()

        # race_id and x8_track_sym
        map_gts_track_name_to_x8_symbol = self.dftrack.set_index('gts_track_name')['x8_track_sym'].to_dict()
        self.df['x8_track_sym'] = self.df['gts_track_name'].map(map_gts_track_name_to_x8_symbol)
        self.df['race_id'] = self.df['x8_track_sym'] + '_' + self.df['date'].astype(str) + '_' + self.df['race_num'].astype(str)

        self.df = melt(self.df, id_vars=['race_id', 'x8_track_sym', 'race_num', 'date'], value_vars=['win', 'place', 'show', 'wildcard', 'a1', 'a2'],
                var_name='factor_gts', value_name='program_number').set_index(
            ['race_id', 'program_number']).reset_index()

        self.df['runner_id'] = self.df['race_id'] + '_' + self.df['program_number']
        self.df['date'] = to_datetime(self.df['date'], format='%Y%m%d')

        # gts was coming with duplicate rows 20180807 and 20180809
        if self.df['runner_id'].nunique() != len(self.df):
            print('gts.load(): GTS data has duplicated rows.. dropping duplicates..')
            self.df.drop_duplicates(subset='runner_id', inplace=True)

        # validate master df
        self._validate()

    def _validate(self):
        """check that the resulting DataFrame values are as expected"""
        if self.df['runner_id'].nunique() != len(self.df):
            Exception('1 runner per row')