# This is not in Production, just a skeleton for a future GBS Wrapper

from warnings import warn
import os
from pandas import concat, DataFrame, read_csv
from horse.betsim import data
from s3fs.core import S3FileSystem
import xml.etree.ElementTree as ET
import untangle


warn('gbs.py This is not in Production yet, just a skeleton for a future GBS Wrapper.')


class GlobalBettingServices:
    """
    this is for gbs data
    """
    def __init__(self, verbose=False):
        self.s3 = S3FileSystem(anon=False)

        track_detail = os.path.join(data.__path__._path[0], 'track_detail.csv')
        dftrack = read_csv(track_detail)
        self.map_track_gbs_to_x8 = dftrack.set_index('gbs_track_sym')['x8_track_sym'].to_dict()
        self.map_track_x8_to_gbs = dftrack.set_index('x8_track_sym')['gbs_track_sym'].to_dict()

        self.verbose = verbose

    def load_acceptance(self, datelist):

        raw_data = []

        for d in datelist:

            date_str = d.strftime('%Y%m%d')

            key = 'x8-gbs/acceptance/%s/' % date_str

            # list of all files in a given direcetory - in this case, all files for a single day
            s3_files = self.s3.ls(key)
            s3_files = DataFrame({'filename': s3_files})
            s3_files['id'] = s3_files['filename'].map(lambda x: x.split('.')[0].split('_')[0]).astype(int)
            s3_files['version'] = s3_files['filename'].map(lambda x: x.split('.')[0].split('_')[1])

            target_files = s3_files.groupby('id').max().index

            for file in target_files:
                with self.s3.open(file, 'rb') as f:
                    contents = f.read()
                    f.close()
                root = untangle.parse(contents.decode('utf-8'))

        self.df_raw = concat(raw_data)


class XML2DataFrame:

    def __init__(self, xml_data):
        self.root = ET.XML(xml_data)

    def parse_root(self, root):
        """Return a list of dictionaries from the text and attributes of the
        children under this XML root."""
        return [self.parse_element(child) for child in root.getchildren()]

    def parse_element(self, element, parsed=None):
        """ Collect {key:attribute} and {tag:text} from thie XML
         element and all its children into a single dictionary of strings."""
        if parsed is None:
            parsed = dict()

        for key in element.keys():
            if key not in parsed:
                parsed[key] = element.attrib.get(key)
            if element.text:
                parsed[element.tag] = element.text
            else:
                raise ValueError('duplicate attribute {0} at element {1}'.format(key, element.getroottree().getpath(element)))

        """ Apply recursion"""
        for child in list(element):
            self.parse_element(child, parsed)
        return parsed

    def process_data(self):
        """ Initiate the root XML, parse it, and return a dataframe"""
        structure_data = self.parse_root(self.root)
        return DataFrame(structure_data)
