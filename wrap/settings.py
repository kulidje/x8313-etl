# wrapper for loading settings from a hjson file validating fields

import os
import hjson
from horse import betsim

def load_settings():
    """
    load settings from horse/settings.hjson, validate required fields,
    """
    # horse directory and the settings.hjson file 
    betsim_path = os.path.dirname(betsim.__file__)
    horse_path = os.path.dirname(betsim_path) # parent dir
    filename = os.path.join(horse_path, 'settings.hjson')

    # error if the settings.hjson file doesn't exist
    if not os.path.exists(filename):
        raise Exception('horse.wrap.load_settings() ERROR: The settings.hjson file does not exist here: %s. Copy sample_settings.hjson to settings.hjson and change the settings to match your environment.' % filename)

    # load the settings file
    settings = hjson.loads(open(filename).read())

    # TODO: error if key variables are missing
    # for key in ['output_path']:
    #     if key not in settings:
    #         raise Exception('horse.wrap.load_settings() ERROR: The settings.hjson file does not contain the required key: %s' % key)

    # add the root directory for x8313/horse/ project files
    settings['horse_path'] = horse_path
    return settings
