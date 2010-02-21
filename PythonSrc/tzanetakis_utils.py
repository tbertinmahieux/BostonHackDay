"""
A set of functions to experiment with the Tzanetakis dataset
In particular, encoding.

T. Bertin-Mahieux (2010) Columbia University
tb2332@columbia.edu
"""


import os
import sys
import numpy as np
import pylab as P
import scipy as sp
import scipy
import scipy.io
# other libraries developed in that project
import feats_utils as FU
# echonest stuff
from pyechonest import track, config
try:
    config.ECHO_NEST_API_KEY = os.environ['ECHONEST_API_KEY']
except:
    config.ECHO_NEST_API_KEY = os.environ['ECHO_NEST_API_KEY']



def tzan_genres():
    """ Return a hardcoded list of Tzanetakis genres """
    return ['classical','country','disco','hiphop','jazz','rock','blues','reggae','pop','metal']


def get_all_tzan_files(mainDir):
    """
    Return a list of filenames for all tzanetakis files.
    INPUT:
       - mainDir     Top directory of the Tzanetakis dataset
    OUTPUT:
       - fileList    List of all filenames
    """
    raise NotImplementedError


def get_en_feats(filename):
    """
    Given a filename from the Tzanetakis dataset, go get the Echo Nest
    features for that file. Features are basically per segment,
    but we have the start time of beats and bars
    INPUT:
       - filename    Tzanetakis file full filename
    OUTPUT:
       - pitches           numpy array (12 x nSegs)
       - segs start time   array
       - beats start time  array
       - bars start time   array
    """
    raise NotImplementedError
    entrack = track.upload(filename)
    segs = entrack.segments
    # pitches
    pitches = [s['pitches'] for s in entrack.segments]
    pitches = np.array(pitches).T
    # seg start
    seg_start = [s['start'] for s in entrack.segments]
    # beats start
    beat_start = [b['start'] for b in entrack.beats]
    # bars start
    bar_start = [b['start'] for b in entrack.bars]
    # return
    return pitches, seg_start, beat_start, bar_start



def die_with_usage():
    """ Help menu. """
    print 'set of functions for handling Tzanetakis dataset'
    sys.exit(0)


if __name__ == '__main__':

    if len(sys.argv) < 1:
        die_with_usage()


