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
import get_echo_nest_metadata as GENM
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
       - duration          in seconds
    """
    entrack = track.upload(filename)
    segs = entrack.segments
    # pitches
    pitches = [s['pitches'] for s in entrack.segments]
    pitches = np.array(pitches).T
    # seg start
    seg_start = np.array([s['start'] for s in entrack.segments])
    # beats start
    beat_start = np.array([b['start'] for b in entrack.beats])
    # bars start
    bar_start = np.array([b['start'] for b in entrack.bars])
    # return
    return pitches, seg_start, beat_start, bar_start, entrack.duration


def filename_to_beatfeat_mat(filename,savefile=''):
    """
    Take a Tzanetakis song (or any song)
    Get the echonest data
    Save the echonest data into a matfile
    Matfile is same path as the filename, extension changes
    """
    # skip if output exists
    if savefile == '':
        savefile = filename+'.mat'
    if os.path.exists(savefile):
        print 'file ' + savefile + ' exists, we skip'
        return
    # get EN features
    pitches, segstart, btstart, barstart, dur = get_en_feats(filename)
    # warp it!
    # see get_echo_nest_metadata.get_beat_synchronous_chromagram()
    segchroma = pitches.T
    warpmat = GENM.get_time_warp_matrix(segstart, btstart, dur)
    btchroma = np.dot(warpmat, segchroma)
    # Renormalize.
    btchroma = (btchroma.T / btchroma.max(axis=1)).T
    # get the start time of bars
    # result for track: 'TR0002Q11C3FA8332D'
    #    barstart.shape = (98,)
    barbts = np.zeros(barstart.shape)
    # get the first (only?) beat the starts at the same time as the bar
    for n, x in enumerate(barstart):
        barbts[n] = np.nonzero((btstart - x) == 0)[0][0]
    # save to matlab file, see:
    # get_echo_nest_metadata.convert_matfile_to_beat_synchronous_chromagram()

    # write file, need to create directories? dangerous but efficient...
    directory = os.path.dirname(savefile)
    if not os.path.exists(directory):
        os.makedirs(directory)
    sp.io.savemat(savefile, dict(btchroma=btchroma.T,
                                 barbts=barbts,       segstart=segstart,
                                 btstart=btstart,     barstart=barstart,
                                 duration=dur))


def die_with_usage():
    """ Help menu. """
    print 'set of functions for handling Tzanetakis dataset'
    sys.exit(0)


if __name__ == '__main__':

    if len(sys.argv) < 1:
        die_with_usage()


