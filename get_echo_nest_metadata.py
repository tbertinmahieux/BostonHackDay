import glob
import os

import scipy as sp
import scipy.io
import numpy as np

from pyechonest import config
from pyechonest import track
config.ECHO_NEST_API_KEY = os.environ['ECHONEST_API_KEY']

print 'ECHONEST_API_KEY' in os.environ

#def convert_matfiles_to_beat_synchronous_chromagrams(matfilepattern, outputdir):
#    matfiles = glob.glob(matfilepattern)
    




def get_beat_synchronous_chromagram(matfile):
    analysis = sp.io.loadmat(matfile)['M'][0][0]

    enid = os.path.split(matfile)[-1].replace('.mat', '').upper()
    entrack = track.Track(enid)
    
    # Echo Nest "segment" synchronous chroma
    segchroma = analysis.pitches.T

    segstart = analysis.start[0]
    btstart = np.array([x['start'] for x in entrack.beats])
    
    # Move segment chromagram onto a regular grid
    warpmat = get_time_warp_matrix(segstart, btstart, entrack.duration)
    btchroma = np.dot(warpmat, segchroma)

    # Renormalize.
    btchroma = (btchroma.T / btchroma.max(axis=1)).T

    return btchroma

    
def get_time_warp_matrix(segstart, btstart, duration):
    seglen = np.concatenate((segstart[1:], [duration])) - segstart
    btlen = np.concatenate((btstart[1:], [duration])) - btstart

    warpmat = np.zeros((len(segstart), len(btstart)))
    for n in xrange(len(btstart)):
        start = btstart[n]
        end = start + btlen[n]
        start_idx = np.nonzero((segstart - start) >= 0)[0][0] - 1
        try:
            end_idx = np.nonzero((segstart - end) >= 0)[0][0]
        except IndexError:
            end_idx = start_idx
        warpmat[start_idx:end_idx, n] = 1
        warpmat[start_idx, n] = ((start - segstart[start_idx])
                                 / seglen[start_idx])
        warpmat[end_idx, n] = ((segstart[end_idx] - end)
                               / seglen[end_idx])
        warpmat[:,n] /= np.sum(warpmat[:,n])

    return warpmat.T
    
