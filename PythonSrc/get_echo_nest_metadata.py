import glob
import logging
import multiprocessing
import os
import sys
import time

print 'ECHONEST_API_KEY' in os.environ

import scipy as sp
import scipy.io
import numpy as np

from pyechonest import config
from pyechonest import track
config.ECHO_NEST_API_KEY = os.environ['ECHONEST_API_KEY']
config.ANALYSIS_VERSION = 1

try:
    import ronwtools
    ronwtools.initialize_logging()
except:
    pass

def convert_matfile_to_beat_synchronous_chromagram(matfile, savedir):
    logging.info('Processing %s', matfile)

    path, filename = os.path.split(matfile)
    currdir = os.path.join(savedir, path)
    if not os.path.exists(currdir):
        os.makedirs(currdir)
    savefile = os.path.join(currdir, filename)
    if os.path.exists(savefile):
        logging.info('Skipping file %s because %s already exists.',
                     matfile, savefile)
        return
        
    btchroma, barbts = get_beat_synchronous_chromagram(matfile)

    logging.info('Saving results to %s', savefile)
    sp.io.savemat(savefile, {'btchroma': btchroma.T, 'barbts': barbts})


def get_beat_synchronous_chromagram(matfile):
    analysis = sp.io.loadmat(matfile)['M'][0][0]

    enid = os.path.split(matfile)[-1].replace('.mat', '').upper()
    logging.info('Calling Echo Nest on: %s', enid)
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

    barstart = np.array([x['start'] for x in entrack.bars])
    barbts = np.zeros(barstart.shape)
    for n, x in enumerate(barstart):
        barbts[n] = np.nonzero((btstart - x) == 0)[0][0]

    return btchroma, barbts

    
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


def pickleable_wrapper(args):
    #time.sleep(np.random.rand(1)[0])
    return convert_matfile_to_beat_synchronous_chromagram(*args)

def main(matfilepath, savedir, nprocesses=100):
    logging.info('Reading .mat files from %s', matfilepath)
    logging.info('Saving files to %s', savedir)
    logging.info('Using %d processes', nprocesses)
    matfiles = glob.glob(os.path.join(matfilepath, '*/*/*.mat'))

    args = [(x, savedir) for x in np.random.permutation(matfiles)]
    if nprocesses > 1:
        pool = multiprocessing.Pool(processes=nprocesses)
        pool.map(pickleable_wrapper, args)
    else:
        for argset in args:
            pickleable_wrapper(argset)
             


if __name__ == '__main__':
    args = sys.argv[1:] 
    (matfilepath, savedir, nprocesses) = args[:3]
    nprocesses = int(nprocesses)
    main(matfilepath, savedir, nprocesses)
    
