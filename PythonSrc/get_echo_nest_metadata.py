import glob
import logging
import os
import sys
import time

import scipy as sp
import scipy.io
import numpy as np

from pyechonest import config
from pyechonest import track
try:
    config.ECHO_NEST_API_KEY = os.environ['ECHO_NEST_API_KEY']
except:
    config.ECHO_NEST_API_KEY = os.environ['ECHONEST_API_KEY']
config.ANALYSIS_VERSION = 1


try:
    import ronwtools
    logfile = 'logging_' + time.ctime().replace('  ',' ').replace(' ','_') .replace(':','_')+ '.txt'
    print 'logging to file: ' + logfile
    ronwtools.initialize_logging(filename=logfile)
except:
    pass

try:
    import multiprocessing
    use_multiproc = True
except:
    use_multiproc = False


def convert_matfile_to_beat_synchronous_chromagram(matfile, savedir):
    logging.info('Processing %s', matfile)

    path, filename = os.path.split(matfile)
    currdir = os.path.join(savedir, path)
    logging.debug('path=%s.',path)
    logging.debug('savedir=%s.',savedir)
    logging.debug('currdir=%s.',currdir)
    if not os.path.exists(currdir):
        os.makedirs(currdir)
    savefile = os.path.join(currdir, filename)
    if os.path.exists(savefile):
        logging.info('Skipping file %s because %s already exists.',
                     matfile, savefile)
        return

    try:
        btchroma, barbts, segstart, btstart, barstart, duration \
                  = get_beat_synchronous_chromagram(matfile)

        logging.info('Saving results to %s', savefile)
        sp.io.savemat(savefile, dict(btchroma=btchroma.T,
                                     barbts=barbts,       segstart=segstart,
                                     btstart=btstart,     barstart=barstart,
                                     duration=duration))
    except:
        logging.error('PROBLEM with file %s, skipping.',matfile)
        return
        

def get_beat_synchronous_chromagram(matfile):
    """
    - Takes the analysis from Dan's matlab files
    - Transform the analysis per segment in analysis per beat
    - Add the info: which beats are in which bar
    - add some timing info: length of the song, startng time of beats
    and starting time of bars
    """

    # analysis from Dan's matlab files
    if sys.version_info[1]==5:
        analysis =  sp.io.loadmat(matfile)['M']
    else:
        analysis =  sp.io.loadmat(matfile)['M'][0][0]

    # get EchoNest id, get full metadata including beats and bars
    enid = os.path.split(matfile)[-1].replace('.mat', '').upper()
    logging.info('Calling Echo Nest on: %s', enid)
    entrack = track.Track(enid)
    
    # Echo Nest "segment" synchronous chroma
    # 12 values per line (one segment per line)
    # result for track: 'TR0002Q11C3FA8332D'
    #    segchroma.shape = (708, 12)
    segchroma = analysis.pitches.T

    # get the series of starts for segments, beats, and bars
    # result for track: 'TR0002Q11C3FA8332D'
    #    segstart.shape = (708,)
    #    btstart.shape = (304,)
    #    barstart.shape = (98,)
    if sys.version_info[1] == 5:
        segstart = analysis.start
    else:
        segstart = analysis.start[0]
    btstart = np.array([x['start'] for x in entrack.beats])
    barstart = np.array([x['start'] for x in entrack.bars])

    # CHROMA PER BEAT
    # Move segment chromagram onto a regular grid
    # result for track: 'TR0002Q11C3FA8332D'
    #    warpmat.shape = (304, 708)
    #    btchroma.shape = (304, 12)
    warpmat = get_time_warp_matrix(segstart, btstart, entrack.duration)
    btchroma = np.dot(warpmat, segchroma)

    # Renormalize.
    btchroma = (btchroma.T / btchroma.max(axis=1)).T

    # CHROMA PER BAR
    # similar to chroma per beat
    #warpmat = get_time_warp_matrix(segstart, barstart, entrack.duration)
    #barchroma = np.dot(warpmat, segchroma)
    #barchroma = (barchroma.T / barchroma.max(axis=1)).T

    # get the start time of bars
    # result for track: 'TR0002Q11C3FA8332D'
    #    barstart.shape = (98,)
    barstart = np.array([x['start'] for x in entrack.bars])
    barbts = np.zeros(barstart.shape)
    # get the first (only?) beat the starts at the same time as the bar
    for n, x in enumerate(barstart):
        barbts[n] = np.nonzero((btstart - x) == 0)[0][0]

    return btchroma, barbts, segstart, btstart, barstart, entrack.duration

    
def get_time_warp_matrix(segstart, btstart, duration):
    """
    Returns a matrix (#beats,#segs)
    #segs should be larger than #beats, i.e. many events or segs
    happen in one beat.
    """

    # length of beats and segments in seconds
    # result for track: 'TR0002Q11C3FA8332D'
    #    seglen.shape = (708,)
    #    btlen.shape = (304,)
    #    duration = 238.91546    meaning approx. 3min59s
    seglen = np.concatenate((segstart[1:], [duration])) - segstart
    btlen = np.concatenate((btstart[1:], [duration])) - btstart

    warpmat = np.zeros((len(segstart), len(btstart)))
    # iterate over beats (columns of warpmat)
    for n in xrange(len(btstart)):
        # beat start time and end time in seconds
        start = btstart[n]
        end = start + btlen[n]
        # np.nonzero returns index of nonzero elems
        # find first segment that starts after beat starts - 1
        try:
            start_idx = np.nonzero((segstart - start) >= 0)[0][0] - 1
        except IndexError:
            # no segment start after that beats, can happen close
            # to the end, simply ignore, maybe even break?
            continue
        # find first segment that starts after beat ends
        try:
            end_idx = np.nonzero((segstart - end) >= 0)[0][0]
        except IndexError:
            end_idx = start_idx
        # fill col of warpmat with 1 for the elem in between
        # (including start_idx, excluding end_idx)
        warpmat[start_idx:end_idx, n] = 1
        
        # FOLLOWING CODE SEEMS WRONG... SEE NEW CODE BELOW
        #warpmat[start_idx, n] = ((start - segstart[start_idx])
        #                         / seglen[start_idx])
        #warpmat[end_idx, n] = ((segstart[end_idx] - end)
        #                       / seglen[end_idx])

        # if the beat started after the segment, keep the proportion
        # of the segment that is inside the beat
        warpmat[start_idx, n] = 1. - ((start - segstart[start_idx])
                                 / seglen[start_idx])
        # if the segment ended after the beat ended, keep the proportion
        # of the segment that is inside the beat
        if end_idx - 1 > start_idx:
            warpmat[end_idx-1,n] = ((end - segstart[end_idx-1])
                                    / seglen[end_idx-1])
        # normalize so the 'energy' for one beat is one
        warpmat[:,n] /= np.sum(warpmat[:,n])

    # return the transpose, meaning (#beats , #segs)
    return warpmat.T


def pickleable_wrapper(args):
    #time.sleep(np.random.rand(1)[0])
    return convert_matfile_to_beat_synchronous_chromagram(*args)


def get_all_matfiles(basedir) :
    """From a root directory, go through all subdirectories
    and find all matlab files. Return them in a list.
    This is copied from feats_utils... too bad!!!!! damn imports!"""
    allfiles = []
    for root, dirs, files in os.walk(basedir):
        matfiles = glob.glob(os.path.join(root,'*.mat'))
        for f in matfiles :
            allfiles.append( os.path.abspath(f) )
    return allfiles


def main(matfilepath, savedir, nprocesses=100):
    """
    Compute the beat synchronous chromagrams for each song in
    our cowbell dataset:
    - take a matlab file from Dan
    - get the full metadata from EchoNest
    - compute beat synchronous chromagram
    - save to a matlab file
    """
    logging.info('Reading .mat files from %s', matfilepath)
    logging.info('Saving files to %s', savedir)
    logging.info('Using %d processes', nprocesses)
    matfiles = glob.glob(os.path.join(matfilepath, '*/*/*.mat'))
    #matfiles = get_all_matfiles(matfilepath)

    args = [(x, savedir) for x in np.random.permutation(matfiles)]
    if nprocesses > 1:
        if not use_multiproc:
            "multiprocessing package not available on this machine"
            sys.exit(0)
        pool = multiprocessing.Pool(processes=nprocesses)
        pool.map(pickleable_wrapper, args)
    else:
        for argset in args:
            pickleable_wrapper(argset)
             

def die_with_usage():
    """
    help menu
    """
    print 'usage: place yourself in the main matlab directory,'
    print 'the one containing the matlab files from Dan'
    print 'launch:'
    print '  python get_echo_nest_metadat.py . <savedir> <#CPU>'
    sys.exit(0)

if __name__ == '__main__':

    if len(sys.argv) < 1:
        die_with_usage()
    
    args = sys.argv[1:] 
    (matfilepath, savedir, nprocesses) = args[:3]
    nprocesses = int(nprocesses)
    main(matfilepath, savedir, nprocesses)
    
