#!/usr/bin/env python
##############################################################
#
# Project by
#               Ron Weis - ronw@ee.columbia.edu
# Thierry Bertin-Mahieux - tb2332@columbia.edu
#
# Find similarity using LSH on some EchoNest features
#
##############################################################


import string
import sys
import os
import os.path
import scipy as SP
import scipy.io
import scipy.signal
from plottools import plotall as PA
import matplotlib
import matplotlib.pyplot as P 
import numpy as N
import glob


##############################################################
# does a resampling
# done columnwise is data is a matrix
# BIG HACK in the case we don't get the exactnew size,
# we add or remove a delta in the ratio
##############################################################
def resample(data, newsize):
    return SP.signal.resample(data, newsize, axis=1)

def matfile_to_enid(matfile):
    return os.path.split(matfile)[-1].replace('.mat', '').upper()

def matfile_to_barfeats(matfile, newsize=16, keyinvariant=False, downbeatinvariant=False):
    """Convert beat-synchronous chroma features from matfile to a set
    of fixed length chroma features for every bar."""
    mat = read_matfile(matfile)
    chroma = mat['btchroma']
    bars = mat['barbts'][:,0]

    if keyinvariant and downbeatinvariant:
        invariance_fun = lambda bar: N.abs(N.fft.rfft2(bar))
    elif keyinvariant:
        invariance_fun = lambda bar: N.abs(N.fft.rfft(bar, axis=0))
    elif downbeatinvariant:
        invariance_fun = lambda bar: N.abs(N.fft.rfft(bar, axis=1))
    else:
        invariance_fun = lambda bar: bar

    barfeats = []
    for n in xrange(len(bars)):
        try:
            end = bars[n+1]
        except IndexError:
            end = chroma.shape[1]
        feat = invariance_fun(resample(chroma[:,bars[n]:end], newsize))
        barfeats.append(feat.flatten())

    enid = matfile_to_enid(matfile)
    barlabels = ['%s:%d' % (enid, x) for x in bars]

    return N.asarray(barfeats).T, barlabels


def matfiles_to_feats_to_txt(matfiles,featfile,descfile):
    """Take all matlab files, get barfeats, output the results
    into 2 text files, one for features and one for description.
    There's one bar per line, e.g. both file should have as many
    lines. The feat file format fits with E2LSH input format."""

    fidFeat = open(featfile,'w')
    fidDesc = open(descfile,'w')
    # iterate over matfiles
    for matfile in matfiles :
        barfeats, barlabels = matfile_to_barfeats(matfile)
        # iterate over beats
        for n in range(barfeats.shape[1]) :
            # write features
            N.tofile(fidFeat,sep=' ')
            fidFeat.write('\n')
            # write descriptions
            fidDesc.write(barlables[n])
            fidDesc.write('\n')
    # close files, and done
    fidFeat.close()
    fidDesc.close()


def get_all_matfiles(basedir) :
    """From a root directory, go through all subdirectories
    and find all matlab files. Return them in a list."""
    allfiles = []
    for root, dirs, files in os.walk(basedir):
        matfiles = glob.glob(os.path.join(root,'*.mat'))
        for f in matfiles :
            allfiles.append( os.path.abspath(f) )
    return allfiles


##############################################################
# imshow
# wrapper around matplotlib.pyplot with proper params
##############################################################
def imshow(data) :
    PA([data],
       aspect='auto', interpolation='nearest')
    P.show()


##############################################################
# read a matlabfile
# uses scipy utility function
##############################################################
def read_matfile(filename):
    return SP.io.loadmat(filename)



##############################################################
# help menu
##############################################################
def die_with_usage():
    print 'feats_utils.py'
    print 'a set of functions to get features, plot them,'
    print 'and transform them'
    print 'goal: similarity through LSH'
    sys.exit(0)




##############################################################
# MAIN
##############################################################
if __name__ == '__main__' :

    if len(sys.argv) < 2 :
        die_with_usage()


    print('dummy tests!')

    # load and plot
    data = read_matfile('../tr0002q11c3fa8332d.mat')
    chromas = data['btchroma']
    beats = data['barbts']
    P.figure()
    imshow(chromas)

    # resample and show
    chromas2 = resample(chromas,50)
    P.figure()
    imshow(chromas2)
