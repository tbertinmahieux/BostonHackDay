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
import scipy as SP
import scipy.io
import scipy.signal
from plottools import plotall as PA
import matplotlib
import matplotlib.pyplot as P 
import numpy as N


##############################################################
# does a resampling
# done columnwise is data is a matrix
# BIG HACK in the case we don't get the exactnew size,
# we add or remove a delta in the ratio
##############################################################
def resample(data, newsize):
    return SP.signal.resample(data, newsize, axis=1)


def matfile_to_barfeats(matfile, newsize=16):
    """Convert beat-synchronous chroma features from matfile to a set
    of fixed length chroma features for every bar."""
    mat = read_matfile(matfile)
    chroma = mat['btchroma']
    bars = mat['barbts'][:,0]

    barfeats = N.empty((chroma.shape[0] * newsize, len(bars)))
    for n in xrange(len(bars)):
        try:
            end = bars[n+1]
        except IndexError:
            end = chroma.shape[1]
        barfeats[:,n] = resample(chroma[:,bars[n]:end], newsize).flatten()

    return barfeats


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
