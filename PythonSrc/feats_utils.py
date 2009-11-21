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
from plottools import plotall as PA
import matplotlib
import matplotlib.pyplot as P 
import numpy as N
try:
    import scikits.samplerate as samplerate
except:
    import pysamplerate as samplerate


##############################################################
# does a resampling
# done columnwise is data is a matrix
# BIG HACK in the case we don't get the exactnew size,
# we add or remove a delta in the ratio
##############################################################
def resample(data, newsize):
    #rs = samplerate.resample
    rs = lambda x,y: samplerate.resample(x, y, type='sinc_fastest')

    delta = .0001
    res = N.zeros([N.shape(data)[0],newsize])
    for l in range(N.shape(data)[0]):
        newdata  = rs(data[l,:], newsize * 1. / N.shape(data)[1]);
        if N.asarray(newdata).shape[0] > newsize :
            newdata  = rs(data[l,:], newsize * 1. / N.shape(data)[1] - delta);
        if N.asarray(newdata).shape[0] > newsize :
            newdata  = rs(data[l,:], newsize * 1. / N.shape(data)[1] + delta);
        if N.asarray(newdata).shape[0] != newsize :
            print 'resample, bad new size!!!'
            continue
        res[l,:] = newdata.T
    return res


def matfile_to_barfeats(matfile, newsize=16):
    """"""
    mat = read_matfile(matfile)
    chroma = mat['btchroma']
    bars = mat['barbts'][:,0]

    barfeats = N.empty((chroma.shape[0] * newsize, len(bars)))
    for n in xrange(len(bars)):
        try:
            end = bars[n+1]
        except IndexError:
            end = len(chroma)
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
