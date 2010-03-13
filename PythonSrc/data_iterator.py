#! /usr/bin/env
"""
This class describe an iterator over a set of matlab files that contains
the data

T. Bertin-Mahieux (2010) Columbia University
tb2332@columbia.edu
"""

import os
import sys
import glob
import numpy as np
import scipy as sp
import scipy.io
#import pylab as P

import feats_utils as FU


class DataIterator:

    def __init__(self):
        """ Constructor """
        #print 'creating an instance of DataIterator'
        self.idx = 0
        self._stopAfterOnePass = 1
        self.matfiles = []
        self.featsize = 4
        self.usebars = 1
        self.fidx = 0 # which file are we at
        self.pidx = 0 # which pattern are we at
        self.currfeats = []  # current features
        self.nPatternSeen = 0 # for stats
        self.passSize = -1 # number of element in one pass, unknown to start with
        self.barbts = [] # index of first beats of bars
        self.offset = 0 # percentage of a bar to offset
        self.partialbar = 1

    def setMatfiles(self,mfiles):
        """
        Set the list of matfiles of interest.
        Order of the files is ALWAYS shuffled.
        """
        self.matfiles = mfiles
        np.random.shuffle(self.matfiles)

    def setOffset(self,val):
        """
        Set the offset, value between 0 and 1
        """
        assert(val>=0)
        assert(val<1)
        self.offset = val;

    def useBars(self,n):
        """
        If superior to 0, we return a pattern per <n> bar.
        The length of the pattern
        is therefore variable. Default: true.
        """
        if n > 1:
            assert(self.partialbar == 1)
        self.usebars = n

    def getBars(self):
        """ Returns the number of bars we use, or 0 """
        return self.usebars

    def usePartialBar(self,val):
        """
        Returns a percentage of the bar, does nothing if 1. Value between
        0 and 1. self.useBars() must be set to 1.
        """
        assert(val>0)
        assert(val<=1)
        assert(self.usebars == 1)
        self.partialbar = val

    def setFeatsize(self,n):
        """
        Length of a pattern, 4? 8? 12? default: 4
        Useless if we use bars
        """
        self.featsize = n

    def getFeatsize(self):
        """ Returns the feature size, or -1 if we use bars """
        if self.usebars == 1 :
            return -1
        return self.featsize

    def resetIterations(self):
        """ Reshuffle files and set the counters to 0 """
        np.random.shuffle(self.matfiles)
        self.fidx = 0
        self.pidx = 0
        self.nPatternSeen = 0

    def stopAfterOnePass(self,boolean):
        """ Whether we stop after one pass, or never stop, default: True """
        if boolean:
            self._stopAfterOnePass = 1
        else:
            self._stopAfterOnepass = 0

    def passSize(self):
        """
        Returns the number of element in one pass, or -1 if
        unknown (it's unknown until the end of one pass)
        """
        return self.passSize

    def stats(self):
        """ Return a string to print """
        s = 'DataIterator: featsize=' + str(self.getFeatsize()) + ', '
        s = s + 'usebars=' + str(self.getBars()) + ', '
        s = s + str(len(self.matfiles))
        s = s + ' files, saw ' + str(self.nPatternSeen) + ' patterns'
        return s
        
    def next(self):
        """
        Returns the next pattern
        """
        # SAME FILE
        # next pidx
        if self.pidx > 0 :
            if self.usebars == 0 :
                if self.pidx + self.featsize > self.currfeats.shape[1]:
                    self.fidx = self.fidx + 1
                    self.pidx = 0
                    self.barbts = []
                else :
                    x1 = self.pidx
                    x2 = x1 + self.featsize
                    self.pidx = x2
                    self.nPatternSeen = self.nPatternSeen + 1
                    return self.currfeats[:,x1:x2]
            else :
                if self.pidx >= self.currfeats.shape[1]:
                    self.fidx = self.fidx + 1
                    self.pidx = 0
                    self.barbts = []
                else : # we use bars
                    x1 = self.pidx
                    idx1 = np.where(self.barbts == self.pidx)[0][0]
                    if self.offset > 0 and idx1 + self.usebars + 1 >= len(self.barbts):
                        # we're at the end, we have an offset, cant make it
                        self.fidx = self.fidx + 1
                        self.pidx = 0
                        self.barbts = []
                    elif idx1 + self.usebars < len(self.barbts):
                        # simple case
                        x2 = self.barbts[idx1+self.usebars]
                        self.pidx = x2
                        curroffset = 0
                        if self.offset > 0:
                            curroffset = np.round((x2-x1)*self.offset/self.usebars)
                        self.nPatternSeen = self.nPatternSeen + 1
                        if self.partialbar < 1:
                            x2 = x1 + np.round((x2-x1)*self.partialbar)
                            if x2 == x1:
                                x2 = x1+1
                        return self.currfeats[:,x1+curroffset:x2+curroffset]
                    else :
                        # end of the song case, no offset
                        assert(self.offset == 0)
                        x2 = self.currfeats.shape[1]
                        self.pidx = x2
                        self.nPatternSeen = self.nPatternSeen + 1
                        if self.partialbar < 1:
                            x2 = x1 + np.round((x2-x1)*self.partialbar)
                            if x2 == x1:
                                x2 = x1+1
                        return self.currfeats[:,x1:x2]
        # NEW FILE
        # in case song does not contain features long enough
        while True:
            # one pass done?
            if self.fidx >= len(self.matfiles):
                if self.passSize < 0:
                    self.passSize = self.nPatternSeen
                if self._stopAfterOnePass == 1:
                    self.fidx = 0
                    self.pidx = 0
                    raise StopIteration
                else:
                    self.fidx = 0
                    continue
            # load features
            if sys.version_info[1] == 5:
                mat = sp.io.loadmat(self.matfiles[self.fidx])
            else:
                mat = sp.io.loadmat(self.matfiles[self.fidx], struct_as_record=True)
            self.currfeats = mat['btchroma']
            self.barbts = mat['barbts']
            if sys.version_info[1] != 5:
                self.barbts = self.barbts[0]
            # enough features?
            if type(self.barbts) == type(0.0): # weird problem sometimes
                self.fidx = self.fidx + 1
                continue
            if self.currfeats.size == 0 or len(self.currfeats.shape) < 2:
                self.fidx = self.fidx + 1
                continue
            if self.usebars == 0 and self.currfeats.shape[1] < self.featsize :
                self.fidx = self.fidx + 1
                continue
            if self.usebars >= 1 and np.array(self.barbts).size < self.usebars :
                self.fidx = self.fidx + 1
                continue
            if self.usebars >= 1 and self.offset > 0 and np.array(self.barbts).size + 1 < self.usebars :
                self.fidx = self.fidx + 1
                continue
            # get first feature
            if self.usebars == 0:
                self.pidx = self.featsize
                self.nPatternSeen = self.nPatternSeen + 1
                return self.currfeats[:,0:self.pidx]
            else :
                if np.array(self.barbts).size == self.usebars and self.offset==0:
                    self.pidx = self.currfeats.shape[1]
                    self.nPatternSeen = self.nPatternSeen + 1
                    return self.currfeats[:,0:self.pidx]
                elif np.array(self.barbts).size == self.usebars:
                    # just enough bar, but we need offset... too bad
                    self.fidx = self.fidx + 1
                    continue
                else:
                    startidx = 0
                    if self.offset > 0 :
                        startidx = np.round(self.barbts[self.usebars] * self.offset / self.usebars)
                    self.pidx = self.barbts[self.usebars]
                    self.nPatternSeen = self.nPatternSeen + 1
                    x2 = self.pidx+startidx
                    if self.partialbar < 1:
                        x2 = startidx + np.round((x2-startidx)*self.partialbar)
                        if x2 == startidx:
                            x2 = startidx+1
                    return self.currfeats[:,startidx:x2]
            

    def __iter__(self):
        """ Returns itself, part of the python iterator interface """
        return self
        


# debugging
if __name__ == '__main__' :

    print 'debugging iterator'
    dt = DataIterator()

    # find all files from current dir
    allfiles = FU.get_all_matfiles('.')

    # SET THE DATA ITERATOR
    dt.setMatfiles(allfiles) # set the matlab files
    dt.useBars(2)            # work based on bars, 2 bars at a time
    featsize = 4
    dt.setFeatsize(featsize) # useless because we work on bars
    dt.stopAfterOnePass(True)# stop after one pass, otherwise never stops
    
    print dt.stats()         # print basic stats

    cnt = 0
    maxIter = 500000
    #bestPattern = np.random.rand(12,featsize)
    #maxP = -1000
    #minP = 1000
    for p in dt :
        #bestPattern = bestPattern + .001 * (p - bestPattern)
        #lastP = p
        #maxP = max(maxP,p.max())
        #minP = min(minP,p.min())
        print p.shape
        cnt = cnt + 1
        if cnt > maxIter:
            break

    print 'counter = ' + str(cnt)
    print dt.stats()

    #print 'maxP = ' + str(maxP)
    #print 'minP = ' + str(minP)

    #P.imshow(bestPattern,interpolation='nearest')
    #P.figure()
    #P.imshow(p,interpolation='nearest')
    #P.show()

    
