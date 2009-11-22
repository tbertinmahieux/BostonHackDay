#!/usr/bin/env python

"""
Library to resynthetize chroma beats.
Reimplementation of the MATLAB code by Dan Ellis.

Project started by:
               Ron Weis - ronw@ee.columbia.edu
 Thierry Bertin-Mahieux - tb2332@columbia.edu

"""


import os
import sys
import numpy as N


def chromsynth(F,bp=.5,sr=22050,dowt=1) :
    """Takes a chroma frame F and output a signal.
    Code reimplemented from the MATLAB code of D. Ellis.
    INPUT:
           F - chromas, one beat per 
          bp - period of one beat per second (default=0.5)
          sr - signal rate (default=22050)
        dowt - put weight on sinusoide (default=1, true)
    OUTPUT:
      signal - signal to be saved as a .wav
    """

    nchr,nbts = F.shape;

    # get actual times
    bts = bp*N.asarray(range(0,nbts+1))

    if len(bts) < nbts+1:  # +1 to have end time of final note
        medbt = N.median(N.diff(bts))
        bts = N.append(bts,bts[-1]+medbt*N.asarray(range(1,(nbts+1-len(bts)+1))))
    # crossfade overlap time
    dt = 0.01;
    dtsamp = N.round(dt*sr);

    # Generate 12 basic shepard tones
    dur = N.max(N.diff(bts)) # max duration
    dursamp = N.round(dur*sr)
    nchr = 12
    octs = 10
    basefrq = 27.5*(pow(2,3/12))  # A1+3 semis = C2;

    tones = N.zeros([nchr, dursamp + 2*dtsamp + 1])
    tt = N.asarray(range(tones.shape[1]-1+1))/sr;

    # what bin is the center freq?
    f_ctr = 440
    f_sd = 0.5
    f_bins = basefrq*N.power(2,N.asarray(range(nchr*octs -1 + 1))/nchr)
    f_dist = N.log(f_bins/f_ctr)/N.log(2)/f_sd

    # Gaussian weighting centered of f_ctr, with f_sd
    if dowt > 0 :
        f_wts = N.exp(-0.5*N.power(f_dist,2))
    else : # flat weights
        f_wts = N.ones([1,len(f_dist)])
    
    # Sum up sinusoids
    for i in range(1,nchr+1):
        for j in range(1,octs+1):
            bin = nchr*(j-1-1) + i-1;
            omega = 2* N.pi * f_bins[bin];
            tones[i-1,:] = tones[i-1,:]+f_wts[bin]*N.sin(omega*tt);

    # resynth
    x = N.zeros([1,N.round(N.max(bts)*sr)])

    ee = N.round(bts[0])
    for b in range(1,F.shape[1]+1):
        ss = ee+1
        ee = N.round(sr * bts[b+1-1])
        twin = 0.5*(1-N.cos(N.asarray(range(dtsamp-1+1))/(dtsamp-1)*N.pi))
        twin = N.append(twin,N.ones([1,ee-ss+1]))
        #twin = N.append(twin,N.fliplr(twin))
        twin = N.append(twin,N.flipud(twin))
        sss = ss - dtsamp
        eee = ee + dtsamp
        if eee > x.shape[1]:
            twin = twin[1-1:len(twin)-(eee-x.shape[1])]
            eee = x.shape[1]
        if sss < 1:
            twin = twin[(2-sss)-1:len(twin)]
            sss = 1
  
        ll = 1+eee-sss
        dd = N.zeros([1,ll])
        for i in range(1,nchr+1):
            if F[i-1,b-1]>0:
                dd = dd + F[i-1,b-1]*tones[i-1,1-1:ll-1+1]

        x[0,sss-1:eee-1+1] = x[0,sss-1:eee-1+1] + N.asarray(twin) * dd;

    # return signal in x
    return x





def die_with_usage():
    """Help menu."""
    print 'library to resynthetize chroma beats'
    sys.exit(0)


if __name__ == '__main__' :
    """main, just for debugging."""
    if len(sys.argv) < 2 :
        die_with_usage()


