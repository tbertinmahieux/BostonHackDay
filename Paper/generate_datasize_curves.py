"""
Script to generate one figure from our Shingles paper

T. Bertin-Mahieux (2010) Columbia University
tb2332@columbia.edu
"""

import os
import sys
import time
import numpy as np
import pylab as P



def create_fig():

    # DATA

    sizes = [1000,5000,10000,50000,100000,250000,500000,1000000,2000000]
    labels = ['1K','5K','10K','50K','100K','250K','500K','1M','2M']
    
    # data 1 bar 4 beats
    bar1beat4 = [0.150370,0.145327,0.143567,0.142257,0.142195,0.141796,0.142081,0.142118,0.141596]
    # data 2 bars 8 beats
    bar2beat8 = [0.171300,0.165960,0.165063,0.162439,0.161588,0.161910,0.161233,0.161057,0.161068]
    # data 1 bar 4 beats not random! first K in data
    bar1beat4FIRST = [0.157142,0.150045,0.148357,0.146069,0.145765,0.145819,0.145772,0.145924,0]
    # data 0 bar 4 beats
    bar0beat4 = [0.155474,0.149619,0.148056,0.146328,0.146004,0,0,0,0]


    # START DISPLAYING STUFF
    
    P.figure()
    P.hold(True)
    # axis
    xmin = 0
    xmax = len(labels)
    ymin = 0.13
    ymax = 0.17
    P.axis([xmin,xmax,ymin,ymax])

    # plot1
    P.plot(bar1beat4,'o-',label='1 bar 4 beats')
    # plot2
    P.plot(bar2beat8,'x-',label='2 bars 8 beats')
    # plot3
    P.plot(bar1beat4FIRST,'--',label='1 bar 4 beats non random')
    # plot4
    P.plot(bar0beat4,'-.',label='0 bars 4 beats')
    
    # labels
    P.xticks(P.arange(len(labels)),list(labels))
    
    # x titles and y titles
    P.xlabel('data size')
    P.ylabel('average error')

    # legend
    P.legend()

    # main title
    P.title('Encoding error per training data size for certain conditions')

    P.show()
















def die_with_usage():
    """ help menu """
    print 'create figure'
    print 'usage: python generate_datasize_curves.py -go'
    sys.exit(0)


if __name__ == '__main__':

    if len(sys.argv) < 2:
        die_with_usage()

    create_fig()
