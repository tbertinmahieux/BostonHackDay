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

    sizes = [0,1000,5000,10000,50000,100000,250000,500000]
    labels = ['0','1K','5K','10K','50K','100K','250K','500K']
    
    # data 1 bar 4 beats
    bar1beat4 = [0.052247,0.041941,0.038449,0.036892,0.036722,
                 0.035973,0.035910,0.035904]
    # data 2 bars 8 beats
    bar2beat8 = [0.066548,0.049282,0.046441,0.045507,0.044301,
                 0.044183,0.043896,0.043906]
    # data 1 bar 4 beats not random! first K in data
    bar1beat4FIRST = [0.052247,0.044359,0.039107,0.038257,0.036157,
                      0.036697,0.036598,0.035935]
    # data 0 bar 4 beats
    #bar0beat4 = [0.155474,0.149619,0.148056,0.146328,0.146004,0.146024,0.145962,0.145873,0.145955]


    # START DISPLAYING STUFF
    
    P.figure()
    P.hold(True)
    # axis
    xmin = 0
    xmax = len(labels)
    ymin = 0.035
    ymax = 0.067
    P.axis([xmin,xmax,ymin,ymax])

    # plot1
    P.plot(bar1beat4,'o-',label='1 bar 4 beats')
    # plot2
    P.plot(bar2beat8,'x-',label='2 bars 8 beats')
    # plot3
    P.plot(bar1beat4FIRST,'--',label='1 bar 4 beats non random')
    # plot4
    #P.plot(bar0beat4,'-.',label='0 bars 4 beats')
    
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
