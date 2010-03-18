"""
Script to generate one figure from our Shingle paper
The one with accuracy per beats

T. Bertin-Mahieux (2010) Columbia University
tb2332@columbia.edu
"""

import os
import sys
import time
import numpy as np
import pylab as P



def create_fig():

    data = [0.013725,0.025346,0.032688,0.037181,0.040858,0.043237,0.045515,
            0.046832,0.048579,0.049530,0.050717,0.052020]
    labels = range(1,13)
    assert len(data) == len(labels),'labels should go from 1 to 12'

    # START DISPLAYING STUFF
    
    P.figure()
    P.hold(True)
    # axis
    xmin = 1
    xmax = len(labels)
    ymin = 0.013
    ymax = 0.053
    P.axis([xmin,xmax,ymin,ymax])

    # plot data
    P.plot(labels,data,'-o')

    # plot 4 - 8
    P.axvline(linewidth=1,color='r',x=4,ymin=0,ymax=1)
    P.axvline(linewidth=1,color='r',x=8,ymin=0,ymax=1)

    # labels
    #P.xticks(P.arange(len(labels)),list(labels))

    # x titles and y titles
    P.xlabel('number of beats')
    P.ylabel('average error')

    # main title
    P.title('Encoding error per number of beats (bar information ignored)')

    P.show()






def die_with_usage():
    """ help menu """
    print 'create figure'
    print 'usage: python generate_beats_curves.py -go'
    sys.exit(0)


if __name__ == '__main__':

    if len(sys.argv) < 2:
        die_with_usage()

    create_fig()
