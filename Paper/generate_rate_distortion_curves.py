"""
Code to generate the curve where the number of codes is squared
when the pattern size is doubled
"""



import os
import sys
import time
import numpy as np
import pylab as P



def create_fig():

    # DATA
    sizes = [1,2,4,8,16,32]
    #labels = ['2','4','16','256']
    labels = ['1/4','1/2','1','2','4','8']

    data1 = [0.027235,0.030586,0.043468,0.041904045]
    range1 = [0,1,2,3]
    # 2 beats - 1 bar - 2 bars - 4 bars
    data2 = [0.032700,0.054496,0.049674,0.049894]
    range2 = [1,2,3,4]
    # 1 bar - 2 bars - 4 bars - 8 bars
    data3 = [0.058189,0.055909,0.055811,0.054860]
    range3 = [2,3,4,5]

    # RATE
    # R = (#code index + #roll index) / (pattern size in beats)
    # R = (#codes used + 12 * #codes used) / (pattern size in beats)
    # R = (32/pattern_size * 13) / (pattern size in beats)
    # R = (32 * 13) / (pattern size in beats)^2
    rates = np.ones(len(sizes)) * 32 * 13
    rates /= np.array(sizes)

    # plot1
    P.plot(rates[range1],data1,'o-',label='1/4, 1/2, 1, 2 bars')
    # plot2
    P.plot(rates[range2],data2,'o-',label='1/2, 1, 2, 4 bars')
    # plot3
    P.plot(rates[range3],data3,'o-',label='1, 2, 4, 8 bars')

    # legend
    P.legend(loc='upper right')
    # main title
    P.title('Encoding error per code and pattern size')
    # done, show
    P.show()

    

def die_with_usage():
    """ help menu """
    print 'create figure'
    print 'usage: python generate_expo_codes_curves.py -go'
    sys.exit(0)


if __name__ == '__main__':

    if len(sys.argv) < 2:
        die_with_usage()

    create_fig()
