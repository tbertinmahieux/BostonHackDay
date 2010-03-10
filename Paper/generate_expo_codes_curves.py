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
    labels = ['2','4','16','256']

    # 2 beats - 1 bar - 2 bars - 4 bars
    data1 = [0.066416,0.050712,0.050172,0.049894]
    # 1 bar - 2 bars - 4 bars - 8 bars
    data2 = [0.058435,0.054792,0.055489,0.054860]

    
    # START DISPLAYING STUFF
    
    P.figure()
    P.hold(True)
    # axis
    xmin = 0
    xmax = len(labels)
    ymin = 0.49
    ymax = 0.67
    P.axis([xmin,xmax,ymin,ymax])

    # plot1
    P.plot(data1,'o-',label='2 beats, 1 bar, 2 bars, 4 bars')
    # plot2
    P.plot(data2,'x-',label='1 bar, 2 bars, 4 bars, 8 bars')

    # ANNOTATION
    gcf = P.gca()
    # data1 pt1
    gcf.annotate('2 beats', xy=(0, data1[0]),  xycoords='data',
                xytext=(50, 30), textcoords='offset points',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3,rad=.2")
                )
    # data1 pt2
    gcf.annotate('1 bar', xy=(1, data1[1]),  xycoords='data',
                xytext=(50, 30), textcoords='offset points',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3,rad=.2")
                )
    # data1 pt3
    gcf.annotate('2 bars', xy=(2, data1[2]),  xycoords='data',
                xytext=(50, 30), textcoords='offset points',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3,rad=.2")
                )
    # data1 pt4
    gcf.annotate('4 bars', xy=(3, data1[3]),  xycoords='data',
                xytext=(-50, -30), textcoords='offset points',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3,rad=.2")
                )
    # data2 pt1
    gcf.annotate('1 bar', xy=(0, data2[0]),  xycoords='data',
                xytext=(50, -60), textcoords='offset points',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3,rad=-.2")
                )
    # data2 pt2
    gcf.annotate('2 bars', xy=(1, data2[1]),  xycoords='data',
                xytext=(50, 30), textcoords='offset points',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3,rad=.2")
                )
    # data2 pt3
    gcf.annotate('4 bars', xy=(2, data2[2]),  xycoords='data',
                xytext=(50, 30), textcoords='offset points',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3,rad=.2")
                )
    # data2 pt4
    gcf.annotate('8 bars', xy=(3, data2[3]),  xycoords='data',
                xytext=(-50, 30), textcoords='offset points',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3,rad=-.2")
                )
    
    # labels
    P.xticks(P.arange(len(labels)),list(labels))
    
    # x titles and y titles
    P.xlabel('number of codes')
    P.ylabel('average error')

    # legend
    P.legend()

    # main title
    P.title('Encoding error per code and pattern size')




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

