"""
Code to generate the curve where the number of codes is squared
when the pattern size is doubled
"""



import os
import sys
import time
import numpy as np
import pylab as P



def create_fig(penalty=True):

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


    # PENALTY
    if penalty:
        """
        for k in range(len(data1)):
            data1[k] = data1[k] / sizes[range1[k]] * np.log2(12)
        for k in range(len(data2)):
            data2[k] = data2[k] / sizes[range2[k]] * np.log2(12)
        for k in range(len(data3)):
            data3[k] = data3[k] / sizes[range3[k]] * np.log2(12)
        """
        for k in range(len(data1)):
            data1[k] = data1[k] / np.log2(pow(12,range1[k])+1)
        for k in range(len(data2)):
            data2[k] = data2[k] / np.log2(pow(12,range2[k])+1)
        for k in range(len(data3)):
            data3[k] = data3[k] / np.log2(pow(12,range3[k])+1)
    
    # START DISPLAYING STUFF
    
    P.figure()
    P.hold(True)
    # axis
    xmin = 0
    xmax = len(labels)
    if not penalty:
        ymin = 0.027
        ymax = 0.059
    else:
        ymin = min(min(data1),min(data2),min(data3))
        ymax = max(max(data1),max(data2),max(data3))
    P.axis([xmin,xmax,ymin,ymax])

    # plot1
    P.plot(range1,data1,'o-',label='1/4, 1/2, 1, 2 bars')
    # plot2
    P.plot(range2,data2,'x-',label='1/2, 1, 2, 4 bars')
    # plot3
    P.plot(range3,data3,'<-',label='1, 2, 4, 8 bars')

    """
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
    """
                
    # labels
    P.xticks(P.arange(len(labels)),list(labels))
    
    # x titles and y titles
    P.xlabel('bars')
    P.ylabel('average error')

    # legend
    if not penalty:
        P.legend(loc='lower right')
    else:
        P.legend(loc='upper right')
        
    # main title
    P.title('Encoding error per code and pattern size')

    P.show()


    

def die_with_usage():
    """ help menu """
    print 'create figure'
    print 'usage: python generate_expo_codes_curves.py -go (-penalty)'
    sys.exit(0)


if __name__ == '__main__':

    if len(sys.argv) < 2:
        die_with_usage()

    penalty = False
    if len(sys.argv) > 2 and sys.argv[2] == '-penalty':
        penalty=True

    create_fig(penalty)

