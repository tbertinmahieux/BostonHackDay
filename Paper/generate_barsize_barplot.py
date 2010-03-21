
import scipy
import scipy.io
import numpy as np
import sys




def generate():
    """ generate the graph """
    import pylab as P


    # load data
    mat = scipy.io.loadmat('/home/thierry/Desktop/tmpPaper/number_beats_per_bar.mat',struct_as_record=True)

    numBars = mat['numBars']
    numBars = numBars.flatten()

    # rearrange data, from 1 beat to >9
    res = numBars[1:10]
    
    # bar plot
    P.bar(np.array(range(9))+.6,res)

    # text
    P.xlabel('number of beats per bar')
    P.ylabel('frequency in training data')
    P.title('number of beats per bar in the training data based on EchoNest API')

    # x ticks
    labels = [str(x) for x in (range(1,9))]
    labels.append('>9')
    P.xticks(P.arange(1,len(labels)+1),list(labels))


    # done, show
    P.show()


def die_with_usage():
    """ HELP MENU """
    print 'to generate the bar plot that shows size per bar in paper:'
    print '    python generate_barsize_barplot -go'
    sys.exit(0)
    



if __name__ == '__main__':

    if len(sys.argv) < 2:
        die_with_usage()

    generate()
