"""
Simple code to run a typical experiment
MUST change the two lines starting with cd
"""


import os
import sys
import VQutils
import demos
import numpy as np


featsDir = 
testFeatsDir = 


def do_experiment(beats,bars,nCodes,nSamples=0,useFirsts=False):
    """
    Performs an independant experiment!!!!
    """

    # TRAINING
    # go to the folder of features (per beat)
    cd featsDir

    # load everything, unit: 1 bar, resized to 4 beats
    # key invariant, not downbeatinvariant
    featsNorm = demos.get_data_maxener(pSize=beats,keyInv=True,downBeatInv=False,bars=bars)

    # select 50K random samples out of it
    if nSamples == 0:
        nSamples = featsNorm.shape[0]
    if useFirsts:
        featsNorm = featsNorm[:nSamples]
        r = range(featsNorm.shape[0]) # still randomize
        np.random.shuffle(r)
        featsNorm = featsNorm[r[:]]
    else:
        r = range(featsNorm.shape[0])
        np.random.shuffle(r)
        featsNorm = featsNorm[r[:nSamples]]

    # train a codebook of size 100
    codebook,dists = VQutils.online_vq(featsNorm,100,lrate=1e-2,nIter=200):
        
    # TESTING
    # go to the folder of test features (per beat)
    cd testFeatsDir
    
    # load and test
    dists,avg_dists = demos.load_and_encode_data(codebook,pSize=beats,
                                                 keyInv=True,
                                                 downBeatInv=False,bars=bars)
    
    # report result (average sqaure distance per ... pixel?
    # with print outs to know what we are doing
    print 'beats:',beats,', bars:',bars,', nCodes:',nCodes,', nSamples:',nSamples
    if useFirsts:
        print 'we use firsts ', nCodes, ' samples'
    print 'np.average(avg_dists):',np.average(avg_dists)




def die_with_usage():
    print 'launch all experiments set in main'
    print 'DONT FORGET TO HARDCODE PATHS'
    print 'usage:'
    print '   python for_ron -go'
    sys.exit()


    # EXPERIMENT SET 1: 1 bar 4 beats change data size
    # exp1
    do_experiment(4,1,100,nSamples=1000)
    # exp2
    do_experiment(4,1,100,nSamples=5000)
    # exp3
    do_experiment(4,1,100,nSamples=10000)
    # exp4
    do_experiment(4,1,100,nSamples=50000)
    # exp5
    do_experiment(4,1,100,nSamples=100000)
    # exp6
    do_experiment(4,1,100,nSamples=250000)
    # exp7
    do_experiment(4,1,100,nSamples=500000)
    # exp8
    do_experiment(4,1,100,nSamples=1000000)
    # exp9
    do_experiment(4,1,100,nSamples=2000000)

    # EXPERIMENT SET 2: 2 bar 8 beats change data size 
    # exp1
    do_experiment(8,2,100,nSamples=1000)
    # exp2
    do_experiment(8,2,100,nSamples=5000)
    # exp3
    do_experiment(8,2,100,nSamples=10000)
    # exp4
    do_experiment(8,2,100,nSamples=50000)
    # exp5
    do_experiment(8,2,100,nSamples=100000)
    # exp6
    do_experiment(8,2,100,nSamples=250000)
    # exp7
    do_experiment(8,2,100,nSamples=500000)
    # exp8
    do_experiment(8,2,100,nSamples=1000000)
    # exp9
    do_experiment(8,2,100,nSamples=2000000)

