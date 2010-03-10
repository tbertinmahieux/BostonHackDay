"""
Simple code to run a typical experiment
MUST change the two lines starting with cd
"""

import os
import multiprocessing
import sys

import numpy as np
import scipy as sp
import scipy.io

import VQutils
import demos

featsDir = os.path.expanduser('~/projects/ismir10-patterns/beatFeats')
testFeatsDir = os.path.expanduser('~/projects/ismir10-patterns/uspop_mat')
outputDir = os.path.expanduser('~/projects/ismir10-patterns/experiments')

def do_experiment(experiment_dir,beats,bars,nCodes,nSamples=0,useFirsts=False,seed=0):
    """
    Performs an independant experiment!!!!
    """
    try:
        os.makedirs(experiment_dir)
    except OSError:
        pass

    np.random.seed(seed)

    args = dict(experiment_dir=experiment_dir, beats=beats, bars=bars,
                nCodes=nCodes, nSamples=nSamples, useFirsts=useFirsts,seed=seed)
    sp.io.savemat(os.path.join(experiment_dir, 'args.mat'), args)


    # TRAINING
    # go to the folder of features (per beat)
    os.chdir(featsDir)

    # load everything, unit: 1 bar, resized to 4 beats
    # key invariant, not downbeatinvariant
    featsNorm = demos.get_data_maxener(pSize=beats,keyInv=True,downBeatInv=False,bars=bars)

    # select 50K random samples out of it
    if nSamples == 0:
        nSamples = featsNorm.shape[0]
    if useFirsts:
        featsNorm = featsNorm[:nSamples]
        #r = range(featsNorm.shape[0]) # still randomize
        #np.random.shuffle(r)
        #featsNorm = featsNorm[r[:]]
        np.random.shuffle(featsNorm)
    else:
        r = range(featsNorm.shape[0])
        np.random.shuffle(r)
        featsNorm = featsNorm[r[:nSamples]]

    # train a codebook of size 100
    codebook,dists = VQutils.online_vq(featsNorm,nCodes,lrate=1e-2,nIter=200)
    sp.io.savemat(os.path.join(experiment_dir, 'codebook.mat'),
                  dict(codebook=codebook, dists=dists))
            
    # TESTING
    # go to the folder of test features (per beat)
    os.chdir(testFeatsDir)
    
    # load and test
    dists,avg_dists = demos.load_and_encode_data(codebook,pSize=beats,
                                                 keyInv=True,
                                                 downBeatInv=False,bars=bars)
    sp.io.savemat(os.path.join(experiment_dir, 'test.mat'),
                  dict(dists=dists, avg_dists=avg_dists))
    
    # report result (average sqaure distance per ... pixel?
    # with print outs to know what we are doing
    report = ['EXPERIMENT REPORT ******************************',
              'beats: %s , bars: %s , nCodes: %s , nSamples: %s'
              % (beats, bars, nCodes, nSamples)]
    if useFirsts:
        report.append(['we use firsts %s samples' % nCodes])
    report.extend(['np.average(avg_dists): %s' % np.average(avg_dists),
                   '************************************************'])
    reportstr = '\n'.join(reportstr)
    print reportstr
    f = open(os.path.join(experiment_dir, 'report.txt'))
    f.write(reportstr)
    f.close()



def die_with_usage():
    print 'launch all experiments set in main'
    print 'DONT FORGET TO HARDCODE PATHS'
    print 'usage:'
    print '   python for_ron -go nprocesses'
    sys.exit()


data_sizes = [1000, 5000, 10000, 50000, 100000, 250000, 500000, 1000000,
              2000000]
experiment_args = []
# EXPERIMENT SET 1: 1 bar 4 beats change data size
experiment_args.extend([(os.path.join(outputDir, 'set1exp%d' % n), 4,1,100,x)
                        for n,x in enumerate(data_sizes)])
# EXPERIMENT SET 2: 2 bar 8 beats change data size
experiment_args.extend([(os.path.join(outputDir, 'set2exp%d' % n), 8,2,100,x)
                        for n,x in enumerate(data_sizes)])
# EXPERIMENT SET 3: 0 bar 4 beats change data size 
experiment_args.extend([(os.path.join(outputDir, 'set3exp%d' % n), 4,0,100,x)
                        for n,x in enumerate(data_sizes)])
# EXPERIMENT SET 4: 1 bar 4 beats change data size, use first samples
experiment_args.extend([(os.path.join(outputDir, 'set4exp%d' % n), 4,1,100,x,True)
                        for n,x in enumerate(data_sizes)])

def do_experiment_wrapper(args):
    return do_experiment(*args)

if __name__ == '__main__':

    if len(sys.argv) < 3:
        die_with_usage()

    nprocesses = int(sys.argv[2])
    pool = multiprocessing.Pool(processes=nprocesses)
    pool.map(do_experiment_wrapper, experiment_args)

