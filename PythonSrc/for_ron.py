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

def do_experiment(experiment_dir,beats,bars,nCodes,nSamples=0,useFirsts=False,seed=0,offset=0,partialbar=1):
    """
    Performs an independant experiment!!!!
    """
    try:
        os.makedirs(experiment_dir)
    except OSError:
        pass

    np.random.seed(seed)

    args = dict(experiment_dir=experiment_dir, beats=beats, bars=bars,
                nCodes=nCodes, nSamples=nSamples, useFirsts=useFirsts,seed=seed,
                offset=offset,partialbar=partialbar)
    sp.io.savemat(os.path.join(experiment_dir, 'args.mat'), args)

    # TRAINING
    # go to the folder of features (per beat)
    os.chdir(featsDir)

    if not os.path.exists(os.path.join(experiment_dir, 'codebook.mat')):
        # load everything, unit: 1 bar, resized to 4 beats
        # key invariant, not downbeatinvariant
        featsNorm = demos.get_data_maxener(pSize=beats,keyInv=True,downBeatInv=False,bars=bars,offset=offset,partialbar=partialbar)
        
        # select nSamples random samples out of it
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

        del featsNorm
    else:
        mat = sp.io.loadmat(os.path.join(experiment_dir, 'codebook.mat'))
        codebook = mat['codebook']
        dists = codebook['dists']
        
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
              'beats: %s , bars: %s , nCodes: %s , nSamples: %s , offset: %s , partialbar: %s'
              % (beats, bars, nCodes, nSamples, offset, partialbar)]
    if useFirsts:
        report.append('we use firsts %s samples' % nCodes)
    report.extend(['np.average(avg_dists): %s' % np.average(avg_dists),
                   '************************************************', ''])
    reportstr = '\n'.join(report)
    print reportstr
    f = open(os.path.join(experiment_dir, 'report.txt'), 'w')
    f.write(reportstr)
    f.close()



def die_with_usage():
    print 'launch all experiments set in main'
    print 'DONT FORGET TO HARDCODE PATHS'
    print 'usage:'
    print '   python for_ron -go nprocesses experiment_set_number(1-5)'
    sys.exit()


def do_experiment_wrapper(args):
    return do_experiment(*args)


data_sizes = [1000, 5000, 10000, 50000, 100000, 250000, 500000, 1000000,
              2000000]
experiment_args = [
    # EXPERIMENT SET 1: 1 bar 4 beats change data size
    [(os.path.join(outputDir, 'set1exp%d' % n), 4,1,100,x)
     for n,x in enumerate(data_sizes)],
    # EXPERIMENT SET 2: 2 bar 8 beats change data size
    [(os.path.join(outputDir, 'set2exp%d' % n), 8,2,100,x)
     for n,x in enumerate(data_sizes)],
    # EXPERIMENT SET 3: 0 bar 4 beats change data size 
    [(os.path.join(outputDir, 'set3exp%d' % n), 4,0,100,x)
     for n,x in enumerate(data_sizes)],
    # EXPERIMENT SET 4: 1 bar 4 beats change data size, use first samples
    [(os.path.join(outputDir, 'set4exp%d' % n), 4,1,100,x,True)
     for n,x in enumerate(data_sizes)],
    # EXPERIMENT SET 5: table 1 in the paper 
    [(os.path.join(outputDir, 'set5exp0'), 3, 0, 100, 50000),
     (os.path.join(outputDir, 'set5exp1'), 5, 0, 100, 50000),
     (os.path.join(outputDir, 'set5exp2'), 7, 0, 100, 50000),
     (os.path.join(outputDir, 'set5exp3'), 4, 1, 100, 50000),
     (os.path.join(outputDir, 'set5exp4'), 8, 2, 100, 50000),
     (os.path.join(outputDir, 'set5exp5'), 12, 2, 100, 50000),
     (os.path.join(outputDir, 'set5exp6'), 16, 2, 100, 50000)],
    # EXPERIMENT SET 6: table 2 in paper (codebook size vs distortion with
    #  constant numbre of training samples)
    [(os.path.join(outputDir, 'set6exp%d' % n), 4, 1, x, 500000)
     for n,x in enumerate([1, 10, 50, 100, 200, 500, 1000])],
    # EXPERIMENT SET 7: exp 5 (table 1) extended: bar alignment vs taking every n-beats
    [(os.path.join(outputDir, 'set7exp0'), 2, 0, 100, 50000),
     (os.path.join(outputDir, 'set7exp1'), 4, 0, 100, 50000),
     (os.path.join(outputDir, 'set7exp2'), 6, 0, 100, 50000),
     (os.path.join(outputDir, 'set7exp3'), 8, 0, 100, 50000),
     (os.path.join(outputDir, 'set7exp4'), 9, 0, 100, 50000),
     (os.path.join(outputDir, 'set7exp5'), 10, 0, 100, 50000),
     (os.path.join(outputDir, 'set7exp6'), 11, 0, 100, 50000),   
     (os.path.join(outputDir, 'set7exp7'), 12, 0, 100, 50000),
     (os.path.join(outputDir, 'set7exp8'), 1, 0, 100, 50000),
     (os.path.join(outputDir, 'set7exp9'), 3, 0, 100, 50000),
     (os.path.join(outputDir, 'set7exp10'), 5, 0, 100, 50000),
     (os.path.join(outputDir, 'set7exp11'), 7, 0, 100, 50000)],
    # EXPERIMENT SET 8: testing offsets
    [(os.path.join(outputDir, 'set8exp0'),4,1,100,50000,False,0,0,1),
     (os.path.join(outputDir, 'set8exp1'),4,1,100,50000,False,0,0.25,1),
     (os.path.join(outputDir, 'set8exp2'),4,1,100,50000,False,0,0.50,1),
     (os.path.join(outputDir, 'set8exp3'),4,1,100,50000,False,0,0.75,1),
     (os.path.join(outputDir, 'set8exp4'),4,1,100,50000,False,0,0,.25),
     (os.path.join(outputDir, 'set8exp5'),4,1,100,50000,False,0,0.25,.25),
     (os.path.join(outputDir, 'set8exp6'),4,1,100,50000,False,0,0.50,.25),
     (os.path.join(outputDir, 'set8exp7'),4,1,100,50000,False,0,0.75,.25),
     (os.path.join(outputDir, 'set8exp8'),4,1,100,50000,False,0,0,.50),
     (os.path.join(outputDir, 'set8exp9'),4,1,100,50000,False,0,0.50,.25)],
    # EXPERIMENT SET 9: for visualization
    [(os.path.join(outputDir, 'set9exp0'),4,1,500,500000),
     (os.path.join(outputDir, 'set9exp1'),8,2,500,500000),
     (os.path.join(outputDir, 'set9exp2'),16,4,500,500000),
     (os.path.join(outputDir, 'set9exp3'),32,8,500,500000)],
    # EXPERIMENT SET 10: more on offset with partial bars
    [(os.path.join(outputDir, 'set10exp0'),4,1,100,50000,False,0,0.00,.50),
     (os.path.join(outputDir, 'set10exp1'),4,1,100,50000,False,0,0.25,.50),
     (os.path.join(outputDir, 'set10exp2'),4,1,100,50000,False,0,0.50,.50),
     (os.path.join(outputDir, 'set10exp3'),4,1,100,50000,False,0,0.75,.50),
     (os.path.join(outputDir, 'set10exp4'),4,1,100,50000,False,0,0.00,.75),
     (os.path.join(outputDir, 'set10exp5'),4,1,100,50000,False,0,0.25,.75),
     (os.path.join(outputDir, 'set10exp6'),4,1,100,50000,False,0,0.50,.75),
     (os.path.join(outputDir, 'set10exp7'),4,1,100,50000,False,0,0.75,.75)],
    # EXPERIMENT SET 11: redoing pattern doubles => nCodes squares
    [(os.path.join(outputDir, 'set11exp0'),2,1,2,128000,False,0,0.00,.25),
     (os.path.join(outputDir, 'set11exp1'),2,1,2,128000,False,0,0.25,.25),
     (os.path.join(outputDir, 'set11exp2'),2,1,2,128000,False,0,0.50,.25),
     (os.path.join(outputDir, 'set11exp3'),2,1,2,128000,False,0,0.75,.25),
     (os.path.join(outputDir, 'set11exp4'),2,1,4,128000,False,0,0.00,.50),
     (os.path.join(outputDir, 'set11exp5'),2,1,4,128000,False,0,0.25,.50),
     (os.path.join(outputDir, 'set11exp6'),2,1,4,128000,False,0,0.50,.50),
     (os.path.join(outputDir, 'set11exp7'),2,1,4,128000,False,0,0.75,.50),
     (os.path.join(outputDir, 'set11exp7'),4,1,16,128000,False,0),
     (os.path.join(outputDir, 'set11exp7'),8,2,256,128000,False,0),
     (os.path.join(outputDir, 'set11exp4'),2,1,2,128000,False,0,0.00,.50),
     (os.path.join(outputDir, 'set11exp5'),2,1,2,128000,False,0,0.25,.50),
     (os.path.join(outputDir, 'set11exp6'),2,1,2,128000,False,0,0.50,.50),
     (os.path.join(outputDir, 'set11exp7'),2,1,2,128000,False,0,0.75,.50),
     (os.path.join(outputDir, 'set11exp8'),1,1,2,128000,False,0,0.00,.25),
     (os.path.join(outputDir, 'set11exp9'),1,1,2,128000,False,0,0.25,.25),
     (os.path.join(outputDir, 'set11exp10'),1,1,2,128000,False,0,0.50,.25),
     (os.path.join(outputDir, 'set11exp11'),1,1,2,128000,False,0,0.75,.25)],
    # EXPERIMENT SET 12: for visualization
    [(os.path.join(outputDir, 'set12exp0'),4,1,500,500000),
     (os.path.join(outputDir, 'set12exp1'),8,2,500,500000),
     (os.path.join(outputDir, 'set12exp2'),16,4,500,500000),
     (os.path.join(outputDir, 'set12exp3'),32,8,500,500000)],
    # EXPERIMENT SET 13: for visualization
    [(os.path.join(outputDir, 'set13exp0'),4,1,200,200000),
     (os.path.join(outputDir, 'set13exp1'),8,2,200,200000),
     (os.path.join(outputDir, 'set13exp2'),16,4,200,200000),
     (os.path.join(outputDir, 'set13exp3'),32,8,200,200000)],
    # EXPERIMENT SET 14: getting error with offset to be the same by adding
    # more codes, to get the result for a bar aligned code
    [(os.path.join(outputDir, 'set14exp0'),4,1,115,50000,False,0,0.25,1),
     (os.path.join(outputDir, 'set14exp1'),4,1,130,50000,False,0,0.25,1),
     (os.path.join(outputDir, 'set14exp2'),4,1,145,50000,False,0,0.25,1),
     (os.path.join(outputDir, 'set14exp3'),4,1,160,50000,False,0,0.25,1),
     (os.path.join(outputDir, 'set14exp4'),4,1,115,50000,False,0,0.50,1),
     (os.path.join(outputDir, 'set14exp5'),4,1,130,50000,False,0,0.50,1),
     (os.path.join(outputDir, 'set14exp6'),4,1,145,50000,False,0,0.50,1),
     (os.path.join(outputDir, 'set14exp7'),4,1,160,50000,False,0,0.50,1),
     (os.path.join(outputDir, 'set14exp8'),4,1,115,50000,False,0,0.75,1),
     (os.path.join(outputDir, 'set14exp9'),4,1,130,50000,False,0,0.75,1),
     (os.path.join(outputDir, 'set14exp10'),4,1,145,50000,False,0,0.75,1),
     (os.path.join(outputDir, 'set14exp11'),4,1,160,50000,False,0,0.75,1)],
    # EXPERIMENT SET 15: redoing set 11 with good names...!
    [(os.path.join(outputDir, 'set15exp0'),1,1,2,128000,False,0,0.00,.25),
     (os.path.join(outputDir, 'set15exp1'),1,1,2,128000,False,0,0.25,.25),
     (os.path.join(outputDir, 'set15exp2'),1,1,2,128000,False,0,0.50,.25),
     (os.path.join(outputDir, 'set15exp3'),1,1,2,128000,False,0,0.75,.25),
     (os.path.join(outputDir, 'set15exp4'),2,1,4,128000,False,0,0.00,.50),
     (os.path.join(outputDir, 'set15exp5'),2,1,4,128000,False,0,0.25,.50),
     (os.path.join(outputDir, 'set15exp6'),2,1,4,128000,False,0,0.50,.50),
     (os.path.join(outputDir, 'set15exp7'),2,1,4,128000,False,0,0.75,.50),
     (os.path.join(outputDir, 'set15exp8'),4,1,16,128000,False,0),
     (os.path.join(outputDir, 'set15exp9'),8,2,256,128000,False,0),
     (os.path.join(outputDir, 'set15exp10 '),2,1,2,128000,False,0,0.00,.50),
     (os.path.join(outputDir, 'set15exp11'),2,1,2,128000,False,0,0.25,.50),
     (os.path.join(outputDir, 'set15exp12'),2,1,2,128000,False,0,0.50,.50),
     (os.path.join(outputDir, 'set15exp13'),2,1,2,128000,False,0,0.75,.50),
     (os.path.join(outputDir, 'set15exp14'),4,1,4,128000,False,0),
     (os.path.join(outputDir, 'set15exp15'),8,2,16,128000,False,0),
     (os.path.join(outputDir, 'set15exp16'),16,4,256,128000,False,0),
     (os.path.join(outputDir, 'set15exp17'),4,1,16,128000,False,0),
     (os.path.join(outputDir, 'set15exp18'),8,2,256,128000,False,0)],
    ]

        
if __name__ == '__main__':
    if len(sys.argv) < 4:
        die_with_usage()

    nprocesses = int(sys.argv[2])
    pool = multiprocessing.Pool(processes=nprocesses)

    # Python indexes from 0, the argument indexes from 1.
    experiment_set_number = int(sys.argv[3]) - 1
    args = experiment_args[experiment_set_number]
    try:
        args = [args[int(sys.argv[4]) - 1]]
    except IndexError:
        pass
    
    pool.map(do_experiment_wrapper, args)
