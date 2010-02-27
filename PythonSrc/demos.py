"""
More of a demo than a useful code
"""


def get_data_maxener_16_true_false_bars2():
    return get_data_maxener(pSize=16,keyInv=True,downBeatInv=False,bars=2)

def get_data_maxener_8_true_false_bars2():
    return get_data_maxener(pSize=8,keyInv=True,downBeatInv=False,bars=2)


def get_data_maxener(pSize=16,keyInv=True,downBeatInv=False,bars=2):
    """
    Util function for something we do all the time
    Remove the empty patterns
    INPUT:
        - pSize        (default: 16)
        - keyInv       (default: True)
        - downBeatInv  (default: False)
        - bars         (default: 2)
    """
    import data_iterator
    import feats_utils as FU
    import numpy as np
    import time
    # start time
    tstart = time.time()
    # get maltab files
    allfiles = FU.get_all_matfiles('.')
    print len(allfiles),' .mat files found'
    # create and set iterator
    data_iter = data_iterator.DataIterator()
    data_iter.setMatfiles(allfiles) # set matfiles
    if bars > 0:
        data_iter.useBars( bars )            # a pattern spans 'bars' bars
    else:
        data_iter.useBars(0)                 # important to set it to zero!
        data_iter.setFeatsize( pSize )       # a pattern is a num. of beats
    data_iter.stopAfterOnePass(True)# stop after one full iteration
    # get features
    featsNorm = [FU.normalize_pattern_maxenergy(p,pSize,keyInv,downBeatInv).flatten() for p in data_iter]
    print 'found ', len(featsNorm),' patterns before removing empty ones'
    # make it an array
    featsNorm = np.array(featsNorm)
    # remove empyt patterns
    res = [np.sum(r) > 0 for r in featsNorm]
    res2 = np.where(res)
    featsNorm = featsNorm[res2]
    # time?
    print 'all patterns acquired and normalized in ' + str(time.time()-tstart) + 'seconds'
    print 'featsNorm.shape = ',featsNorm.shape
    return featsNorm


def get_all_barfeats():
    """
    Returns all barfeats, we assume we're at the top of beatFeats dir
    """
    import data_iterator
    import feats_utils as FU
    import numpy as np
    import time

    # get maltab files
    allfiles = FU.get_all_matfiles('.')

    # create and set iterator
    data_iter = data_iterator.DataIterator()
    data_iter.setMatfiles(allfiles) # set matfiles
    data_iter.useBars(2)            # a pattern spans two bars
    data_iter.stopAfterOnePass(True)# stop after one full iteration

    # get all feats
    tstart = time.time()
    featsNorm = [FU.normalize_pattern(np.array(p),16,True,True).flatten() for p in data_iter]
    print 'all patterns acquired and normalized in ' + str(time.time()-tstart) + 'seconds'

    # some stats
    print 'number of matfiles: ' + str(len(allfiles))
    print 'number of patterns: ' + str(len(featsNorm))

    # get one nice big matrix
    featsNorm = np.array(featsNorm)

    # and... we're done, let's launch the algo!


    import scipy.cluster
    import scipy.cluster.vq as SCVQ

    # run a 20 iteration, looking for a codebook of size 10
    tstart = time.time()
    codebook, distortion = SCVQ.kmeans2(featsNorm,10,20,minit='points')
    print 'kmeans performed in ' + str(time.time()-tstart) + 'seconds'
