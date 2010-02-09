"""
More of a demo than a useful code
"""




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
    feats = [p.flatten() for p in data_iter]
    print 'all patterns acquired in ' + str(time.time()-tstart) + 'seconds'

    # some stats
    print 'number of matfiles: ' + str(len(allfiles))
    print 'number of patterns: ' + str(len(feats))

    # get features normalized (with default flags)
    tstart = time.time()
    featsNorm = [FU.normalize_pattern(np.array(p).reshape(1,len(p))) for p in feats]
    print 'all patterns normalized in ' + str(time.time()-tstart) + 'seconds'

    # get one nice big matrix
    featsNorm = np.array(featsNorm).reshape(len(featsNorm),featsNorm[0].shape[1])

    # and... we're done, let's launch the algo!


    import scipy.cluster
    import scipy.cluster.vq as SCVQ

    # run a 20 iteration, looking for a codebook of size 10
    tstart = time.time()
    codebook, distortion = SCVQ.kmeans2(featsNorm,10,20,minit='points')
    print 'kmeans performed in ' + str(time.time()-tstart) + 'seconds'
