"""
More of a demo than a useful code
"""


# to normalize for size
# np.average(np.sqrt(np.square(dists16)/(12*16.)))



def load_and_encode_data(codebook,pSize=4,keyInv=True,
                         downBeatInv=False,bars=1,partialbar=1,offset=0):
    """
    Load a dataset, and encode it with codebook
    Return dists, avg_dists
    """
    assert(codebook.shape[1] == pSize * 12)
    import VQutils
    # get data
    featsNorm = get_data_maxener(pSize=pSize,keyInv=keyInv,
                                 downBeatInv=downBeatInv,bars=bars,
                                 partialbar=partialbar,offset=offset)
    # encode
    best_code_per_p, dists, avg_dists = VQutils.find_best_code_per_pattern(featsNorm,codebook,scale=False)
    return dists, avg_dists


def get_data_maxener_16_true_false_bars2():
    return get_data_maxener(pSize=16,keyInv=True,downBeatInv=False,bars=2)

def get_data_maxener_8_true_false_bars2():
    return get_data_maxener(pSize=8,keyInv=True,downBeatInv=False,bars=2)

def get_data_maxener(pSize=4,keyInv=True,downBeatInv=False,bars=1,partialbar=1,offset=0):
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
        if partialbar < 1:
            assert(bars==1)
            data_iter.usePartialBar( partialbar )
    else:
        data_iter.useBars(0)                 # important to set it to zero!
        data_iter.setFeatsize( pSize )       # a pattern is a num. of beats
    if offset > 0:
        data_iter.setOffset(offset)
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



def encode_one_song(filename,codebook,pSize=8,keyInv=True,
                    downBeatInv=False,bars=2):
    """
    returns: song, encoding, song as MAT, encoding as MAT
    matrices are 'derolled'
    """
    import feats_utils as FU
    import numpy as np
    import data_iterator
    import VQutils

    # create data iterator
    data_iter = data_iterator.DataIterator()
    data_iter.setMatfiles([filename]) # set matfiles
    if bars > 0:
        data_iter.useBars( bars )            # a pattern spans 'bars' bars
    else:
        data_iter.useBars(0)                 # important to set it to zero!
        data_iter.setFeatsize( pSize )       # a pattern is a num. of beats
    data_iter.stopAfterOnePass(True)
    # load data
    featsNorm = [FU.normalize_pattern_maxenergy(p,pSize,keyInv,downBeatInv,retRoll=True).flatten() for p in data_iter]
    keyroll = np.array([x[1] for x in featsNorm])
    dbroll = np.array([x[2] for x in featsNorm])
    featsNorm = [x[0] for x in featsNorm]
    featsNorm = np.array(featsNorm)
    res = [np.sum(r) > 0 for r in featsNorm]
    res2 = np.where(res)
    featsNorm = featsNorm[res2]
    keyroll = keyroll[res2]
    dbroll = dbroll[res2]
    assert(dbroll.shape[0] == keyroll.shape[0])
    assert(dbroll.shape[0] == featsNorm.shape[0])
    # find code per pattern
    best_code_per_p, dists, avg_dists = VQutils.find_best_code_per_pattern(featsNorm,codebook)
    best_code_per_p = np.asarray([int(x) for x in best_code_per_p])
    encoding = codebook[best_code_per_p]
    # transform into 2 matrices
    assert(featsNorm.shape[0] == encoding.shape[0])
    #featsNormMAT = np.concatenate([x.reshape(12,pSize) for x in featsNorm],axis=1)
    featsNormMAT = np.concatenate([np.roll(np.roll(featsNorm[x].reshape(12,pSize),-keyroll[x],axis=0),-dbroll[x],axis=1) for x in range(featsNorm.shape[0])],axis=1)
    #encodingMAT = np.concatenate([x.reshape(12,pSize) for x in encoding],axis=1)
    encodingMAT = np.concatenate([np.roll(np.roll(encoding[x].reshape(12,pSize),-keyroll[x],axis=0),-dbroll[x],axis=1) for x in range(featsNorm.shape[0])],axis=1)
    # return
    return best_code_per_p,featsNorm,encoding,featsNormMAT,encodingMAT


def get_codeword_histogram(codewords, ncodewords):
    hist = np.zeros(ncodewords)
    for cw in codewords:
        hist[cw] += 1
    return hist


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
