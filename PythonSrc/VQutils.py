#! /usr/bin/env python
"""
Set of functions to perform vector quantization,
meaning find a set of 'typical patterns', get them in
a codebook, then reencode new vectors

Relies in part on the scipy.cluster.vq package
and scipy.spatial.distance

T. Bertin-Mahieux (2010) Columbia University
www.columbia.edu/~tb2332/
"""

import numpy as np
import scipy as sp
import time


def euclidean_dist(a,b):
    """
    Typical euclidean distance. A and B must be row vectors!!!!
    """
    # tested for vectors of size 200, slightly beats:
    #return np.sqrt(np.square(a-b).sum())
    # following faster, but does not work on boar?
    #return np.sqrt(np.dot((a-b),(a-b).T)[0][0])
    return np.sqrt(np.square(a-b).sum())

def euclidean_norm(a):
    """ regular euclidean norm of a numpy vector """
    #return np.sqrt((a*a).sum())
    # slightly faster:
    return np.sqrt(np.square(a).sum())

def normalize(a):
    """ divides by the euclidean norm """
    try:
        return a/euclidean_norm(a)
    except ZeroDivisionError:
        return a
        

def projection_factor(v_from,v_to,v_to_normalized=False):
    """ if we project vector 'v_from' onto vector 'v_to', by how
    much (alpha) do we need to multiply 'to' in order to minimize
    the distance between 'v_from' and 'alpha * v_to'
    It can be expressed as: |A|cos(theta)/|B| = dot(A,B)/|B|^2
    with A projected onto B
    """
    if v_to_normalized:
        # trivial case
        return np.dot(v_from,v_to)
    else:
        # we must normalized by norm^2
        normTo = euclidean_norm(v_to)
        return np.dot(v_from,v_to) / (normTo * normTo)


def encode_scale_oneiter(signal,codebook,cbIsNormalized=False):
    """
    Find the best pattern in the codebook, returns the distance.
    codebook is one pattern per row.
    We allow the pattern to be scaled to be as close as possible
    to the signal
    Returns <indexes of the pattern>,<scalings>,<distances>
    Indexes are returned starting from the code which has the smallest
    scaled distance. If you only care about this code, do:
    idx = idxs[0], alpha = alphas[idx], dist = dists[idx]
    If codebook is normalized, set cbIsNormalized to True,
    it is slightly faster. (Normalized means each element has an
    euclidean norm of 1)
    """
    # find the right scaling
    if cbIsNormalized:
        alphas = np.inner(signal,codebook).T
    else:
        alphas = [projection_factor(signal,r,cbIsNormalized) for r in codebook[:]]
        alphas = np.array(alphas).reshape(codebook.shape[0],1)
    # scale the codebook and compute the distance
    dists = [euclidean_dist(signal,r) for r in (alphas*codebook)[:]]
    # return the index, scaling, and distance for the MIN DISTANCE
    idxs = np.argsort(dists)
    return idxs,alphas.flatten(),dists


def encode_scale(signal,codebook,thresh,cbIsNormalized=False,maxIter=1e8):
    """
    Iteratively encode a signal using the codebook.
    - find the best element in the codebook (with scaling)
    - removes it from the signal
    - start again, until the signal changes by less than threshold
    (taking the norm of the difference of the signal, before and after)
    Return weights (sum of the scaling) and remaining of signal

    If codebook is normalized, set cbIsNormalized to True,
    it is slightly faster. (Normalized means each element has an
    euclidean norm of 1)
    """

    # to accumulate weights/scales
    weights = np.zeros([codebook.shape[0],1])

    # count iterations
    cntIter = 0
    # main loop
    while True:
        oldSignal = signal
        # do one iteration
        idxs, alphas, dists = encode_scale_oneiter(signal,codebook,cbIsNormalized)
        idx = idxs[0]
        alpha = alphas[idx]        
        # add to weights
        weights[idx] += alpha
        # remove what's explained by the codebook
        signal = signal - alpha * codebook[idx]
        # measure difference
        if euclidean_norm(oldSignal - signal) < thresh:
            break
        # check number of iterations
        cntIter += 1
        if cntIter >= maxIter:
            break

    return weights, signal


def encode_dataset_scale(data,codebook,thresh,cbIsNormalized=False):
    """
    Iteratively encode a whole dataset, where each row is a signal
    Data is a big numpy 2d array, number of signals = data.shape[0]
    Codebook is one code per row.
    Return:
         weights, 'orderedweights'
    orderedweights is a weird thing, we're interested in the shape
    of the encoding histogram. We add the weights after sorting them.
    Therefore, we don't care about which code did what, but how many
    codes were really used per signal.
    - Weights and ordered weights are normalized by number of signals
    - Weights are the sum of the weight absolute values per signal
    """
    # initialize
    cnt = 0
    weights = []
    orderedWeights = []
    # iterate over data
    for signal in data[:]:
        # check nan
        if np.isnan(signal).any():
            continue
        # counter
        cnt += 1
        # encode
        w,s = encode_scale(signal,codebook,thresh,cbIsNormalized)
        if cnt == 1:
            weights = np.abs(np.array(w).flatten())
            orderedWeights = np.abs(np.sort(np.array(w).flatten()))
        else:
            weights = weights + np.abs(np.array(w).flatten())
            orderedWeights = orderedWeights + np.abs(np.sort(np.array(w).flatten()))
    # normalize
    print 'encoding of data done, counter=' + str(cnt)
    weights = weights * 1. / cnt
    orderedWeights = orderedWeights * 1. / cnt
    # return
    return weights, orderedWeights




def online_encoding_learn(feats,K,lrate=1.,nIter=10,nEncode=-1,
                          thresh=0.0000001,maxRise=.05):
    """
    Learn a codebook by encoding (think matching pursuit)
    Input:
      - feats    data, one example per row
      - K        size of the codebook, or starting codebook
      - lrate    learning rate
      - nIter    number of iterations on the whole data
      - nEncode  number of encoding iteration, default=codebook size
      - maxRise, stop if, after an iteration, we are worst than (maxRise*100)%
                 of the best average distance ever obtained
    Return:
      - codebook
      - residual
    """
    
    # initialize codebook
    if type(K) == type(0):
        assert(feats.shape[0] >= K)
        fullrange = np.array(range(feats.shape[0]))
        np.random.shuffle(fullrange)
        start_codes_idx = fullrange[:K]
        codebook = feats[start_codes_idx,:]
        for code_idx in range(K):
            codebook[code_idx,:] = normalize(codebook[code_idx,:])
    # existing codebook
    else: 
        codebook = K
        K = codebook.shape[0]
    # initialize number of encoding iter
    if nEncode <= 0:
        nEncode = K

    prev_sum_residual = -1
    best_sum_residual = -1
    nFeats = feats.shape[0]
    # iterate over max iter
    for iteration in range(nIter):
        # start time
        tstart_iter = time.time()
        # sum the norm of the residual
        sum_residual = 0
        # iterate over features
        for pattern in feats[:]:
            # make sure no nan or empty pattern
            if np.isnan(pattern).any():
                continue
            if not (pattern>0).any():
                continue
            # encode
            weights,residual = encode_scale(pattern,codebook,1e-10,
                                            cbIsNormalized=True,maxIter=nEncode)
            # update codebook
            sumWeights = np.array(weights).sum()
            if abs(sumWeights) < 1e-10 :
                continue
            for cbIdx in range(K):
                if abs(weights[cbIdx]) < 1e-10:
                    continue
                codebook[cbIdx,:] += (residual / weights[cbIdx] - codebook[cbIdx,:]) * (weights[cbIdx] / sumWeights * lrate)
                codebook[cbIdx,:] = normalize(codebook[cbIdx,:])
            # add residual
            sum_residual += euclidean_norm(residual)

        # update best sum reesidual
        if best_sum_residual < 0 or best_sum_residual > sum_residual:
            best_sum_residual = sum_residual
        # verbose
        print 'iter '+str(iteration)+' done, avg. residual = ' + str(sum_residual * 1. / nFeats)+', iteration done in ' + str(time.time()-tstart_iter) + 'seconds.'
        # check threshold
        if prev_sum_residual >= 0:
            if abs((sum_residual - prev_sum_residual) * 1./nFeats) < thresh:
                print 'online_encode_learn stops because of thresholding after iter: ' + str(iteration+1)
                break
        if best_sum_residual * (1. + maxRise) < sum_residual:
            break
        prev_sum_residual = sum_residual

    # return codebook, plus recent residual norm sum
    return codebook, (sum_residual * 1. / nFeats)



def online_vq(feats,K,lrate,nIter=10,thresh=0.0000001,maxRise=.05,repulse=False):
    """
    Online vector quantization
    INPUT:
      - matrix of vectors, one feature per row, all equal length
      - K size of the codebook, or an existing codebook
      - lrate
      - max number of iteration, 1 iteration = whole data
      - maxRise, stop if, after an iteration, we are worst than (maxRise*100)%
                 of the best average distance ever obtained
    OUTPUT:
      - codebook (one code per row)
      - average distance between a feature and it's encoding
      - list of which code to use for which data sample
    Inspired by the algorithm here:
    http://en.wikipedia.org/w/index.php?title=Vector_quantization&oldid=343764861
    Codes are normalized, and can be scaled as to better feat a segment
    Codebook initialized using data points.
    Each vector of the codebook is normalized (euclidean norm = 1)
    Thresholding is based on the average distance between points and there
    encoding (as one code from the codebook).
    For efficiency, distance computed during the algorithm and before the
    modification of the codebook, so we break one iteration too late.
    """

    # initialize codebook
    if type(K) == type(0):
        assert(feats.shape[0] >= K)
        fullrange = np.array(range(feats.shape[0]))
        np.random.shuffle(fullrange)
        start_codes_idx = fullrange[:K]
        codebook = feats[start_codes_idx,:]
        for code_idx in range(K):
            codebook[code_idx,:] = normalize(codebook[code_idx,:])
    # existing codebook
    else: 
        codebook = K
        K = codebook.shape[0]

    # init (for thresholding)
    prev_sum_dist = -1
    nFeats = feats.shape[0]
    # keep the best result
    best_sum_dist = -1
    # know which code goes with each pattern
    best_code_per_pattern = np.ones([nFeats.shape[0],1])
    best_code_per_pattern *= -1
    # iterate over max iter
    for iteration in range(nIter):
        # start time
        tstart_iter = time.time()
        # sum of distance
        sum_distance = 0
        # iterate over features
        whichPattern = -1
        for pattern in feats[:]:
            # which pattern we're looking at
            whichPattern += 1
            # make sure no nan
            if np.isnan(pattern).any():
                continue
            if pattern.sum() == 0 :
                continue
            # find closest code
            idxs,weights,dists = encode_scale_oneiter(pattern,codebook,
                                                      cbIsNormalized=True)
            idx = idxs[0]
            weight = weights[idx]
            dist = dists[idx]
            # get that code closer by some learning rate
            codebook[idx,:] += (pattern / weight - codebook[idx,:]) * lrate
            codebook[idx,:] = normalize(codebook[idx,:])
            # remember that code for that pattern
            best_code_per_pattern[whichPattern] = idx

            ######################
            # TEST on repulsiveness
            if repulse:
                idx2 = idxs[1]
                weight2 = weights[1]
                codebook[idx2,:] -= (pattern / weight2 - codebook[idx2,:]) * lrate * (dists[idx] / dists[idx2])
                codebook[idx2,:] = normalize(codebook[idx2,:])
            #####################
            
            # add distance to sum
            sum_distance += dist
        # update best sum dist
        if best_sum_dist < 0 or best_sum_dist > sum_distance:
            best_sum_dist = sum_distance
        # verbose
        print 'iter '+str(iteration)+' done, avg. dist = ' + str(sum_distance * 1. / nFeats)+', iteration done in ' + str(time.time()-tstart_iter) + 'seconds.'
        # check threshold
        if prev_sum_dist >= 0:
            if (sum_distance - prev_sum_dist) * 1./nFeats > thresh:
                print 'online_vq stops because of thresholding after iter: ' + str(iteration+1)
                break
        if best_sum_dist * (1. + maxRise) < sum_distance:
            break
        prev_sum_dist = sum_distance
        
    # return codebook, average distance
    return codebook,(sum_distance * 1. / nFeats), best_code_per_pattern


