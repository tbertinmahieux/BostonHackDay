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
import scipy.spatial
import scipy.spatial.distance as DIST



def euclidean_dist(a,b):
    """ typical euclidean distance """
    return DIST.euclidean(a,b)

def euclidean_norm(a):
    """ regular euclidean norm of a numpy vector """
    return np.sqrt((a*a).sum())

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
    if not v_to_normalized:
        # we must normalized by norm^2
        normTo = euclidean_norm(v_to)
        return np.dot(v_from,v_to) / (normTo * normTo)
    else:
        # trivial case
        return np.dot(v_from,v_to)


def encode_scale_oneiter(signal,codebook,cbIsNormalized=False):
    """
    Find the best pattern in the codebook, returns the distance.
    codebook is one pattern per row.
    We allow the pattern to be scaled to be as close as possible
    to the signal
    Returns <index of the pattern>,<scaling>,<distance>
    If codebook is normalized, set cbIsNormalized to True,
    it is slightly faster. (Normalized means each element has an
    euclidean norm of 1)
    """
    # find the right scaling
    alphas = [projection_factor(signal,r,cbIsNormalized) for r in codebook[:]]
    alphas = np.array(alphas).reshape(codebook.shape[0],1)
    # scale the codebook and compute the distance
    dists = [euclidean_dist(signal,r) for r in (alphas*codebook)[:]]
    # return the index, scaling, and distance for the MIN DISTANCE
    idx = np.argmin(dists)
    return idx,alphas[idx][0],dists[idx]


def encode_scale(signal,codebook,thresh,cbIsNormalized=False):
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
    
    # main loop
    while True:
        oldSignal = signal
        # do one iteration
        idx, alpha, dist = encode_scale_oneiter(signal,codebook,cbIsNormalized)
        # add to weights
        weights[idx] += alpha
        # remove what's explained by the codebook
        signal = signal - alpha * codebook[idx]
        # measure difference
        if euclidean_norm(oldSignal - signal) < thresh:
            break

    return weights, signal



def online_vq(feats,K,lrate,nIter=10,thresh=0.0000001):
    """
    Online vector quantization
    INPUT:
      - matrix of vectors, one feature per row, all equal length
      - K size of the codebook
      - lrate
      - max number of iteration, 1 iteration = whole data
    OUTPUT:
      - codebook (one code per row)
      - average distance between a feature and it's encoding
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
    assert(feats.shape[0] >= K)
    fullrange = np.array(range(feats.shape[0]))
    np.shuffle(fullrange)
    start_codes_idx = fullrange[:K]
    codebook = feats[start_codes_idx,:]
    for code_idx in range(K):
        codebook[code_idx,:] = normalize(codebook[code_idx,:])

    # init (for thresholding)
    prev_sum_dist = -1
    nFeats = feats.shape[0]
    # iterate over max iter
    for iteration in range(nIter):
        # sum of distance
        sum_distance = 0
        # iterate over features
        for pattern in feats[:]:
            # find closest code
            idx,weight,dist = encode_scale_oneiter(pattern,codebook,
                                                   cbIsNormalized=True)
            # get that code closer by some learning rate
            codebook[idx,:] += (pattern - (weight * codebook[idx,:])) * lrate
            # add distance to sum
            sum_distance += dist
        # check threshold
        if prev_sum_dist >= 0:
            if abs((prev_sum_dist - sum_distance) * 1./nFeats) < thresh:
                break
        prev_sum_dist = sum_distance
            
    # return codebook, average distance
    return codebook,(sum_distance * 1. / nFeats)


        
