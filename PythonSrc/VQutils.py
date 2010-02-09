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
    alphas = [project_factor(signal,r,cbIsNormalized) for r in codebook[:]]
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

    If codebook is normalized, set cbIsNormalized to True,
    it is slightly faster. (Normalized means each element has an
    euclidean norm of 1)
    """

    while True:
        oldSignal = signal
        # do one iteration
        idx, alpha, dist = encode_scale_oneiter(signal,codebook,cbIsNormalized)
        # remove what's explained by the codebook
        signal = signal - alpha * codebook[idx]
        # measure difference
        if euclidean_norm(oldSignal - signal) < thresh:
            break
