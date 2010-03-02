#! /usr/bin/env python
"""
Set of functions to perform vector quantization,
meaning find a set of 'typical patterns', get them in
a codebook, then reencode new vectors

Relies less and less on the scipy.cluster.vq package
and scipy.spatial.distance

T. Bertin-Mahieux (2010) Columbia University
www.columbia.edu/~tb2332/
"""

import numpy as np
import scipy as sp
import time
import copy

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


def encode_oneiter(signal,codebook):
    """
    Find the best pattern in the codebook, returns the distance.
    codebook is one pattern per row.
    Codebook is not scaled
    Returns <indexes of the pattern>,<scalings>,<distances>
    Indexes are returned starting from the code which has the smallest
    distance. If you only care about this code, do:
    idx = idxs[0], dist = dists[idx]
    """
    # compute the distances
    dists = [euclidean_dist(signal,r) for r in codebook]
    # return the index, scaling, and distance for the MIN DISTANCE
    idxs = np.argsort(dists)
    return idxs,dists


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
        alphas = np.inner(signal,codebook).reshape(codebook.shape[0],1)
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
        codebook = copy.deepcopy(K)
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



def get_codes_ordering(best_code_per_p, nCodes):
    """
    Receives an array as long as the number of features, with
    the index of best code for each feature.
    Returns an ordered list of the code index (by number of times
    each code is used)
    Also returns frequencies
    This is a small util function, nothing more.
    Usage:
       res = get_codes_ordering(best_code_per_p,200)
       orderedCodebook = codebook[res]
    """
    freqs = np.zeros([nCodes,1])
    best_code_per_p_flat = best_code_per_p.flatten()
    for k in range(best_code_per_p.size):
        freqs[best_code_per_p_flat[k]] += 1
    return np.flipud(np.argsort(freqs.flatten())), freqs.flatten()


def find_best_code_per_pattern(dataset,codebook,scale=False):
    """
    Find the best code for each pattenr in the dataset.
    Dataset: one code per row (samething for codebook)
    If scale=True, we assume codebook normalized
    return a vector, length=dataset size, contains the index
    of the closest code for each pattern (or -1 if there was a problem)
    RETURN:
       - index of the best code per pattern
       - distance for pattern to the best code
       - average distance for pattern to the best code (per pixel)
    """
    best_code_per_p = np.ones([dataset.shape[0],1]) * -1
    p_dists = np.ones([dataset.shape[0],1]) * -1
    avg_p_dists = np.ones([dataset.shape[0],1]) * -1
    for k in range(dataset.shape[0]):
        pattern = dataset[k,:].reshape(1,dataset.shape[1])
        if scale:
            idxs,weights,dists = encode_scale_oneiter(pattern,codebook,
                                                      cbIsNormalized=True)
        else:
            idxs,dists = encode_oneiter(pattern,codebook)
        best_code_per_p[k] = idxs[0]
        p_dists[k] = dists[0]
        #avg_p_dists[k] = np.sum(np.abs(pattern - codebook[idxs[0]]))*1./pattern.size
        avg_p_dists[k] = np.average(np.abs(pattern - codebook[idxs[0]]))
    # done
    return best_code_per_p, p_dists, avg_p_dists


def add_image(P,im,x,y,width=.15):
    """
    Used by LLE_my_codebook to add a specific image to a given plot
    I should start my plotting library... or still Ron's...

    Image is a matrix, (x,y) is in data coord, image is centered

    INPUT:
      - P      pylab object (P in LLE_my_codebook)
      - im     matrix representing the image
      - x      x position in data coord
      - y      y position in data coord
      - width  width in fig size, height automatically found
    """
    # current axes
    curr_axes = P.gca()
    # current axis in data coord
    (minX,maxX,minY,maxY) = P.axis()
    # get placement of mainx axes in figure
    bbox = curr_axes.bbox._bbox
    fX = bbox.x0
    fY = bbox.y0
    fWidth = bbox.x1 - bbox.x0
    fHeight = bbox.y1 - bbox.y0
    # find pos relative to main axes
    relX = (x - minX) * 1. / (maxX-minX)
    relY = (y - minY) * 1. / (maxY-minY)
    # find pos relative to main figure
    relX = (relX * fWidth) + fX
    relY = (relY * fHeight) + fY
    # height relative to width
    height = im.shape[0] * 1. / im.shape[1] * width
    # create new axis
    a = P.axes([relX-width/2.,relY-height/2.,width,height])
    # set to x and y ticks
    P.setp(a, xticks=[], yticks=[])
    # plot image, new axes is the current default
    P.imshow(im,interpolation='nearest',origin='lower')
    # set back axes
    P.axes(curr_axes)



def LLE_my_codebook(codebook,nNeighbors=5):
    """
    Performs LLE on the codebook
    Display the result
    LLE code not mine, see code for reference.
    """
    import pylab as P
    import LLE
    # compute LLE, goal is 2D
    LLEres = LLE.LLE(codebook.T,nNeighbors,2)
    # plot that result
    P.plot(LLEres[0,:],LLEres[1,:],'.')
    P.hold(True)
    # prepare to plot
    patch_size = codebook[0,:].size / 12
    tx = P.gca().get_xaxis_transform()
    ty = P.gca().get_yaxis_transform()
    # plot extreme left codebook
    idx = np.argmin(LLEres[0,:])
    add_image(P,codebook[idx,:].reshape(12,patch_size),LLEres[0,idx],LLEres[1,idx])
    # plot extrem right codebook
    idx = np.argmax(LLEres[0,:])
    add_image(P,codebook[idx,:].reshape(12,patch_size),LLEres[0,idx],LLEres[1,idx])
    # plot extreme up codebook
    idx = np.argmax(LLEres[1,:])
    add_image(P,codebook[idx,:].reshape(12,patch_size),LLEres[0,idx],LLEres[1,idx])
    # plot extreme down codebook
    idx = np.argmin(LLEres[1,:])
    add_image(P,codebook[idx,:].reshape(12,patch_size),LLEres[0,idx],LLEres[1,idx])
    # plot middle codebook
    idx = np.argmin([euclidean_dist(r,np.zeros(2)) for r in LLEres.T])
    add_image(P,codebook[idx,:].reshape(12,patch_size),LLEres[0,idx],LLEres[1,idx])
    # done, release, show
    P.hold(False)
    P.show()



def online_vq(feats,K,lrate,nIter=10,thresh=0.0000001,maxRise=.05,scale=False,repulse=False):
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
        if scale:
            for code_idx in range(K):
                codebook[code_idx,:] = normalize(codebook[code_idx,:])
    # existing codebook
    else: 
        codebook = copy.deepcopy(K)
        K = codebook.shape[0]

    # init (for thresholding)
    prev_sum_dist = -1
    nFeats = feats.shape[0]
    # keep the best result
    best_sum_dist = -1
    # know which code goes with each pattern
    #best_code_per_pattern = np.ones([nFeats,1])
    #best_code_per_pattern *= -1
    # not scaled? artificial weights
    if not scale:
        weights = np.ones(K)
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
            if scale:
                idxs,weights,dists = encode_scale_oneiter(pattern,codebook,
                                                      cbIsNormalized=True)
            else:
                # no scaling, i.e. weights = 1
                idxs,dists = encode_oneiter(pattern,codebook)
            idx = idxs[0]
            weight = weights[idx]
            dist = dists[idx]
            # get that code closer by some learning rate
            codebook[idx,:] += (pattern / weight - codebook[idx,:]) * lrate
            if scale:
                codebook[idx,:] = normalize(codebook[idx,:])
            # remember that code for that pattern
            #best_code_per_pattern[whichPattern] = idx
            # bad idea, codes will change

            ######################
            # TEST on repulsiveness
            if repulse:
                idx2 = idxs[1]
                weight2 = weights[1]
                codebook[idx2,:] -= (pattern / weight2 - codebook[idx2,:]) * lrate * (dists[idx] / dists[idx2])
                if scale:
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
    return codebook,(sum_distance * 1. / nFeats)#, best_code_per_pattern


