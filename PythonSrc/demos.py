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
    featsNorm = [FU.normalize_pattern_maxenergy(p,pSize,keyInv,downBeatInv,retRoll=True) for p in data_iter]
    keyroll = np.array([x[1] for x in featsNorm])
    dbroll = np.array([x[2] for x in featsNorm])
    featsNorm = [x[0].flatten() for x in featsNorm]
    if len(featsNorm) == 0: # empty song
        return [],[],[],[],[]
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
    assert best_code_per_p.shape[0] > 0,'empty song, we should have caught that'
    encoding = codebook[best_code_per_p]
    # transform into 2 matrices, with derolling!!!!!!!!!
    assert(featsNorm.shape[0] == encoding.shape[0])
    #featsNormMAT = np.concatenate([x.reshape(12,pSize) for x in featsNorm],axis=1)
    featsNormMAT = np.concatenate([np.roll(np.roll(featsNorm[x].reshape(12,pSize),-keyroll[x],axis=0),-dbroll[x],axis=1) for x in range(featsNorm.shape[0])],axis=1)
    #encodingMAT = np.concatenate([x.reshape(12,pSize) for x in encoding],axis=1)
    encodingMAT = np.concatenate([np.roll(np.roll(encoding[x].reshape(12,pSize),-keyroll[x],axis=0),-dbroll[x],axis=1) for x in range(featsNorm.shape[0])],axis=1)
    # return
    return best_code_per_p,featsNorm,encoding,featsNormMAT,encodingMAT


def get_codeword_histogram(codewords, ncodewords):
    """
    Needs documentation, what is codeworks? best_code_per_pattern....
    """
    hist = np.zeros(ncodewords)
    for cw in codewords:
        hist[cw] += 1
    return hist



def merge_codebook(codebook,nGoal,freqs = []):
    """
    merge the codebook in an iterative and greedy way.
    Algo:
      - finds closest pair of codes
      - merge them, using freqs if available
      - repeat until desired number of codes (nGoal)
    Returns smaller codebook, #codes=nGoal
    Also returns frequencies of the new codebook
    Code not optimized!!!!!! close to n^3 operations
    """
    import numpy as np
    import VQutils as VQU
    import copy
    # set freqs, sanity checks
    if freqs == []:
        freqs = np.ones(codebook.shape[0])
    freqs = np.array(freqs)
    assert(freqs.size == codebook.shape[0])
    assert(nGoal < codebook.shape[0])
    assert(nGoal > 0)
    # let's go!
    cb = copy.deepcopy(codebook)
    for k in range(codebook.shape[0] - nGoal):
        # compute dists for all pairs
        dists = np.zeros([cb.shape[0],cb.shape[0]])
        for l in range(dists.shape[0]):
            dists[l,l] = np.inf
            for c in range(l+1,dists.shape[1]):
                dists[l,c] = VQU.euclidean_dist(cb[l],cb[c])
                dists[c,l] = np.inf
        # find closest pair
        pos = np.where(dists==dists.min())
        code1 = pos[0][0]
        code2 = pos[1][0]
        print 'iter',k,' min distance=',dists.min(),' codes=',code1,',',code2
        assert(code1 < code2)#code1 should be smaller from how we filled dists
        # merge
        #cb[code1,:] = np.mean([cb[code1,:]*freqs[code1],cb[code2,:]*freqs[code2]],axis=0) * 1. / (freqs[code1] + freqs[code2])
        cb[code1,:] = np.mean([cb[code1,:],cb[code2,:]],axis=0)
        freqs[code1] += freqs[code2]
        # remove
        if code2 + 1 < cb.shape[0]:
            cb[code2,:] = cb[-1,:]
            freqs[code2] = freqs[-1]
        cb = cb[:-1]
        freqs = freqs[:-1]
    # done
    return cb, freqs

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




def add_image(P,im,x,y,width=.15,height=0,gray=True):
    """
    Used by LLE_my_codebook to add a specific image to a given plot
    I should start my plotting library... or still Ron's...

    Image is a matrix, (x,y) is in data coord, image is centered

    INPUT:
      - P      pylab object (P in LLE_my_codebook)
      - im     matrix representing the image
      - x      x position in data coord
      - y      y position in data coord
      - width  width in fig size, height automatically if height=0
      - height height in fig size, if 0 based on width
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
    if height == 0:
        height = im.shape[0] * 1. / im.shape[1] * width
    # create new axis
    a = P.axes([relX-width/2.,relY-height/2.,width,height])
    # set to x and y ticks
    P.setp(a, xticks=[], yticks=[])
    # plot image, new axes is the current default
    if gray:
        P.imshow(im,interpolation='nearest',origin='lower',cmap=P.cm.gray_r)
    else:
        P.imshow(im,interpolation='nearest',origin='lower')
    # set back axes
    P.axes(curr_axes)



def LLE_my_codebook(codebook,nNeighbors=5,nRand=5):
    """
    Performs LLE on the codebook
    Display the result
    LLE code not mine, see code for reference.
    nRand=number of random images added
    """
    import pylab as P
    import LLE
    import numpy as np
    import VQutils as VQU
    # compute LLE, goal is 2D
    LLEres = LLE.LLE(codebook.T,nNeighbors,2)
    # plot that result
    P.plot(LLEres[0,:],LLEres[1,:],'.')
    P.hold(True)
    # prepare to plot
    patch_size = codebook[0,:].size / 12
    # add random
    for k in range(nRand):
        idx = np.random.randint(LLEres.shape[1])
        add_image(P,codebook[idx,:].reshape(12,patch_size),LLEres[0,idx],LLEres[1,idx],.08)
    # plot extreme left codebook
    idx = np.argmin(LLEres[0,:])
    add_image(P,codebook[idx,:].reshape(12,patch_size),LLEres[0,idx],LLEres[1,idx])
    # plot extreme right codebook
    idx = np.argmax(LLEres[0,:])
    add_image(P,codebook[idx,:].reshape(12,patch_size),LLEres[0,idx],LLEres[1,idx])
    # plot extreme up codebook
    idx = np.argmax(LLEres[1,:])
    add_image(P,codebook[idx,:].reshape(12,patch_size),LLEres[0,idx],LLEres[1,idx])
    # plot extreme down codebook
    idx = np.argmin(LLEres[1,:])
    add_image(P,codebook[idx,:].reshape(12,patch_size),LLEres[0,idx],LLEres[1,idx])
    # plot middle codebook
    idx = np.argmin([VQU.euclidean_dist(r,np.zeros(2)) for r in LLEres.T])
    add_image(P,codebook[idx,:].reshape(12,patch_size),LLEres[0,idx],LLEres[1,idx])
    # done, release, show
    P.hold(False)
    P.show()




def LLE_my_codebook2(codebook,nLines=5,nCols=15,nNeighbors=5):
    """
    Cut the 2d space in squares (nLines and nCols) and plot one random
    patter inside each, if there are any
    """
    import pylab as P
    import LLE
    import numpy as np
    import VQutils as VQU
    # compute LLE, goal is 2D
    LLEres = LLE.LLE(codebook.T,nNeighbors,2)
    # plot that result
    P.subplot(2,1,1)
    P.plot(LLEres[0,:],LLEres[1,:],'.')
    P.subplot(2,1,2)
    P.plot(LLEres[0,:],LLEres[1,:],'.') # should get covered up
    P.hold(True)
    # needed later to plot
    patch_size = codebook[0,:].size / 12
    # cut the space
    minx = LLEres[0,:].min()
    maxx = LLEres[0,:].max()+1e-14
    miny = LLEres[1,:].min()
    maxy = LLEres[1,:].max()+1e-14
    # iterate over lines
    for l in range(nLines):
        # find all points that could fit in that 'line'
        s1 = set(np.where(LLEres[1,:]>=miny+(maxy-miny)*l*1./nLines)[0])
        s2 = set(np.where(LLEres[1,:]<miny+(maxy-miny)*(l+1.)/nLines)[0])
        pts_in_line = s1.intersection(s2)
        if len(pts_in_line) == 0:
            continue
        # iterate over cols
        for c in range(nCols):
            # find pts in pts_in_line that could fit that col
            pts_in_square = set()
            for k in pts_in_line:
                if LLEres[0,k]>= minx+(maxx-minx)*c*1./nCols:
                    if LLEres[0,k]< minx+(maxx-minx)*(c+1.)/nCols:
                        pts_in_square.add(k)
            # choose random one
            if len(pts_in_square) == 0:
                continue
            pts_in_square = list(pts_in_square)
            idx = pts_in_square[np.random.randint(len(pts_in_square))]
            # plot it
            posx = minx+(maxx-minx)*(c+.5)/nCols
            posy = miny+(maxy-miny)*(l+.5)/nLines
            add_image(P,codebook[idx,:].reshape(12,patch_size),posx,posy,width=(maxx-minx)*1./nCols,height=(maxy-miny)*1./nLines)

    # done, release, show
    P.hold(False)
    P.show()

            

def freqs_for_my_artists(filenames,codebook,pSize=8,keyInv=True,
                         downBeatInv=False,bars=2):
    """
    Creates a dictionnary artist -> frequency
    Therefore, we know which codes were mostly used.
    Dictionnary not normalized

    filenames are expected to be: */artist/album/*.mat
    """
    import numpy as np
    import os

    res = {}
    nCodes = codebook.shape[0]
    # iterate over songs
    for f in filenames:
        # get artist ( sure there's a better split, like ...split(f)[-3] )
        tmp, song = os.path.split(f)
        tmp,album = os.path.split(tmp)
        tmp,artist = os.path.split(tmp)
        # encode song
        a,b,c,d,e = encode_one_song(f,codebook,pSize=pSize,keyInv=keyInv,
                                    downBeatInv=downBeatInv,bars=bars)
        best_code_per_p,featsNorm,encoding,featsNormMAT,encodingMAT = a,b,c,d,e
        # add it to freq
        if not res.has_key(artist):
            res[artist] = np.zeros([1,nCodes])
        for code in best_code_per_p:
            res[artist][0,int(code)] += 1
    # done, return dictionary
    return res
        
