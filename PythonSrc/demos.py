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



def freqs_my_songs(filenames,codebook,pSize=8,keyInv=True,
                   downBeatInv=False,bars=2,normalize=False):
    """
    Returns a list of numpy.array containing frequency for each
    code in the codebook for each file in filenames
    """
    import numpy as np
    import VQutils as VQU
    res = []
    nCodes = codebook.shape[0]
    for f in filenames:
        # encode song
        a,b,c,d,e = encode_one_song(f,codebook,pSize=pSize,keyInv=keyInv,
                                    downBeatInv=downBeatInv,bars=bars)
        best_code_per_p,featsNorm,encoding,featsNormMAT,encodingMAT = a,b,c,d,e
        # get freqs
        freqs = np.zeros([1,nCodes])
        for code in best_code_per_p:
            freqs[0,int(code)] += 1
        if normalize and len(best_code_per_p) > 0:
            freqs *= 1./ VQU.euclidean_norm(freqs)
        res.append(freqs)
    # done, return res
    return res




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
        



def l0_dist(a,b):
    """
    Compute the number of common non nul elems
    returns (size - nCommon elems) so it is a distance (smaller=closer)
    """
    import numpy as np
    non_nul_a = np.where(a.flatten()>0)[0]
    nCommon = np.where(b.flatten()[non_nul_a]>0)[0].shape[0]
    return a.flatten().shape[0] - nCommon

def knn_from_freqs_on_artists(filenames,codebook,pSize=8,keyInv=True,
                              downBeatInv=False,bars=2,normalize=True,
                              confMatrix=True,use_l0_dist=False,use_artists=False):
    """
    Performs a leave-one-out experiments where we try to guess the artist
    from it's nearest neighbors in frequencies
    We use squared euclidean distance.

    filenames are expected to be: */artist/album/*.mat
    if confMatrix=True, plot it.
    if use_artists, song are matched to artist, not other songs

    RETURNS:
    - confusion matrix
    - freqs per file
    - artist per file
    """
    import numpy as np
    import os
    import VQutils as VQU
    import time
    import copy

    nCodes = codebook.shape[0]
    # get frequencies for all songs
    tstart = time.time()
    freqs = freqs_my_songs(filenames,codebook,pSize=pSize,keyInv=keyInv,
                           downBeatInv=downBeatInv,bars=bars,
                           normalize=normalize)
    print 'all frequencies computed in',(time.time()-tstart),'seconds.'
    # get artists for all songs
    artists = []
    for f in filenames:
        tmp, song = os.path.split(f)
        tmp,album = os.path.split(tmp)
        tmp,artist = os.path.split(tmp)
        artists.append(artist)
    artists = np.array(artists)
    # names of artists
    artist_names = np.unique(np.sort(artists))
    nArtists = artist_names.shape[0]
    # sanity check
    assert(len(filenames)==len(artists))
    # compute distance between all songs
    nFiles = len(filenames)
    tstart = time.time()
    if not use_artists:
        dists = np.zeros([nFiles,nFiles])
        for l in range(nFiles):
            for c in range(l+1,nFiles):
                if len(freqs[l])==0 or len(freqs[c])==0:
                    dists[l,c] = np.inf
                    dists[c,l] = np.inf
                    continue
                if use_l0_dist:
                    dists[l,c] = l0_dist(freqs[l],freqs[c])
                else:
                    dists[l,c] = VQU.euclidean_dist(freqs[l],freqs[c])
                    dists[c,l] = dists[l,c]
        for l in range(nFiles): # fill diag with inf
            dists[l,l] = np.inf
    else:
        # create a matrix songs * nArtists
        dists = np.zeros([nFiles,nArtists])
        # precompute cntArtists and artistFreqs, not normalized
        cntArtists = {}
        artistFreqs = {}
        for k in artist_names:
            cntArtists[k] = 0
            artistFreqs[k] = np.zeros([1,nCodes])
        for k in range(artists.shape[0]):
            art = artists[k]
            cntArtists[art] += 1
            artistFreqs[art] += freqs[k]
        # iterate over files
        for l in range(nFiles):
            currArtist = artists[l]
            currCntArtists = copy.deepcopy(cntArtists)
            currCntArtists[currArtist] -= 1
            currArtistFreqs = copy.deepcopy(artistFreqs)
            currArtistFreqs[currArtist] -= freqs[l]
            for k in currArtistFreqs.keys(): # normalize
                currArtistFreqs[k] *= 1. / currCntArtists[k]
            # fill in the line in dists
            for c in range(nArtists):
                art = artist_names[c]
                if use_l0_dist:
                    dists[l,c] = l0_dist(freqs[l],currArtistFreqs[art])
                else:
                    dists[l,c] = VQU.euclidean_dist(freqs[l],currArtistFreqs[art])
    print 'distances computed in',(time.time()-tstart),'seconds.'
    # confusion matrix
    confMat = np.zeros([nArtists,nArtists])
    # performs leave-one-out KNN
    nExps = 0
    nGood = 0
    randScore = 0 # sums prob of having it right by luck, must divide by nExps
    for songid in range(nFiles):
        if len(freqs[songid]) == 0:
            continue
        # get close matches ordered, remove inf
        orderedMatches = np.argsort(dists[songid,:])
        orderedMatches[np.where(dists[1,orderedMatches] != np.inf)]
        # artist
        artist = artists[songid]
        nMatches = orderedMatches.shape[0]
        if use_artists:
            assert nMatches == nArtists
        # get stats
        nExps += 1
        if not use_artists:
            nGoodMatches = np.where(artists[orderedMatches]==artist)[0].shape[0]
            if nGoodMatches == 0:
                continue
            randScore += nGoodMatches * 1. / nMatches
            pred_artist = artists[orderedMatches[0]]
        else:
            randScore += 1. / nArtists
            pred_artist = artist_names[orderedMatches[0]]
        if pred_artist == artist:
            nGood += 1
        # fill confusion matrix
        real_artist_id =np.where(artist_names==artist)[0][0]
        pred_artist_id =np.where(artist_names==pred_artist)[0][0]
        print songid,') real artist:',artist,'id=',real_artist_id,', pred artist:',pred_artist,'id=',pred_artist_id
        confMat[real_artist_id,pred_artist_id] += 1
    # done, print out
    print 'nExps:',nExps
    print 'rand accuracy:',(randScore*1./nExps)
    print 'accuracy:',(nGood*1./nExps)
    # plot confusion matrix
    if confMatrix:
        short_names = np.array([x[:2] for x in artist_names])
        import pylab as P
        P.imshow(confMat,interpolation='nearest',cmap=P.cm.gray_r,
                 origin='lower')
        P.yticks(P.arange(artist_names.shape[0]),list(artist_names))
        P.xticks(P.arange(artist_names.shape[0]),list(short_names))
        P.title('confusion matrix (real/predicted)')
        P.ylabel('TRUE')
        P.xlabel('RECOG')
        P.colorbar()
    # return confusion matrix
    return confMat,freqs,artists



def test_align(filenames,codebook):
    """
    see test align one song
    """
    import numpy as np
    import time
    # results
    n_exp_done = 0
    n_0 = 0
    n_1 = 0
    n_2 = 0
    n_3 = 0
    # verbose
    counter = 0
    tstart = time.time()
    # iter on files
    for f in filenames:
        counter += 1
        # print for 10, 50 and 75 %
        if np.round(filenames==len(filenames)*.1) == counter:
            print '10% of the files done in',time.time()-tstart,'seconds.'
        if np.round(filenames==len(filenames)*.5) == counter:
            print '50% of the files done in',time.time()-tstart,'seconds.'
        if np.round(filenames==len(filenames)*.75) == counter:
            print '75% of the files done in',time.time()-tstart,'seconds.'
        # done printing

                   
        res = test_align_one_song(f,codebook)
        if res < 0:
            continue
        n_exp_done += 1
        if res == 0:
            n_0 += 1
        elif res == 1:
            n_1 += 1
        elif res == 2:
            n_2 += 1
        elif res == 3:
            n_3 += 1
        else:
            print 'weird result:',res
    # print results
    print 'number exp done:', n_exp_done
    if n_exp_done == 0:
        return
    print 'accuracy:',(n_0 * 1./n_exp_done),'%'
    print 'details:'
    print 'n_0:',(n_0 * 1./n_exp_done),'%'
    print 'n_1:',(n_1 * 1./n_exp_done),'%'
    print 'n_2:',(n_2 * 1./n_exp_done),'%'
    print 'n_3:',(n_3 * 1./n_exp_done),'%'
    

def test_align_one_song(filename,codebook):
    """
    Experiment on how good can we find the alignment of a song
    Designed for a codebook of pSize=4, bars=1
    If song has non 4 beats patterns, problem

    Return is complex:
      - -1      if could not perform test
      - 0       if test succesful
      - 1-2-3   by how many beats we missed
    """

    import scipy
    import scipy.io
    import numpy as np
    import feats_utils as FU
    import VQutils as VQU

    mat = mat = scipy.io.loadmat(filename)
    btstart = mat['btstart']
    barstart = mat['barstart']
    try:
        if btstart.shape[1] < 3 or barstart.shape[1] < 3:
            return -1 # can not complete
    except IndexError:
        return -1 # can not complete
    except AttributeError:
        return -1 # can not complete
    # find bar start based on beat index
    barstart_idx = [np.where(btstart==x)[1][0] for x in barstart.flatten()]
    barstart_idx.append(btstart.shape[1])
    # find bar lengths
    barlengths = np.diff(barstart_idx)
    # find not4 elems
    not4 = np.where(barlengths!=4)[0]
    # find longest sequence of bars of length 4 beats
    seqs_of_4 = np.diff(not4)
    longest_seq_length = np.max(seqs_of_4) -1
    if longest_seq_length < 10: # why 10? bof....
        return -1 # can not complete
    # find best seq pos
    pos1 = not4[np.argmax(seqs_of_4)]+1
    pos2 = not4[np.argmax(seqs_of_4)+1]
    # longest sequence should be in range(pos1,pos2)
    # sanity checks
    assert pos2 - pos1 == longest_seq_length
    for k in range(pos1,pos2):
        assert barlengths[k] == 4
    # position in beats
    beat_pos_1 = barstart_idx[pos1]
    beat_pos_2 = beat_pos_1 + 4 * longest_seq_length
    assert beat_pos_2 == btstart.shape[1] or np.where(barstart_idx==beat_pos_2)[0].shape[0]>0
    # load actual beat features
    btchroma = mat['btchroma']
    # try everything: offset 0 to 3
    best_offset = -1
    best_avg_dist = np.inf
    for offset in range(4):
        avg_dist = 0
        for baridx in range(longest_seq_length-1):
            pos = beat_pos_1 + offset + baridx * 4
            feats = btchroma[:,pos:pos+4]
            featsNorm = FU.normalize_pattern_maxenergy(feats,newsize=4,
                                                       keyinvariant=True,
                                                       downbeatinvariant=False)
            # measure with codebook
            tmp,dists = VQU.encode_oneiter(featsNorm.flatten(),codebook)
            avg_dist += (dists[0] * dists[0]) * 1. / featsNorm.size
        if best_avg_dist > avg_dist:
            best_avg_dist = avg_dist
            best_offset = offset
    # done, return offset, which is 0 if fine
    return best_offset
