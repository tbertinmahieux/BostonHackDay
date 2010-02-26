#!/net/python/bin/python

import numpy
import pylab
#from mytools.plotting import hist

######################################################################
#  Locally Linear Embedding
######################################################################

__doc__="""locally linear embedding code in python
written by Jake VanderPlas
University of Washington
http://www.astro.washington.edu/vanderplas
"""


USE_SVD = True

def dimensionality(M,k,v=0.9,quiet=True):
    M = numpy.asmatrix(M)
    d,N = M.shape
    assert k<N
    m_estimate = []
    var_total = 0
    for row in range(N):
        if row%500==0:print 'finished %s out of %s' % (row,N)
        #-----------------------------------------------
        #  find k nearest neighbors
        #-----------------------------------------------
        M_Mi = numpy.asarray(M-M[:,row])
        vec = (M_Mi**2).sum(0)
        nbrs = numpy.argsort(vec)[1:k+1]
        
        #compute distances
        x = numpy.asmatrix(M[:,nbrs] - M[:,row])
        #singular values of x give the variance:
        # use this to compute intrinsic dimensionality
        sig2 = (numpy.linalg.svd(x,compute_uv=0))**2

        #sig2 is sorted from large to small
        
        #use sig2 to compute intrinsic dimensionality of the
        # data at this neighborhood.  The dimensionality is the
        # number of eigenvalues needed to sum to the total
        # desired variance
        sig2 /= sig2.sum()
        S = sig2.cumsum()
        m = S.searchsorted(v)
        if m>0:
            m += ( (v-S[m-1])/sig2[m] )
        else:
            m = v/sig2[m]
        m_estimate.append(m)
        
        r = numpy.sum(sig2[m:])
        var_total += r

    if not quiet: print 'average variance conserved: %.3g' % (1.0 - var_total/N)

    return m_estimate


def LLE(M,k,m,quiet=False):
    """
    Perform a Locally Linear Embedding analysis on M
    
    >> LLE(M,k,d,quiet=False)
    
     - M is a numpy array of rank (d,N), consisting of N
        data points in d dimensions.

     - k is the number of neighbors to use in the embedding

     - m is the number of dimensions to which the dataset will
        be reduced.

    Based on the algorithm outlined in
     'An Introduction to Locally Linear Embedding'
        by L. Saul and S. Roewis

    Using imrovements suggested in
     'Locally Linear Embedding for Classification'
        by D. deRidder and R.P.W. Duin
    """
    M = numpy.asmatrix(M)
    d,N = M.shape

    assert k<N
    assert m<=d

    
    if not quiet:
        print 'performing LLE on %i points in %i dimensions...' % (N,d)

    #build the weight matrix
    W = numpy.zeros((N,N))

    if not quiet:
        print ' - constructing [%i x %i] weight matrix...' % W.shape

    m_estimate = []
    var_total = 0.0
    
    for row in range(N):
        #-----------------------------------------------
        #  find k nearest neighbors
        #-----------------------------------------------
        M_Mi = numpy.asarray(M-M[:,row])
        vec = (M_Mi**2).sum(0)
        nbrs = numpy.argsort(vec)[1:k+1]
        
        #-----------------------------------------------
        #  compute weight vector based on neighbors
        #-----------------------------------------------

        #compute [k x k] covariance matrix of distances
        M_Mi = numpy.asmatrix(M_Mi[:,nbrs])
        Q = M_Mi.T * M_Mi

        #singular values of M_Mi give the variance:
        # use this to compute intrinsic dimensionality
        sig2 = (numpy.linalg.svd(M_Mi,compute_uv=0))**2

        #use sig2 to compute intrinsic dimensionality of the
        # data at this neighborhood.  The dimensionality is the
        # number of eigenvalues needed to sum to the total
        # desired variance
        v=0.9
        sig2 /= sig2.sum()
        S = sig2.cumsum()
        m_est = S.searchsorted(v)
        if m_est>0:
            m_est += ( (v-S[m_est-1])/sig2[m_est] )
        else:
            m_est = v/sig2[m_est]
        m_estimate.append(m_est)
        
        #Covariance matrix may be nearly singular:
        # add a diagonal correction to prevent numerical errors
        # correction is equal to the sum of the (d-m) unused variances
        #  (as in deRidder & Duin)
        r = numpy.sum(sig2[m:])
        var_total += r
        Q += r*numpy.identity(Q.shape[0])
        #Note that Roewis et al instead uses "a correction that 
        #   is small compared to the trace":
        #r = 0.001 * float(Q.trace())
    
        #solve for weight
        w = numpy.linalg.solve(Q,numpy.ones((Q.shape[0],1)))[:,0]
        w /= numpy.sum(w)

        #update row of the weight matrix
        W[row,nbrs] = w

    if not quiet:
        print ' - finding [%i x %i] null space of weight matrix...' % (m,N)
    #to find the null space, we need the bottom d+1
    #  eigenvectors of (W-I).T*(W-I)
    #Compute this using the svd of (W-I):
    I = numpy.identity(W.shape[0])
    U,sig,VT = numpy.linalg.svd(W-I,full_matrices=0)
    indices = numpy.argsort(sig)[1:m+1]

    print 'm_estimate: %.2f +/- %.2f' % (numpy.median(m_estimate),numpy.std(m_estimate))
    print 'average variance conserved: %.3g' % (1.0 - var_total/N)
    
    return numpy.array(VT[indices,:])

def new_LLE_pts(M,M_LLE,k,x):
    """
    inputs:
       - M: a rank [d * N] data-matrix
       - M_LLE: a rank [m * N] matrixwhich is the output of LLE(M,k,m)
       - k: the number of neighbors used to produce M_LLE
       - x: a length d data vector OR a rank [d * Nx] array
    returns:
       - y: the LLE reconstruction of x
    """
    M = numpy.matrix(M)
    M_LLE = numpy.matrix(M_LLE)

    d,N = M.shape
    m,N2 = M_LLE.shape
    assert N==N2

    #make sure x is a column vector
    if numpy.rank(x) == 1:
        x = numpy.matrix(x).T
    else:
        x = numpy.matrix(x)
    assert x.shape[0] == d
    Nx = x.shape[1]

    W = numpy.matrix(numpy.zeros([Nx,N]))

    for i in range(x.shape[1]):
        #  find k nearest neighbors
        M_xi = numpy.array(M-x[:,i])
        vec = (M_xi**2).sum(0)
        nbrs = numpy.argsort(vec)[1:k+1]
        
        #compute covariance matrix of distances
        M_xi = numpy.matrix(M_xi[:,nbrs])
        Q = M_xi.T * M_xi

        #singular values of x give the variance:
        # use this to compute intrinsic dimensionality
        sig2 = (numpy.linalg.svd(M_xi,compute_uv=0))**2
    
        #Covariance matrix may be nearly singular:
        # add a diagonal correction to prevent numerical errors
        # correction is equal to the sum of the (d-m) unused variances
        #  (as in deRidder & Duin)
        r = numpy.sum(sig2[m:])
        Q += r*numpy.identity(Q.shape[0])
        #Note that Roewis et. al. instead uses "a correction that 
        #   is small compared to the trace":
        #r = 0.001 * float(Q.trace())
        
        #solve for weight
        w = numpy.linalg.solve(Q,numpy.ones((Q.shape[0],1)))[:,0]
        w /= numpy.sum(w)

        W[i,nbrs] = w
        print 'x[%i]: variance conserved: %.2f' % (i,1.0- sig2[m:].sum())

    #multiply weights by projections of neighbors to get y
    print M_LLE.shape
    print W.shape
    print len(nbrs)
    
    return numpy.array( M_LLE  * numpy.matrix(W).T )





######################################################################
#  Hessian Locally Linear Embedding
######################################################################

def HLLE(M,k,d,quiet=False):
    """
    Perform a Hessian Eigenmapping analysis on M

    >> HLLE(M,k,d,quiet=False)
    
     - M is a numpy array of rank (dim,N), consisting of N
        data points in dim dimensions.

     - k is the number of neighbors to use in the embedding

     - d is the number of dimensions to which the dataset will
        be reduced.
    
    Implementation based on algorithm outlined in
     'Hessian Eigenmaps: new locally linear embedding techniques
      for high-dimensional data'
        by C. Grimes and D. Donoho, March 2003
    """
    M = numpy.asmatrix(M)
    dim,N = M.shape
    
    if not quiet:
        print 'performing HLLE on %i points in %i dimensions...' % (N,dim)
    
    dp = d*(d+1)/2
    W = numpy.asmatrix( numpy.zeros([dp*N,N]) )
    
    if not quiet:
        print ' - constructing [%i x %i] weight matrix...' % W.shape
        
    for i in range(N):
        #-----------------------------------------------
        #  find nearest neighbors
        #-----------------------------------------------
        M_Mi = numpy.asarray(M-M[:,i])
        vec = sum(M_Mi*M_Mi,0)
        nbrs = numpy.argsort(vec)[1:k+1]

        #-----------------------------------------------
        #  center the neighborhood using the mean
        #-----------------------------------------------
        nbrhd = M[:,nbrs]
        nbrhd -= nbrhd.mean(1)

        #-----------------------------------------------
        #  compute local coordinates
        #   using a singular value decomposition
        #-----------------------------------------------
        U,vals,VT = numpy.linalg.svd(nbrhd,full_matrices=0)
        nbrhd = numpy.asmatrix( (VT.T)[:,:d] )

        #-----------------------------------------------
        #  build Hessian estimator
        #-----------------------------------------------
        ct = 0
        Yi = numpy.asmatrix(numpy.zeros([k,dp]))
        
        for mm in range(d):
            for nn in range(mm,d):
                Yi[:,ct] = numpy.multiply(nbrhd[:,mm],nbrhd[:,nn])
                ct += 1
        Yi = numpy.concatenate( [numpy.ones((k,1)), nbrhd, Yi],1 )

        #-----------------------------------------------
        #  orthogonalize linear and quadratic forms
        #   with QR factorization
        #  make the weights sum to 1
        #-----------------------------------------------
        Q,R = numpy.linalg.qr(Yi)
        w = numpy.asarray(Q[:,d+1:].T)
        S = w.sum(1) #sum along rows

        #if S[i] is too small, set it equal to 1.0
        S[numpy.where(numpy.abs(S)<0.0001)] = 1.0
        W[ i*dp:(i+1)*dp , nbrs ] = (w.T/S).T

    #-----------------------------------------------
    # To find the null space, we want the
    #  first d+1 eigenvectors of W.T*W
    # Compute this using an svd of W
    #-----------------------------------------------
    if not quiet:
        print ' - computing [%i x %i] null space of weight matrix...' % (d,N)

    #Fast, but memory intensive
    if USE_SVD:
        U,sig,VT = numpy.linalg.svd(W,full_matrices=0)
        del U
        indices = numpy.argsort(sig)[1:d+1]
        Y = VT[indices,:] * numpy.sqrt(N)

    #Slower, but uses less memory
    else:
        C = W.T*W
        del W
        sig2,V = numpy.linalg.eigh(C)
        del C
        indices = range(1,d+1) #sig2 is sorted in ascending order
        Y = V[:,indices].T * numpy.sqrt(N)

    #-----------------------------------------------
    # Normalize Y
    #  we need R = (Y.T*Y)^(-1/2)
    #   do this with an SVD of Y
    #      Y = U*sig*V.T
    #      Y.T*Y = (V*sig.T*U.T) * (U*sig*V.T)
    #            = U*(sig*sig.T)*U.T
    #   so
    #      R = V * sig^-1 * V.T
    #-----------------------------------------------
    if not quiet:
        print ' - normalizing null space via SVD...'

    #Fast, but memory intensive
    if USE_SVD:
        U,sig,VT = numpy.linalg.svd(Y,full_matrices=0)
        del U
        S = numpy.asmatrix(numpy.diag(sig**-1))
        R = VT.T * S * VT
        return numpy.asarray(Y*R)

    #Slower, but uses less memory
    else:
        C = Y*Y.T
        sig2,U = numpy.linalg.eigh(C)
        U = U[:,::-1] #eigenvectors should be in descending order
        sig2=sig2[::-1]
        S = numpy.asmatrix(numpy.zeros(U.shape))
        for i in range(d):
            S[i,i] = (1.0*sig2[i])**-1.5
        return numpy.asarray(C * U * S * U.T * Y)




######################################################################
#  Modified Gram-Schmidt
######################################################################

def mgs(A):
    """
    Modified Gram-Schmidt version of QR factorization

    returns matrices Q,R such that A = Q*R
    where Q is an orthogonal matrix,
          R is an upper-right triangular matrix
    """
    #copy A and make sure it's a matrix
    Q = 1.0*numpy.asmatrix(A)
    m,n = Q.shape
    #assume m>=n
    R = numpy.asmatrix(numpy.zeros([n,n]))
    for i in range(n):
        v = Q[:,i]
        R[i,i] = numpy.sqrt(numpy.sum(numpy.multiply(v,v)))
        Q[:,i] /= R[i,i]
        for j in range(i+1,n):
            R[i,j] = Q[:,i].T * Q[:,j]
            Q[:,j] -= R[i,j] * Q[:,i]

    return Q,R
