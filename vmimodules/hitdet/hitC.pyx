cimport cython
import numpy as np
cimport numpy as np

from libc.math cimport exp
from scipy.linalg.cython_blas cimport dger

cdef extern from "math.h":
    double exp(double x)
    int round(double x)
    int abs(int abs)
    double abs( double abs)

cdef double pi = 1 / np.pi


@cython.boundscheck(False)
@cython.wraparound(False)
def gauss2dC(pars, int dim, norm=True,
                 double sig=1):
    """
    Maps out gaussian peaks on an image. Expects them to be centred.

    Parameters:
    pars = Dataframe with x_gau and y_gau columns
    dim = Image size (dim, dim)
    norm = Whether to normalise the peaks to unit volume. If False, all 
           gaussian parameters are required.
    sig = Width of the normalised peaks
    """
    cdef:
        int i, j, jj
        int ystart, ystop, yrng, xstart, xstop, xrng
        int inc = 1
        double qmax
        double cntr = (dim - 1) / 2

        double x, y, xci, yci
        double[:] xc = pars.x_gau.values
        double[:] yc = pars.y_gau.values
        int peakno = xc.shape[0]
        
        double[:] xvec = np.zeros(dim)
        double[:] yvec = np.zeros(dim)
        double[::1, :] img = np.zeros([dim, dim], order='F')
        
    sig  = <double> 1 / (2 * sig * sig)
    qmax = <double> pi * sig
    
    for i in range(peakno):
        xci = xc[i] + cntr
        yci = yc[i] + cntr
        
        ystart = <int> max(round(yci) - 40, 0)
        ystop  = <int> min(round(yci) + 40, dim - 1)
        yrng   = <int> max(ystop - ystart, 0)

        xstart = <int> max(round(xci) - 40, 0)
        xstop  = <int> min(round(xci) + 40, dim - 1)
        xrng   = <int> max(xstop - xstart, 0)
        
        for j in range(yrng):
            jj = <int> j + ystart
            y = <double> jj
            y = y - yci
            y = y * y 
            y = y * sig
            yvec[j] = exp(-y)
            
        for j in range(xrng):
            jj = <int> j + xstart          
            x = <double> jj
            x = x - xci
            x = x * x 
            x = x * sig
            xvec[j] = exp(-x)
            
        dger(&yrng, &xrng, &qmax, 
             &yvec[0], &inc, &xvec[0], &inc, 
             &img[ystart, xstart], &dim)  

    return np.asarray(img, order='C'
                     )

@cython.boundscheck(False)
@cython.wraparound(False)
def gaussquadC(pars, int dim, double sig=1):
    """
    Maps out normalised gaussian peaks on a single quadrant. Expects them to be centred.

    Parameters:
    pars = Dataframe with x_gau and y_gau columns
    dim = Quadrant size (dim, dim)
    sig = Width of the normalised peaks
    """
    cdef:
        int i, j, jj
        int ystart, ystop, yrng, xstart, xstop, xrng
        int inc = 1
        double qmax
        double cntr = (dim - 1) / 2

        double x, y, xci, yci
        double[:] xc = pars.x_gau.values
        double[:] yc = pars.y_gau.values
        int peakno = xc.shape[0]
        
        double[:] xvec = np.zeros(dim)
        double[:] yvec = np.zeros(dim)
        double[::1, :] img = np.zeros([dim, dim], order='F')
        
    sig  = <double> 1 / (2 * sig * sig)
    qmax = <double> pi * sig
    
    for i in range(peakno):
        xci = <double> abs(xc[i])
        yci = <double> abs(yc[i])
        
        ystart = <int> max(round(yci) - 40, 0)
        ystop  = <int> min(round(yci) + 40, dim)
        yrng   = <int> max(ystop - ystart, 0)

        xstart = <int> max(round(xci) - 40, 0)
        xstop  = <int> min(round(xci) + 40, dim)
        xrng   = <int> max(xstop - xstart, 0)
        
        for j in range(yrng):
            jj = <int> j + ystart
            y = <double> jj
            y = y - yci
            y = y * y 
            y = y * sig
            yvec[j] = exp(-y)
            
        for j in range(xrng):
            jj = <int> j + xstart          
            x = <double> jj
            x = x - xci
            x = x * x 
            x = x * sig
            xvec[j] = exp(-x)
            
        dger(&yrng, &xrng, &qmax, 
             &yvec[0], &inc, &xvec[0], &inc, 
             &img[ystart, xstart], &dim)  

    return np.asarray(img, order='C'
                     )
    
#===============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
def pixC(pars, int dim):

    cdef:
        double[:,:] gss = np.zeros((dim, dim))
        int cntr = (dim - 1) / 2
        double[:] yc = pars.y_gau.values, xc = pars.x_gau.values
        int length = yc.shape[0]
        unsigned int i
    
    for i in range(length):
        y = <int> round(yc[i] + cntr)
        x = <int> round(xc[i] + cntr)
        gss[y,x] = gss[y,x] + 1

    return np.asarray(gss, order='C')

@cython.boundscheck(False)
@cython.wraparound(False)
def quadC(pars, int dim):

    cdef:
        double[:,:] gss = np.zeros((dim, dim))
        double[:] yc = pars.y_gau.values, xc = pars.x_gau.values
        int length = yc.shape[0]
        unsigned int i
    
    for i in range(length):
        y = <int> abs(round(yc[i]))
        x = <int> abs(round(xc[i]))
        gss[y,x] = gss[y,x] + 1

    return np.asarray(gss)
