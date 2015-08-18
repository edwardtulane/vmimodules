
# coding: utf-8

# In[1]:

import numpy as np, scipy as sp
import pandas as pd, seaborn as sb
import scipy.ndimage as im



def comp_hist(img, c, s):
    """
    Applies sigmoidal histogram compression to 'img'.

    Parameters:
    c, s = centre and strength of the compression

    Returns: Compressed image.
    """

    den = np.exp(-s * (img - c)) + 1
    return 1. / den


def otsu(t, dist):
    """
    Object function for finding Otsu's threshold.

    Parameters:
    t, dist = threshold, graylevels of the image

    Returns: Interclass variance.
    """
    
    P, Q = dist[dist < t], dist[dist >= t]
    fun = P.sum() * P.std() ** 2 + Q.sum() * Q.std() ** 2
    return fun


def find_otsus_thr(img):
    """
    Optimisation routine that attempts to find Otsu's threshold.

    Parameters: Image 'img'

    Return: Threshold 't' if the optimisation was successful.
    """
    
    img = img.ravel()
    res = sp.optimize.minimize(otsu, 0.5, args=(img), method='Nelder-Mead')
    
    if (res.success & (res.x > 0.) & (res.x < 1.)):
        return res.x
    else:
        raise Exception("Otsu's thresholding failed!")



def seg_lev(img, no_lev):
    """
    Linear inclusive graylevel segmentation of the image 'img'.

    Returns: 3D-array of the segments ordered from the highest threshold to the lowest.
    """

    img_seg = np.zeros([no_lev] +  list(img.shape), dtype=np.bool_)
    levels = np.linspace(0, 1, no_lev + 1)
    levels = levels[::-1]
    for i in range(no_lev):
        slc = img_seg[i]
        slc[(img >= levels[i])] = True
    return img_seg



def detect_hits_patch(img, no_levels):
    """
    Inner peak finding loop for the isolated patches.

    Returns: Dictionary with the local hit indices.
    """

    hits = dict()
    dim = list(img.shape)
    seeds = list()
    
    seg = seg_lev(img, no_levels)
    null_map = np.zeros(dim, dtype=np.bool_)

    for img in seg:
        lb, cn = im.label(img)
        hitmap = np.zeros([cn] +  dim, dtype=np.bool_)
        obs = im.find_objects(lb, cn)

        for i, o in enumerate(obs):
            slc = hitmap[i]
            slc[o] = True
            crit = np.array([np.all(np.logical_and(slc, s) == s) 
                             for s in seeds])
            cnect = np.argwhere(crit).flatten()

            if crit.sum() == 0:
                hits[len(seeds)] = np.where(lb == i + 1)
                new_seed = null_map.copy()
                new_seed[hits[len(seeds)]] = True
                seeds.append(new_seed)
                
            elif crit.sum() == 1:
                hits[cnect[0]] = np.where(lb == i + 1)

    return hits

def detect_hits_img(img, comp_cntr, comp_strgth, no_levels, imax=0, dilate=False):
    """
    Outer peak finding loop for an entire VMI image.

    Parameters:
    cntr, strgth = Histogram compression
    no_levels, imax = segmentation levels, dynamic range reference

    Returns:
    hits_sgl, hits_mlt = Dataframes for single and multi-level thresholding results
    otsu_t = Otsu's threshold used for binary segmentation
    mask = false positive detection based on duplicate and very close hits
    """


    hit_col = ['y_cntr', 'x_cntr']
    sgl_cols = ['x_wid', 'y_wid', 'count']
    prop_cols = ['area', 'volume', 'x_gau', 'y_gau', 'discriminant', 'rank', 'cond_no']

    struct = im.generate_binary_structure(2, 1)

    # Vandermonde exponents
    y_exp = np.array([0,1,2,0,0])
    x_exp = np.array([0,0,0,1,2])

    img_wrk = img.astype(np.float_)
 
    if np.any(imax):
        assert img.shape == imax.shape
        
        ma = np.zeros(img.shape, dtype=np.bool_)
        ma = np.where(imax > np.percentile(imax, 10), True, False)
        img_wrk[ma] = img_wrk[ma] / imax[ma]
        img_wrk[~ma] = 0.       

    else:
        img_wrk /= img.max()
        
    img_cmp = comp_hist(img_wrk, comp_cntr, comp_strgth)
    otsu_t = find_otsus_thr(img_cmp)
    
    det_glob = np.where(img_cmp > otsu_t, True, False)

    if dilate:
        det_glob = im.binary_dilation(det_glob, structure=struct)

    lab_gl, cnt_gl = im.label(det_glob)
    obs_cgl = sp.ndimage.find_objects(lab_gl, cnt_gl)
    
    hits_sgl = im.center_of_mass(img_wrk, lab_gl, np.arange(cnt_gl))
    hits_sgl = pd.DataFrame(hits_sgl, columns=hit_col)
    
    hits_mlt, prop_mlt = list(), list()
    prop_sgl = list()
    
    for obj in obs_cgl:
        y_pos, x_pos = obj[0].start, obj[1].start
        p = np.array([y_pos, x_pos])
        y_wid = obj[0].stop - obj[0].start
        x_wid = obj[1].stop - obj[1].start
 
        patch = img_wrk[obj]
   
        hits_lc = detect_hits_patch(patch, no_levels)
        prop_sgl.append( (x_wid, y_wid, len(hits_lc)) )
    
        for k, v in hits_lc.items():
            hit = patch[v]
            v = np.array(v)
            are, vol = hit.shape[0], hit.sum()

            # centre of mass centroiding
            hits_mlt.append( (np.sum(hit * v, axis=1) / vol) + p)

            # Gaussian least-squares centroiding
            van =  (v[0][:,None] ** y_exp) * (v[1][:,None] ** x_exp)
            z = np.log(hit)

            fit, res, rank, svd = np.linalg.lstsq(van, z)

            y_gau, x_gau = -fit[1] / (2 * fit[2]) + p[0], -fit[3] / (2 * fit[4]) + p[1]
            dscr, cond = (fit[4] * fit[2]) > 0, svd[0] / svd[-1]
            prop_mlt.append( (are, vol, x_gau, y_gau, dscr, rank, cond) )

            
    # Multi-level hits
    hits_mlt = pd.DataFrame(hits_mlt, columns=hit_col)
    prop_mlt = pd.DataFrame(prop_mlt, columns=prop_cols)
    hits_mlt = pd.concat([hits_mlt, prop_mlt], axis=1)
        
    # Single-level hits
    patches = pd.DataFrame(prop_sgl, columns=sgl_cols)

    # Distance matrix
    di = sp.spatial.distance.pdist(hits_mlt.icol([0,1]))
    di_sq = sp.spatial.distance.squareform(di)
    
    unus = np.ones_like(di_sq, dtype=np.bool_)
    trimask = np.triu(unus,k=1)
    
    # Filter on the tridiagonal distance matrix and duplicate detection
    di_sq[~trimask] = 100.
    sm = np.where(di_sq < 2.)
    
    dups = hits_mlt.duplicated()
    neigh = np.ones_like(dups, dtype=np.bool_)
    neigh[sm[1]] = False
    mask = neigh & ~dups

    return hits_sgl.join(patches), hits_mlt, otsu_t, mask


