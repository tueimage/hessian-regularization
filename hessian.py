from scipy.ndimage.filters import correlate

from gaussian_kernels import *


def hessian2d(img, sigma, scale_norm=True):

    if scale_norm:
        c = sigma**2
    else:
        c = 1

    def conv(_img, _w):
        return correlate(correlate(_img, _w[0][np.newaxis], mode='reflect'), _w[1][np.newaxis].T, mode='reflect')

    dxx = c * conv(img, ndgauss((sigma, sigma), n=(0, 2), separable=True))
    dyy = c * conv(img, ndgauss((sigma, sigma), n=(2, 0), separable=True))
    dxy = c * conv(img, ndgauss((sigma, sigma), n=(1, 1), separable=True))

    return dxx, dyy, dxy


def hessian2d_eig(dxx, dyy, dxy):

    c = np.sqrt((dxx - dyy)*(dxx - dyy) + 4*dxy*dxy)

    l1 = (dxx + dyy + c)/2
    l2 = (dxx + dyy - c)/2

    return l1, l2


def blobiness2d(img, sigma, alpha=0.5, gamma=5):
    '''
    The case in [1] where N = 2, M = 0.

    References:
    [1] Antiga, Luca. "Generalizing vesselness with respect to dimensionality and shape." The Insight Journal 3 (2007): 1-14.
    '''

    dxx, dyy, dxy = hessian2d(img, sigma)

    l1, l2 = hessian2d_eig(dxx, dyy, dxy)

    L = np.stack((l1, l2), axis=2)
    L_abs = np.abs(L)

    ra = np.min(L_abs, axis=2) / np.max(L_abs, axis=2)

    s = np.sqrt(np.sum(L_abs**2, axis=2))

    m0 = (1 - np.exp(-ra**2 / 2 / alpha**2))
    m0 *= (1 - np.exp(-s**2 / 2 / gamma**2))
    m0 *= (l1 < 0) * (l2 < 0)

    return m0


def vesselness2d(img, sigma, beta=0.5, gamma=5):
    '''
    The case in [1] where N = 2, M = 1.

    References:
    [1] Antiga, Luca. "Generalizing vesselness with respect to dimensionality and shape." The Insight Journal 3 (2007): 1-14.
    '''

    dxx, dyy, dxy = hessian2d(img, sigma)

    l1, l2 = hessian2d_eig(dxx, dyy, dxy)

    L = np.stack((l1, l2), axis=2)
    L_abs = np.abs(L)

    rb = np.min(L_abs, axis=2) / np.max(L_abs, axis=2)

    s = np.sqrt(np.sum(L_abs ** 2, axis=2))

    m1 = np.exp(-rb**2 / 2 / beta**2)
    m1 *= (1 - np.exp(-s**2 / 2 / gamma**2))
    m1 *= (np.max(L, axis=2) < 0)

    return m1
