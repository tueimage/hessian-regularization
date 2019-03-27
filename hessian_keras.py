import keras.backend as K

from gaussian_kernels import *


def hessian2d_keras(img, sigma, scale_norm=True):

    if scale_norm:
        c = sigma**2
    else:
        c = 1

    def conv(_img, _w):
        w0 = _w[0][np.newaxis, :, np.newaxis, np.newaxis]
        w0 = K.constant(w0)
        w1 = _w[1][:, np.newaxis, np.newaxis, np.newaxis]
        w1 = K.constant(w1)

        return K.conv2d(K.conv2d(_img, w0, padding='same'), w1, padding='same')

    dxx = c * conv(img, ndgauss((sigma, sigma), n=(0, 2), separable=True))
    dyy = c * conv(img, ndgauss((sigma, sigma), n=(2, 0), separable=True))
    dxy = c * conv(img, ndgauss((sigma, sigma), n=(1, 1), separable=True))

    return dxx, dyy, dxy


def hessian2d_eig_keras(dxx, dyy, dxy):

    c = K.sqrt((dxx - dyy)*(dxx - dyy) + 4*dxy*dxy)

    l1 = (dxx + dyy + c)/2
    l2 = (dxx + dyy - c)/2

    return l1, l2


def blobiness2d_keras(img, sigma, alpha=0.5, gamma=5):
    '''
    The case in [1] where N = 2, M = 0.

    References:
    [1] Antiga, Luca. "Generalizing vesselness with respect to dimensionality and shape." The Insight Journal 3 (2007): 1-14.
    '''

    dxx, dyy, dxy = hessian2d_keras(img, sigma)

    l1, l2 = hessian2d_eig_keras(dxx, dyy, dxy)

    L = K.stack((l1, l2), axis=3)
    L_abs = K.abs(L)

    ra = K.min(L_abs, axis=3) / K.max(L_abs, axis=3)

    s = K.sqrt(K.sum(L_abs**2, axis=3))

    m0 = (1 - K.exp(-ra**2 / 2 / alpha**2))
    m0 *= (1 - K.exp(-s**2 / 2 / gamma**2))
    m0 *= K.cast(K.less(l1, 0), 'float32') * K.cast(K.less(l2, 0), 'float32')

    return m0
