import numpy as np


KERNEL_SHAPE_MUL = 6


def hermitian(x, n=0):

    x = np.asarray(x)

    if n < 0:
        raise ValueError('The order of the Hermite polynomial can not be a negative number.')
    elif n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2*x
    else:
        # recursive formulation of Hermitian polynomials
        return 2*x*hermitian(x, n-1) - 2*(n-1)*hermitian(x, n-2)


def gaussian(x, sigma=1, n=0):

    x = np.asarray(x)

    if sigma <= 0:
        raise ValueError('The standard deviation of the Gaussian function must be a positive number.')

    if n < 0:
        raise ValueError('The order of the Gaussian derivative can not be a negative number.')
    elif n == 0:
        return np.exp(-(x*x)/(2*sigma*sigma)) / np.sqrt(2*np.pi)/sigma
    else:
        # n-th order derivatives of Gaussian can be computed as a product of a Gaussian and a n-th order Hermitian
        # polynomial, with an appropriate scaling factor
        c = sigma * np.sqrt(2)
        return hermitian(x/c, n) * gaussian(x, sigma, 0) * (-1/c)**n


def ndgauss(sigma, n=None, kernel_shape=None, separable=False):

    sigma = np.asarray(sigma)

    if n is None:
        n = np.zeros(sigma.shape)
    else:
        n = np.asarray(n)

    if n.ndim > 1 or sigma.ndim > 1 or len(n) != len(sigma):
        raise ValueError('The standard deviation and derivative order parameters must be vectors of the same length.')

    if kernel_shape is None:
        # round up to nearest odd
        kernel_shape = np.ceil(KERNEL_SHAPE_MUL*sigma) // 2 * 2 + 1
    else:
        kernel_shape = np.asarray(kernel_shape)

    x = [np.arange(s) - s // 2 for s in kernel_shape]

    if separable:
        return [gaussian(_x, _sigma, _n) for i, (_x, _sigma, _n) in enumerate(zip(x, sigma, n))]

    else:
        x = np.meshgrid(*x)

        x = [gaussian(_x, _sigma, _n) for _x, _sigma, _n in zip(x, sigma, n)]

        return np.prod(x, axis=0)
