from scipy.ndimage.filters import correlate

from gaussian_kernels import *
from test_common import *

# %% test hermitian()

x = np.linspace(-2, 2, 100)

plt.figure()

for n in range(5):
    plt.plot(x, hermitian(x, n))

plt.show()

# %% test gaussian()

x = np.linspace(-2, 2, 100)

plt.figure()

for n in range(5):
    plt.plot(x, gaussian(x, 1, n))

plt.show()

# %% test ndgauss()

show_image(ndgauss((10, 20), (0, 0), (80, 80)))
show_image(ndgauss((10, 20), (1, 0), (80, 80)))
show_image(ndgauss((10, 20), (0, 1), (80, 80)))
show_image(ndgauss((10, 20), (0, 2), (80, 80)))
show_image(ndgauss((10, 20), (2, 0), (80, 80)))

# %%

w = ndgauss((10, 20), (2, 0), (80, 80), separable=True)

show_image(w[0]*w[1][np.newaxis].T - ndgauss((10, 20), (2, 0), (80, 80)))
