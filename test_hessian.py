from skimage import data
from skimage.color import rgb2gray
from skimage import io

from hessian import *
from test_common import *

# %% test hessian()

img = data.hubble_deep_field()[0:500, 0:500]
img = rgb2gray(img)
dxx, dyy, dxy = hessian2d(img, 10)

show_image(img)

show_image(dxx)
show_image(dyy)
show_image(dxy)

# %% test hessian_eig()

img = data.hubble_deep_field()[0:500, 0:500]
img = rgb2gray(img)

dxx, dyy, dxy = hessian2d(img, 5)
l1, l2 = hessian2d_eig(dxx, dyy, dxy)

show_image(img)

show_image(l1)
show_image(l2)

# %% test blobiness2d()

img = data.hubble_deep_field()[0:500, 0:500]
img = rgb2gray(img)

show_image(img)

show_image(blobiness2d(img, 15))

show_image(blobiness2d(img, 15, alpha=0.1, gamma=1))

show_image(blobiness2d(img, 15, alpha=2, gamma=20))

show_image(blobiness2d(img, 5) + blobiness2d(img, 15))

# %% test vesselness2d()

img = io.imread('data/fundus2D.png')
img = img.astype('float')
img = 255-rgb2gray(img)
img = img[100:-100, 100:-100]

show_image(img)

show_image(vesselness2d(img, 3, beta=0.5, gamma=15))

# %%

v = np.zeros((img.shape[0], img.shape[1], 10))

for s in np.linspace(1.5, 5.5, 8):
    v[:, :, -1] = vesselness2d(img, s, beta=0.5, gamma=15)

show_image(np.max(v, axis=2))
