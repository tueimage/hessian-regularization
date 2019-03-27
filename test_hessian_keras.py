import os

from skimage import data
from skimage.color import rgb2gray
from skimage import io

import keras.backend as K
import tensorflow as tf

from hessian_keras import *
from test_common import *

os.environ['KMP_DUPLICATE_LIB_OK']='True'
sess = tf.InteractiveSession()

# %% test hessian()

img = data.hubble_deep_field()[0:500, 0:500]
img = rgb2gray(img)
img = np.reshape(img, (1, img.shape[0], img.shape[1], 1))
img = K.variable(img)

dxx, dyy, dxy = hessian2d_keras(img, 10)

tf.initialize_all_variables().run()

show_image(np.squeeze(img.eval()))

show_image(np.squeeze(dxx.eval()))
show_image(np.squeeze(dyy.eval()))
show_image(np.squeeze(dxy.eval()))

# %% test hessian_eig()

img = data.hubble_deep_field()[0:500, 0:500]
img = rgb2gray(img)
img = np.reshape(img, (1, img.shape[0], img.shape[1], 1))
img = K.variable(img)

dxx, dyy, dxy = hessian2d_keras(img, 5)
l1, l2 = hessian2d_eig_keras(dxx, dyy, dxy)

tf.initialize_all_variables().run()

show_image(np.squeeze(img.eval()))

show_image(np.squeeze(l1.eval()))
show_image(np.squeeze(l2.eval()))

# %% test blobiness2d()

img = data.hubble_deep_field()[0:500, 0:500]
img = rgb2gray(img)
img = np.reshape(img, (1, img.shape[0], img.shape[1], 1))
img = K.variable(img)

b = blobiness2d_keras(img, 5, alpha=0.5, gamma=5)

tf.initialize_all_variables().run()

show_image(np.squeeze(img.eval()))

show_image(np.squeeze(b.eval()))
