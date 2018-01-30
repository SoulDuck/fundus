import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_imgs=mnist.train.images.reshape([-1,28,28,1])
train_labs=mnist.train.labels
val_imgs = mnist.validation.images.reshape([-1,28,28,1])
val_labs= mnist.validation.labels
print np.shape(train_imgs)
print np.shape(train_labs)
print np.shape(val_imgs)
print np.shape(val_labs)