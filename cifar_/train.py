import sys , os
import cnn
import tensorflow as tf
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import resnet_

resnet_.


n_classes = 10
x_ = tf.placeholder(dtype = tf.float32 , shape=[None ,32 ,32 ,3 ])
y_ = tf.placeholder(dtype = tf.float32 , shape=[None , n_classes] )
phase_train = tf.placeholder(dtype = tf.bool , name = 'phase_train')
n_filters_per_box = [16, 16, 32, 32]
n_blocks_per_box = [5, 5, 5, 5]
stride_per_box = [5, 5, 5, 5]
use_bottlenect = True

model = resnet_.Resnet(x_, phase_train, n_filters_per_box, n_blocks_per_box, stride_per_box, \
                       use_bottlenect, n_classes=2, activation=tf.nn.relu, logit_type='gap')
print model.logit

