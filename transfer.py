#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import pickle
import os
import utils
import data
import glob
from cnn import affine  ,dropout
from PIL import Image
from sklearn.decomposition import PCA
import PIL

"""saved pre trained list 
 1.inception v3 model """

inception_v3_url= "http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz"
vgg16_url = "https://s3.amazonaws.com/cadl/models/vgg16.tfmodel"

def download_and_extract_model(url , data_dir):

    utils.donwload(url, download_dir= data_dir)
    name=url.split('/')[-1]
    file_path=os.path.join(data_dir,name)
    utils.extract(file_path,data_dir)


#dense net 처럼 다시 설계해야 한다
"""
class Transfer
    |
    |-inception_v3 
    |-resnet
    |-inception_v5
    |-vgg net
"""

class Transfer_inception_v3(object):
    """

    """
    def __init__(self , data_dir ,x_ , phase_train , keep_prob ,out_channels ):
        """

        :param data_dir: path for folder saved .pb file
        :param out_channels:  [1024 , n_classes]
        """

        # inception
        x_jpeg_name = "DecodeJpeg/contents:0"  # input image for jpeg format
        x_name = "DecodeJpeg:0"  # input image for the other format
        x_resized_image = "ResizeBilinear:0"
        softmax = "softmax:0"
        logits = "softmax/logits:0"
        transfer_layer = "pool_3:0"

        # load graph ,variable and initialize session
        self.graph = tf.Graph() #make graph
        with self.graph.as_default() :#set graph to default
            #만약 여기서 with 을 주지 않으면 , 아래 그래프을 복원하는 것들은 실행 되지 않는다.
            self.data_dir=data_dir
            pb_path_list=glob.glob(os.path.join(data_dir , '*.pb'))
            assert len(pb_path_list) ==1 , 'the number of protobuffer has to be 1 , {}'.format(len(pb_path_list))
            pb_path=pb_path_list[0]
            print 'Proto Buffer file path - {}'.format(pb_path)

            gfile = tf.gfile.FastGFile(pb_path, 'rb') #get file point ,
            graph_def = tf.GraphDef() # make graph definition,
            graph_def.ParseFromString(gfile.read())  # load pb file to into graph_def
            tf.import_graph_def(graph_def, name='')  # import graph_def into tensorflow graph

            self.x_ = tf.get_default_graph().get_tensor_by_name(x_name)
            # Get tensor name from inception v3 graph
            self.pred = tf.get_default_graph().get_tensor_by_name(softmax)
            self.logits = tf.get_default_graph().get_tensor_by_name(logits)
            self.resized_image = tf.get_default_graph().get_tensor_by_name(x_resized_image)
            self.transfer_layer = tf.get_default_graph().get_tensor_by_name(transfer_layer)
            self.sess= tf.Session(graph=self.graph) # import graph to session
            self.transfer_layer_len=self.transfer_layer.get_shape()[3] #transfer layer shape : [1,1,1,2048]
            self.x_1 = x_
            self.phase_train = phase_train
            self.out_channels = out_channels
            self.keep_prob = keep_prob
        self._build_model()

        #feed dict
        # Image is passed in as a 4-dimension
        # The pixels MUST be values between 0 and 255 (float or int).

    def _create_feed_dict(self , image ):
        feed_dict = {self.x_: image}
        return feed_dict

    def classify(self , image ):
        assert np.max(image) > 1 , 'max values of images {} '.format(np.max(image))
        feed_dict=self._create_feed_dict(image)
        pred = self.sess.run(fetches=self.pred , feed_dict= feed_dict)
        print 'pred shape : {}'.format(np.shape(pred))
        pred = np.squeeze(pred)
        return pred

    def resize_image(self , image ):
        assert np.max(image) > 1, 'max values of images {} '.format(np.max(image))
        feed_dict = self._create_feed_dict(image)
        resized_image = self.sess.run(fetches=self.resized_image, feed_dict=feed_dict)
        resized_image = resized_image.squeeze(axis=0)

    def normalize_image(self , image):
        assert np.max(image) > 1, 'max values of images {} '.format(np.max(image))
        return image/255.

    def get_transfer_values(self , image ):
        assert np.max(image) > 1, 'max values of images {} '.format(np.max(image))
        # Get transfer value from pooling layer
        feed_dict=self._create_feed_dict(image)
        transfer_values = self.sess.run(fetches=self.transfer_layer , feed_dict = feed_dict )
        transfer_values=np.squeeze(transfer_values)
        return transfer_values
    def images_to_transfer_values(self , images):
        multiple_values=map(lambda image : self.get_transfer_values(image) , images )
        multiple_values=np.asarray(multiple_values)
        print 'multiple values shape {}'.format(np.shape(multiple_values))
        return multiple_values

    def images2caches(self ,cache_path , images , new_flag=True):
        if os.path.isfile(cache_path):
            print 'load saved caches '
            with open(cache_path, mode='rb') as file:
                obj = pickle.load(file)
        elif not os.path.isfile(cache_path) or new_flag:
            print("- create Data and saved to cache-file: " + cache_path)
            with open(cache_path, mode='wb') as file:
                obj = self.images_to_transfer_values(images)
                pickle.dump(obj, file)
        return obj
    def _build_model(self):
        n_classes=self.out_channels[-1]
        print 'N classes {} :'.format(n_classes)
        layer=self.x_1
        for i in range(len(self.out_channels)-1):
            layer = affine('fc_{}'.format(i), layer, out_ch=self.out_channels[i])
            layer=dropout(layer,self.phase_train , self.keep_prob)

        self.logits = affine('logits', layer, out_ch=n_classes)



class Transfer_vgg16(object):
    tensor_name_input_image = "images:0"

    # Names of the tensors for the dropout random-values..
    tensor_name_dropout = 'dropout/random_uniform:0'
    tensor_name_dropout1 = 'dropout_1/random_uniform:0'

    # Names for the convolutional layers in the model for use in Style Transfer.
    layer_names = ['conv1_1/conv1_1', 'conv1_2/conv1_2',
                   'conv2_1/conv2_1', 'conv2_2/conv2_2',
                   'conv3_1/conv3_1', 'conv3_2/conv3_2', 'conv3_3/conv3_3',
                   'conv4_1/conv4_1', 'conv4_2/conv4_2', 'conv4_3/conv4_3',
                   'conv5_1/conv5_1', 'conv5_2/conv5_2', 'conv5_3/conv5_3']

    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            path = os.path.join(data_dir, path_graph_def)
            with tf.gfile.FastGFile(path, 'rb') as file:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(file.read())
                tf.import_graph_def(graph_def, name='')
