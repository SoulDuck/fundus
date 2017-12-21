#-*- coding: utf-8 -*-
import matplotlib
import os
if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    matplotlib.use('Agg')

import tensorflow as tf
import cam
import numpy as np
import os
import utils
import data
import fundus_processing

## for mnist dataset ##
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
"""
train_imgs = mnist.train.images.reshape([-1,28,28,1])
train_labs = mnist.train.labels
test_imgs = mnist.test.images.reshape([-1,28,28,1])
test_labs = mnist.test.labels
"""
#for Fundus_300
def get_acc(preds , trues):
    #onehot vector check
    np.ndim(preds) == np.ndim(trues) , 'predictions and True Values has same shape and has to be OneHot Vector'
    if np.ndim(preds) == 2:
        preds_cls =np.argmax(preds , axis=1)
        trues_cls = np.argmax(trues, axis=1)

    else:
        preds_cls=preds
        trues_cls = trues
    acc=np.sum([preds_cls == trues_cls])/float(len(preds_cls))
    return acc





def eval(model_path ,test_images , batch_size  , actmap_save_root_folder='./actmap'):
    """
    :param model_path:
    :param test_images:
    :param batch_size:
    :param save_root_folder: folder to save class activation map
    :return:
    """
    print 'eval'
    b,h,w,c=np.shape(test_images)

    if np.max(test_images) > 1:
        test_images = test_images / 255.
    sess = tf.Session()

    saver = tf.train.import_meta_graph(meta_graph_or_file=model_path+'.meta') #example model path ./models/fundus_300/5/model_1.ckpt
    saver.restore(sess, save_path=model_path) # example model path ./models/fundus_300/5/model_1.ckpt

    x_ = tf.get_default_graph().get_tensor_by_name('x_:0')
    y_ = tf.get_default_graph().get_tensor_by_name('y_:0')
    pred_ = tf.get_default_graph().get_tensor_by_name('softmax:0')
    try:
        is_training_=tf.get_default_graph().get_tensor_by_name('is_training:0')
    except:
        is_training_ = tf.get_default_graph().get_tensor_by_name('phase_train:0')
    top_conv = tf.get_default_graph().get_tensor_by_name('top_conv:0')
    try:
        logits = tf.get_default_graph().get_tensor_by_name('logits:0')
    except:
        logits = tf.get_default_graph().get_tensor_by_name('y_conv:0')
    cam_ = tf.get_default_graph().get_tensor_by_name('classmap:0')
    cam.inspect_cam(sess, cam_, top_conv, test_images, test_labels, x_, y_, is_training_, logits,
                    savedir_root=actmap_save_root_folder)

    #def inspect_cam(sess, cam , top_conv , test_imgs, test_labs, global_step , x_ , y_ , phase_train , y  , savedir='actmap'):
    """
    try:
        print np.shape(vis_abnormal)
        vis_normal=vis_normal.reshape([h,w])
        vis_abnormal = vis_abnormal.reshape([h,w])
        plt.imshow(vis_normal)
        plt.show()
        plt.imshow(vis_abnormal)
        plt.show()
    except Exception as e :
        print e
        pass
    """
    share=len(test_images)/batch_size
    remainder=len(test_images)%batch_size
    predList=[]
    for s in range(share):
        pred = sess.run(pred_, feed_dict={x_: test_images[s * batch_size:(s + 1) * batch_size], is_training_: False})
        print 'pred_ ' ,pred
        predList.extend(pred)
    if not remainder == 0:
        pred = sess.run(pred_, feed_dict={x_: test_images[-1*remainder:], is_training_: False})
        predList.extend(pred)
    assert len(predList) == len(test_images) , 'n pred list : {} n test images : {}'.format(len(predList) , len(test_images))
    tf.reset_default_graph()
    #print 'pred sample ',predList[:1]
    return np.asarray(predList)

def eval_image_with_sparse_croppping(model_path , image , image_size , actmap_save_folder):
    print actmap_save_folder
    cropped_height, cropped_weight = image_size
    sparse_cropped_images = fundus_processing.sparse_crop(image, cropped_height, cropped_weight, lr_flip=True,
                                                          ud_flip=True)
    sparse_cropped_images = fundus_processing.add_padding(sparse_cropped_images, 299, 299)
    #utils.plot_images(sparse_cropped_images)
    pred = eval(model_path, sparse_cropped_images, batch_size=5, actmap_save_root_folder=actmap_save_folder)
    pred_0 = np.sum(pred[:, 0])
    pred_1 = np.sum(pred[:, 1])
    mean_pred = (pred_0 / float(len(pred)) ,pred_1 / float(len(pred)))
    return mean_pred
def eval_image_with_dense_croppping(model_path , image , image_size , actmap_save_folder):
    pass;

def eval_images(model_path , images , image_size , cropping_type , labels=None ):
    mean_preds=[]
    assert  len(images) > 1
    for i , image in enumerate(images):
        if cropping_type =='sparse':
            mean_pred = eval_image_with_sparse_croppping(model_path, image, image_size,
                                                         actmap_save_folder=os.path.join('./actmap' , str(i)))
        elif cropping_type == 'dense':
            mean_pred = eval_image_with_dense_croppping(model_path, image, image_size,
                                                         actmap_save_folder=os.path.join('./actmap' , str(i)))
        else:
            raise AssertionError
        mean_preds.append(mean_pred)
    mean_preds=np.asarray(mean_preds)
    if not labels is None:
        acc = utils.get_acc(pred=mean_preds, labels=labels)
    return mean_preds, acc



#def get_cam_with_sparse_cropped_images()
if __name__ =='__main__':

    train_images, train_labels, train_filenames, test_images, test_labels, test_filenames = data.type1('./fundus_300',
                                                                                                       resize=(
                                                                                                       299, 299))
    model_path = './ensemble_models/step_21600_acc_0.848333358765/model'
    #mean_pred=eval_image_with_sparse_croppping(model_path , test_images[0] , (224, 224) )
    preds=eval_images(model_path ,  test_images , (224,224) , 'sparse',test_labels)
    print len(preds)
    print len(preds[:10])


    """
    sparse_cropped_images=fundus_processing.sparse_crop(test_images[0] , 224 ,224 ,lr_flip=True , ud_flip=True)
    print np.shape(sparse_cropped_images)
    print np.max(sparse_cropped_images)
    sparse_cropped_images=fundus_processing.add_padding(sparse_cropped_images , 299 ,299 )
    print np.shape(sparse_cropped_images)
    print np.max(sparse_cropped_images)
    utils.plot_images(sparse_cropped_images)
    pred=eval(model_path, sparse_cropped_images , batch_size=5 , save_root_folder='./actmap')
    pred_0 = np.sum(pred[:, 0])
    pred_1 = np.sum(pred[:, 1])
    print pred_0/float(len(pred))
    print pred_1/float(len(pred))
    print np.shape(pred)
    """
