#-*- coding: utf-8 -*-
import matplotlib
import os
if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    matplotlib.use('Agg')

import tensorflow as tf
import cam
import glob
import numpy as np
import os
import utils
import data
from PIL import Image
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





def eval(model_path ,test_images , batch_size  , actmap_save_root_folder='./actmap' , ):
    """
    :param model_path:
    :param test_images:
    :param batch_size:
    :param save_root_folder: folder to save class activation map
    :return:
    """
    print 'eval'
    if len(np.shape(test_images)) ==3:
        h,w,ch=np.shape(test_images)
        test_images=test_images.reshape([1,h,w,ch])

    b,h,w,c=np.shape(test_images)

    if np.max(test_images) > 1:
        test_images = test_images / 255.
    sess = tf.Session()
    saver = tf.train.import_meta_graph(meta_graph_or_file=model_path+'.meta') #example model path ./models/fundus_300/5/model_1.ckpt
    saver.restore(sess, save_path=model_path) # example model path ./models/fundus_300/5/model_1.ckpt
    try:
        x_ = tf.get_default_graph().get_tensor_by_name('x_:0')
    except :
        x_ = tf.get_default_graph().get_tensor_by_name('Placeholder:0')
    try:
        y_ = tf.get_default_graph().get_tensor_by_name('y_:0')
    except:
        y_ = tf.get_default_graph().get_tensor_by_name('Placeholder_1:0')

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
    cam.inspect_cam(sess, cam_, top_conv, test_images, None , x_, y_, is_training_, logits,
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

def eval_image_with_sparse_croppping(model_path , image , label , image_size , actmap_save_folder):

    """
    label shape 가 .... [1,2] or [2,] 이면 ....에 대한 처리를 해줘야 한다

    :param model_path:
    :param image:
    :param label:
    :param image_size:
    :param actmap_save_folder:
    :return:
    """
    assert np.ndim(label) ==1 , "{label's rank {} }".format(np.ndim(label))

    cropped_height, cropped_weight = image_size
    sparse_cropped_images = fundus_processing.sparse_crop(image, cropped_height, cropped_weight, lr_flip=False,
                                                          ud_flip=False)
    labels=[]
    for i in range(len(sparse_cropped_images)):
        labels.append(label)
    labels=np.asarray(labels)
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
    print 'n images : {} '.format(len(images))
    if cropping_type == 'central':
        mean_pred = np.squeeze(eval(model_path, images, batch_size=60,
                            actmap_save_root_folder=os.path.join('./actmap', 'central')))
        # 이걸 너무 느리다  batch size  만큼 수정하게 해야 한다
    else:
        for i , image in enumerate(images):

            if cropping_type =='sparse':
                mean_pred = eval_image_with_sparse_croppping(model_path, image, image_size,
                                                             actmap_save_folder=os.path.join('./actmap' , str(i)))
            elif cropping_type == 'dense':
                mean_pred = eval_image_with_dense_croppping(model_path, image, image_size,
                                                             actmap_save_folder=os.path.join('./actmap' , str(i)))
            else:
                raise AssertionError
            print mean_pred
            print np.shape(mean_pred)
            mean_preds.append(mean_pred)

        mean_preds=np.asarray(mean_preds)

    if not labels is None:
        acc = utils.get_acc(true=labels,pred=mean_preds )
        print acc
    return mean_pred



#def get_cam_with_sparse_cropped_images()
if __name__ =='__main__':

    #train_images, train_labels, train_filenames, test_images, test_labels, test_filenames = data.type1('./fundus_300',
    #                                                                                                   resize=(
    #                                                                                                   299, 299))

    #Iruda Image File

    paths=glob.glob('./iruda/*.JPG')
    test_images=map(lambda path :Image.open(path).resize((299,299) , Image.ANTIALIAS), paths)

    test_images=map(np.asarray , test_images)
    test_images=np.asarray((test_images))
    test_labels=np.zeros([len(test_images),2])
    test_labels[:,0]=1

    model_path = './ensemble_models/bottlenect_fc_16@3_32@4_64@6_128@3_no_color_aug_bf_4_l2loss_rotateAug_adam/model'
    model_path = './ensemble_models/alexnet_step_312700_acc_0.849180327869/model'
    model_path = './ensemble_models/5556_resnet_step_47800_acc_0.84262295082/model'
    model_path = './ensemble_models/residual_fc_16@2_32@2_64@2_128@2_no_color_aug_2/model'
    model_path ='./ensemble_models/vgg11_5_step_24000_acc_0.848333358765/model'
    #mean_pred=eval_image_with_sparse_croppping(model_path , test_images[0] , (224, 224) )


    preds=eval_images(model_path ,  test_images/255. , (224,224) , 'central',test_labels)
    preds=np.asarray(preds)
    np.save('./iruda_preds.npy')


    print len(preds)
    print len(preds[:2])
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
