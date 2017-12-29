#-*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from PIL import Image
import PIL
import utils
import numpy as np
import fundus_processing
from skimage.io import imsave
import scipy.misc
def get_class_map(name,x , label , im_width):
    out_ch = int(x.get_shape()[-1])
    conv_resize=tf.image.resize_bilinear(x,[im_width , im_width])
    with tf.variable_scope(name , reuse = True) as scope:
        label_w = tf.gather(tf.transpose(tf.get_variable('w')) , label)
        label_w = tf.reshape(label_w , [-1, out_ch , 1])
    conv_resize = tf.reshape(conv_resize , [-1 , im_width *im_width , out_ch])
    classmap = tf.matmul(conv_resize , label_w , name= 'classmap')
    classmap = tf.reshape(classmap ,[-1 , im_width , im_width] ,name='classmap_reshape')
    return classmap

def inspect_cam(sess, cam, top_conv, test_imgs, test_labs, x_, y_, phase_train, logit, savedir_root='actmap',
                    labels=None , cropped_img_size = (224,224)):

    debug_flag=False
    assert np.ndim(test_imgs) ==4 , "test_imgs's rank must be 4 , test_imgs rank :{}".format(np.ndim(test_imgs))
    num_images, ori_img_h, ori_img_w, ori_img_ch = np.shape(test_imgs)
    if np.max(test_imgs) <= 1:
        test_imgs = test_imgs * 255
        print "test_imgs's max {}".format(np.max(test_imgs))
    try:
        os.mkdir(savedir_root);
    except Exception:
        pass;
    for s in range(num_images):
        utils.show_progress(s , num_images)
        save_dir=os.path.join(savedir_root ,'img_{}'.format(s))
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir);
        ori_img = test_imgs[s]
        if ori_img.shape[-1]==1: #gray image
            plt.imsave('{}/image_test.png'.format(save_dir), ori_img.reshape([ori_img_h, ori_img_w]))
        else :
            if np.max(ori_img) > 1:
                plt.imsave('{}/image_test.png'.format(save_dir), ori_img/255.)
            else:
                plt.imsave('{}/image_test.png'.format(save_dir), ori_img/255.)
        if labels is None: #만약 어디 class의 activation map 을 볼지 지정하지 않는다면 test_labels(정답)의 라벨을 본다
            label = test_labs[s:s+1]
        else :
            label=labels
        top_conv_, logit_ = sess.run([top_conv, logit],
                                     feed_dict={x_: ori_img.reshape([1, ori_img_h, ori_img_w, ori_img_ch]),
                                                phase_train: False})
        cam_= sess.run( cam ,  feed_dict={ y_:label , top_conv:top_conv_ }) #cam_answer

        cam_=np.asarray((map(lambda x: (x-x.min())/(x.max()-x.min()) , cam_))) #-->why need this?


        #기존의 class activation map을 뽑을때 실수를 해서
        #299, 299 사이즈로 activation map을 biinear interpolation 을 했다
        # 그래서 기존의 것과 같이 쓸수 잇도록 수정한다
        try:
            cam_img = cam_.reshape((ori_img_h, ori_img_w))
            cam_img = cam_img.reshape((ori_img_h, ori_img_w)) # why need this?
        except:
            cam_img = cam_.reshape(cropped_img_size)
            cam_img = cam_img.reshape(cropped_img_size)  # why need this?

        plt.imshow(cam_img, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest',
                   vmin=0, vmax=1)
        #plt.show()
        cmap = plt.cm.jet
        plt.imsave('{}/actmap_abnormal_label_0.png'.format(save_dir), cmap(cam_img))
        cam_img=Image.open('{}/actmap_abnormal_label_0.png'.format(save_dir))
        ##임시로 한것이다 나중에 299 가 아닌 224로 고쳐진 코드가 있으면 지우자
        cam_img=cam_img.resize((224,224) , PIL.Image.ANTIALIAS)
        np_cam_img=np.asarray(cam_img) #img 2 numpy
        np_cam_img=fundus_processing.add_padding(np_cam_img.reshape(1,224,224,-1) , 299,299) # padding
        cam_img = Image.fromarray(
            np_cam_img.reshape([ori_img_h, ori_img_w , 4 ]).astype('uint8'))  # padding#numpy 2 img


        ori_img=Image.fromarray(ori_img.astype('uint8')).convert("RGBA")
        #cam_img = Image.fromarray(cam_img.astype('uint8')).convert("RGBA")
        overlay_img = Image.blend(ori_img, cam_img, 0.5)
        plt.imshow(overlay_img)
        plt.imsave('{}/overlay.png'.format(save_dir), overlay_img)
        #plt.show()
        plt.close();
        """
        if test_imgs.shape[-1] == 1:  # grey
            plt.imshow(1 -img.reshape([test_imgs.shape[1], test_imgs.shape[2]]))
            plt.show()
        """


def merge_all_cam(images):
    n, cropped_h, cropped_w, ch = np.shape(images)
    merged_image = np.zeros(299, 299, 3)
    gap_h = int(299 - cropped_h / 2)
    gap_w = int(299 - cropped_w / 2)

    merged_image[:cropped_h, :cropped_w, :] += images[0]
    merged_image[:cropped_h, -cropped_w:, :] += images[2]


if __name__ == '__main__':
    images=[]
    left_top = np.asarray(
        Image.open('/Users/seongjungkim/PycharmProjects/fundus/actmap/sample/left_top.png').convert('RGB'))
    right_top = np.asarray(
        Image.open('/Users/seongjungkim/PycharmProjects/fundus/actmap/sample/right_top.png').convert('RGB'))

    merged = np.add(left_top , right_top)
    images.append(left_top)

    images.append(right_top)
    images.append(merged)
    print np.max(merged)
    print np.max(right_top)
    plt.imshow(merged)
    plt.show()
    print np.shape(np.asarray(images))



"""
def eval_inspect_cam(sess, cam , top_conv ,test_imgs , x, y_ ,phase_train, y , savedir_root):
    ABNORMAL_LABEL =np.asarray([[0,1]])
    NORMAL_LABEL = np.asarray([[1,0]])

    try:
        os.mkdir(savedir_root);
    except Exception:
        pass;

    num_images = len(test_imgs)
    for s in range(num_images):
        print s
        save_dir=os.path.join(savedir_root , 'img_{}'.format(s))
        try:os.mkdir(save_dir);
        except Exception:pass;
        if __debug__ ==True:
            print test_imgs[s].shape
        if test_imgs[s].shape[-1]==1:
            plt.imsave('{}/image_test.png'.format(save_dir) ,test_imgs[s].reshape([test_imgs[s].shape[0] , test_imgs.shape[1]]))
        else :
            plt.imsave('{}/image_test.png'.format(save_dir), test_imgs[s])
        img=test_imgs[s:s+1]
        conv_val , output_val =sess.run([top_conv , y] , feed_dict={x:img , phase_train:False})
        cam_ans_abnormal= sess.run( cam ,  feed_dict={ y_:ABNORMAL_LABEL , top_conv:conv_val ,phase_train:False })
        cam_ans_normal = sess.run(cam, feed_dict={y_: NORMAL_LABEL, top_conv: conv_val , phase_train:False})


        cam_vis_abnormal=list(map(lambda x: (x-x.min())/(x.max()-x.min()) , cam_ans_abnormal))
        cam_vis_normal=list(map(lambda x: (x-x.min())/(x.max()-x.min()) , cam_ans_normal))

        for vis , ori in zip(cam_vis_abnormal , img):

            if ori.shape[-1]==1: #grey
                plt.imshow( 1-ori.reshape([ori.shape[0] , ori.shape[1]]))
            vis_abnormal=vis.reshape([vis.shape[0], vis.shape[1]])
            plt.imshow( vis_abnormal, cmap=plt.cm.jet , alpha=0.5 , interpolation='nearest' , vmin=0 , vmax=1)
            plt.show()
            plt.imsave('{}/actmap_abnormal_label_0.png'.format(save_dir) , vis_abnormal)
        for vis, ori in zip(cam_vis_normal, img):

            if ori.shape[-1] == 1:  # grey
                plt.imshow(1 - ori.reshape([ori.shape[0], ori.shape[1]]))
            vis_normal = vis.reshape([vis.shape[0], vis.shape[1]])
            plt.imshow(vis_normal, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest', vmin=0, vmax=1)
            plt.imsave('{}/actmap_abnormal_label_1.png'.format(save_dir) , vis_normal)
        return vis_abnormal , vis_normal
"""