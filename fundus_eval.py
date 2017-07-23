import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import data
import cam
import utils
from PIL import Image
import PIL
"""
import Image

background = Image.open("bg.png")
overlay = Image.open("ol.jpg")

background = background.convert("RGBA")
overlay = overlay.convert("RGBA")

new_img = Image.blend(background, overlay, 0.5)
new_img.save("new.png","PNG")
"""
NORMAL_LABEL = 0
ABNORMAL_LABEL = 1

def get_activation_map(image , filename):
    try:### error contor
        assert type(image).__module__ == np.__name__##check type if not image
    except AssertionError as ae:
        image=np.asarray(image)

    try:
        assert len(np.shape(image))==4
    except AssertionError as ae :
        h,w,ch=np.shape(image)
        image = image.reshape([1, h, w, ch])

    sess = tf.Session()
    saver = tf.train.import_meta_graph('./cnn_model/best_acc.ckpt.meta')
    saver.restore(sess, './cnn_model/best_acc.ckpt')
    tf.get_default_graph()
    accuray = tf.get_default_graph().get_tensor_by_name('accuracy:0')
    x_ = tf.get_default_graph().get_tensor_by_name('x_:0')
    y_ = tf.get_default_graph().get_tensor_by_name('y_:0')
    cam_ = tf.get_default_graph().get_tensor_by_name('classmap_reshape:0')
    top_conv = tf.get_default_graph().get_tensor_by_name('top_conv:0')
    phase_train = tf.get_default_graph().get_tensor_by_name('phase_train:0')
    y_conv = tf.get_default_graph().get_tensor_by_name('y_conv:0')
    vis_abnormal, vis_normal=cam.eval_inspect_cam(sess, cam_ ,top_conv , image , 1 ,x_ , y_ ,y_conv )
    NORMAL_LABEL = 0
    ABNORMAL_LABEL = 1
    image=np.squeeze(image)
    image=Image.fromarray(image)
    image.save('original_image.png')

    cmap=plt.get_cmap('jet')

    vis_abnormal=cmap(vis_abnormal)
    plt.imsave('actmap_abnormal.png', vis_abnormal)

    vis_abnormal=Image.open('./actmap_abnormal.png')
    plt.imshow(vis_abnormal)
    plt.show()
    original_img=Image.open('./original_image.png')
    plt.imshow(original_img)
    plt.show()
    background = original_img.convert("RGBA")
    overlay = vis_abnormal.convert("RGBA")

    overlay_img = Image.blend(background, overlay, 0.5)
    plt.imshow(overlay_img , cmap=plt.cm.jet)
    plt.show()
    overlay_img.save(filename)

    if __debug__ == False:

        plt.title('activation map : abnormal')
        plt.imshow(vis_abnormal)
        plt.show()
        plt.title('activation map : normal')
        plt.imshow(vis_normal)
        plt.show()
    return vis_normal

def eval(model_folder_path , images, labels=None):
    sess = tf.Session()
    saver = tf.train.import_meta_graph(model_folder_path+'best_acc.ckpt.meta')
    saver.restore(sess, model_folder_path+'best_acc.ckpt')
    tf.get_default_graph()
    accuray = tf.get_default_graph().get_tensor_by_name('accuracy:0')
    prediction = tf.get_default_graph().get_tensor_by_name('softmax:0')

    x_ = tf.get_default_graph().get_tensor_by_name('x_:0')
    y_ = tf.get_default_graph().get_tensor_by_name('y_:0')
    top_conv = tf.get_default_graph().get_tensor_by_name('top_conv:0')
    phase_train = tf.get_default_graph().get_tensor_by_name('phase_train:0')
    y_conv = tf.get_default_graph().get_tensor_by_name('y_conv:0')

    #cam_ = tf.get_default_graph().get_tensor_by_name('classmap_reshape:0')
    #vis_abnormal, vis_normal = cam.eval_inspect_cam(sess, cam_, top_conv, images, 1, x_, y_, y_conv)
    #NORMAL_LABEL = 0
    #ABNORMAL_LABEL = 1

    if not labels==None:
        acc,pred=sess.run([accuray , prediction] , feed_dict={x_:images ,y_ : labels , phase_train: False})
        return acc,pred
    else:
        pred=sess.run([ prediction] , feed_dict={x_:images ,y_ : labels , phase_train: False})
        return pred


""" Usage:
sess=tf.Session()
saver=tf.train.import_meta_graph('./cnn_model/best_acc.ckpt.meta')
saver.restore(sess,'./cnn_model/best_acc.ckpt')
tf.get_default_graph()
accuray=tf.get_default_graph().get_tensor_by_name('accuracy:0')
x_=tf.get_default_graph().get_tensor_by_name('x_:0')
y_=tf.get_default_graph().get_tensor_by_name('y_:0')
cam_=tf.get_default_graph().get_tensor_by_name('classmap_reshape:0')
top_conv = tf.get_default_graph().get_tensor_by_name('top_conv:0')
phase_train=tf.get_default_graph().get_tensor_by_name('phase_train:0')
y_conv = tf.get_default_graph().get_tensor_by_name('y_conv:0')
"""

if __name__ =='__main__':
    test_imgs = np.load('./test_imgs.npy')
    test_labs = np.load('./test_labs.npy')
    test_labs=test_labs.astype(np.int32)
    print test_labs
    act_map=get_activation_map(test_imgs[3], './sample_image.png')

    """
    imgs_list , labels_list=utils.divide_images_labels_from_batch(test_imgs ,test_labs, batch_size=60)
    list_imgs_labs=zip(imgs_list , labels_list)
    mean_acc=[]
    for img,lab in list_imgs_labs:
        test_acc=sess.run([accuray] , feed_dict={x_:img , y_:lab ,phase_train:False})
        mean_acc.append(test_acc)
    print np.mean(mean_acc)
    ####eval Class Activation Map####
    """
    """
    vis_abnormal, vis_normal=cam.eval_inspect_cam(sess, cam_ ,top_conv , test_imgs[0:1] , 1 ,x_ , y_ ,y_conv )
    plt.imshow(vis_abnormal)
    plt.show()
    plt.close()
    plt.imshow(vis_normal)
    plt.show()
    """



