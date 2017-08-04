import glob
import tensorflow as tf
import numpy as np
import matplotlib
import os
if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import data
import cam
import utils
from PIL import Image
import argparse
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

    if not model_folder_path.endswith('/'):
        model_folder_path=model_folder_path+'/'
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
    if type(labels).__module__ == np.__name__ :
        'label data type : numpy '
        acc,pred=sess.run([accuray , prediction] , feed_dict={x_:images ,y_ : labels , phase_train: False})
        return acc,pred
    else:
        'label data not assin '
        pred=sess.run([ prediction] , feed_dict={x_:images ,y_ : labels , phase_train: False})
        return pred


"""
def eval_from_paths(ath_dir , model_dir):
    for file in files:
        if 'test' in file:

            file_name = file.split('/')[-1]  # e.g glaucoma_test_paths.txt
            imgs_name = file_name.replace('paths.txt', 'imgs.npy')  # e.g glaucoma_test_imgs.npy
            labs_name = file_name.replace('paths.txt', 'labs.npy')  # e.g glaucoma_test_imgs.npy

            paths = utils.get_paths_from_text(file)
            if 'normal' in file_name:
                label = 0
            else:
                label = 1
            imgs, labs = data.make_numpy_images_labels(paths, label)
            imgs_list, labs_list = utils.divide_images_labels_from_batch(imgs, labs, 60)
            imgs_labs_list = zip(imgs_list, labs_list)
            acc_list = []
            predict_list = []
            for i, (imgs, labs) in enumerate(imgs_labs_list):
                labs = labs.astype(np.int32)
                labs = data.cls2onehot(labs, 2)
                # np.save(folder_path+imgs_name ,imgs )
                # np.save(folder_path + labs_name, labs)
                acc, predict = eval(model_path, imgs, labs[:len(imgs)])
                print i, ' acc :', acc
                acc_list.append(acc)
                predict_list.append(predict)
            acc_list = np.asarray(acc_list)
            predict_list = np.asarray(predict_list)
            acc = acc_list.mean()
            print 'accuracy', acc
            if __debug__ == True:
                print ''
                print '############debug##############'
                print 'file name', file_name
                print '# paths ', len(paths)
                print 'image shape', np.shape(imgs)
                print 'label', label
                print 'label shape', np.shape(labs)
                # print utils.plot_images(imgs)

"""
def eval_from_numpy_image(path_dir , model_dir):

    return_dict={}
    files=glob.glob(path_dir+'*.txt')
    for file in files:
        if 'test' in file:
            file_name=file.split('/')[-1] #e.g glaucoma_test_paths.txt
            image_name=file_name.replace('paths.txt' , 'images.npy') #e.g glaucoma_test_images.npy
            image_path=os.path.join(path_dir,image_name) # ./fundus/..../glaucoma_test_imgs.npy
            label_name = file_name.replace('paths.txt', 'labels.npy')  # e.g glaucoma_test_labels.npy
            label_path = os.path.join(path_dir, label_name) # ./fundus/..../glaucoma_test_labels.npy
            paths=utils.get_paths_from_text(file)
            if 'normal' in file_name:
                label=0
            else:
                label=1
            imgs=np.load(image_path);labs=np.load(label_path)
            imgs_list,labs_list=utils.divide_images_labels_from_batch(imgs,labs,60)
            imgs_labs_list=zip(imgs_list,labs_list)
            acc_list=[]
            predict_list=[]
            for i,(imgs,labs) in enumerate(imgs_labs_list):
                labs=labs.astype(np.int32)
                labs=data.cls2onehot(labs,2)
                #np.save(folder_path+imgs_name ,imgs )
                #np.save(folder_path + labs_name, labs)
                acc, predict = eval(model_dir, imgs, labs[:len(imgs)])
                print i,' acc :',acc
                acc_list.append(acc)
                predict_list.append(predict)
            acc_list=np.asarray(acc_list).reshape([-1])
            predict_list= np.asarray(predict_list).reshape([-1,2])

            np.reshape(acc_list , [1,-1])

            acc=acc_list.mean()
            print 'accuracy', acc
            if __debug__ ==True:
                print ''
                print '############debug##############'
                print 'file name',file_name
                print '# paths ',len(paths)
                print 'image shape',np.shape(imgs)
                print 'label' , label
                print 'label shape',np.shape(labs)
                #print utils.plot_images(imgs)
            data_name=file_name.replace('_test_paths.txt' , '') #e.g glaucoma_test_paths.txt -->glaucoma
            return_dict[data_name+'_acc']=acc_list
            return_dict[data_name + '_predict'] = predict_list
    return return_dict


def ensemble(model_root_dir, images, labels , batch=60):
    if  len(np.shape(labels)) ==1:
        print '***critical error***'
        print 'labels rank one , this functions need onehot-vector'
        raise ValueError

    path, names, files = os.walk(model_root_dir).next()
    print 'the number of model:', len(names)
    count=0
    for name in names[:1]:
        target_model = os.path.join(model_root_dir, name)
        if labels is None:
            'not implement'
            pred = eval(target_model, images, labels)
            count+=1
        else:
            '# images > batch'
            tot_pred=[]
            list_imgs, list_labs = utils.divide_images_labels_from_batch(images, labels, batch_size=batch)
            list_imgs_labs = zip(list_imgs, list_labs)
            for images , labels in list_imgs_labs:
                _ , tmp_pred = eval(target_model, images, labels)
                tot_pred.extend(tmp_pred)


            print 'length of pred :',len(pred)
            print 'length of cls : ',len(cls)
            print 'model name : ', name , 'accuracy:',acc
            tot_cls=np.argmax(tot_pred , axis=1)
            cls=np.argmax(labels, axis=1)
            acc=np.mean(np.equal(cls, tot_cls))
            if count==0:
                sum_pred=tmp_pred
            else:
                sum_pred+=tmp_pred
            count+=1
    sum_pred=sum_pred/float(count)
    tot_cls = np.argmax(sum_pred, axis=1)
    cls = np.argmax(labels, axis=1)
    acc = np.mean(np.equal(cls, sum_pred))

    print acc ,pred
    """
    for i, pred in enumerate(np_preds):
        if i == 0:
            pred_sum = pred
        else:
            pred_sum += pred
    pred_mean = pred_sum / len(np_preds)
    """
    return pred_mean, acc_mean




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
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_dir" , help='image folder to load')
    parser.add_argument("--model_dir" , help='model folder to load')
    parser.add_argument("--model_root_dir", help='model root folder that saved images')
    args = parser.parse_args()


args.path_dir

cataract_test_imgs=np.load(os.path.join(args.path_dir,'cataract_test_images.npy'))
cataract_test_cls=np.load(os.path.join(args.path_dir,'cataract_test_labels.npy'))
cataract_test_labs=data.cls2onehot(cataract_test_cls , depth=2)

"""
glaucoma_test_imgs=np.load(os.path.join(args.path_dir,'glaucoma_test_images.npy'))
glaucoma_test_cls=np.load(os.path.join(args.path_dir,'glaucoma_test_labels.npy'))
glaucoma_test_labs=data.cls2onehot(glaucoma_test_cls , depth=2)

retina_test_imgs=np.load(os.path.join(args.path_dir,'retina_test_images.npy'))
retina_test_cls=np.load(os.path.join(args.path_dir,'retina_test_labels.npy'))
retina_test_labs=data.cls2onehot(retina_test_cls , depth=2)

normal_test_imgs=np.load(os.path.join(args.path_dir,'normal_test_images.npy'))
normal_test_cls=np.load(os.path.join(args.path_dir,'normal_test_labels.npy'))
normal_test_labs=data.cls2onehot(normal_test_cls , depth=2)
"""

print 'the number of cataract images',len(cataract_test_imgs)
cataract_pred , cataract_acc =ensemble(args.model_root_dir , cataract_test_imgs , cataract_test_labs)
print cataract_pred
print cataract_acc
"""
glaucoma_pred , glaucoma_acc =ensemble(args.model_root_dir , glaucoma_test_imgs , glaucoma_test_labs)
retina_pred , retina_acc =ensemble(args.model_root_dir , retina_test_imgs , retina_test_labs)
normal_pred , normal_acc =ensemble(args.model_root_dir , normal_test_imgs , normal_test_labs)
"""




"""
    if args.model_dir==None or args.model_dir==None:
        print 'args 1 : image and label paths folder to load '
        print 'args 2 : model folder to load '
        exit()
    else:
        folder_path = args.path_dir
        model_path = args.model_dir
    files=glob.glob(folder_path+'*.txt')
    eval_from_numpy_image(path_dir=args.path_dir , model_dir=args.model_dir)
"""


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



