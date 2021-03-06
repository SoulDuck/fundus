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

def get_activation_map(model_dir,image , filename):
    debug_flag=True
    if __debug__ == debug_flag:
        print "debug : fundus_eval.py | get_activation_map"

    try:### error contor
        assert type(image).__module__ == np.__name__##check type if not image
    except AssertionError as ae:
        image=np.asarray(image)

    try:
        assert len(np.shape(image))==4
    except AssertionError as ae :
        h,w,ch=np.shape(image)
        image = image.reshape([1, h, w, ch])

    save_dir , save_name =os.path.split(filename)
    save_name , extension=os.path.splitext(save_name)
    sess = tf.Session()

    saver = tf.train.import_meta_graph(os.path.join(model_dir , 'best_acc.ckpt.meta'))
    saver.restore(sess, os.path.join(model_dir , 'best_acc.ckpt'))
    tf.get_default_graph()

    accuray = tf.get_default_graph().get_tensor_by_name('accuracy:0')

    x_ = tf.get_default_graph().get_tensor_by_name('x_:0')
    y_ = tf.get_default_graph().get_tensor_by_name('y_:0')
    cam_ = tf.get_default_graph().get_tensor_by_name('classmap_reshape:0')
    top_conv = tf.get_default_graph().get_tensor_by_name('top_conv:0')
    phase_train = tf.get_default_graph().get_tensor_by_name('phase_train:0')
    y_conv = tf.get_default_graph().get_tensor_by_name('y_conv:0')

    vis_abnormal, vis_normal=cam.eval_inspect_cam(sess, cam_ ,top_conv , image , 1 ,x_ , y_ , phase_train , y_conv )

    NORMAL_LABEL = 0
    ABNORMAL_LABEL = 1

    #save Image

    image=np.squeeze(image)
    image=np.uint8(image)
    image=Image.fromarray(image)
    image.save(os.path.join(save_dir,save_name+'_original_image'+extension)) # e.g) extension = '.jpg'
    cmap=plt.get_cmap('jet')
    vis_abnormal=cmap(vis_abnormal)
    plt.imsave(os.path.join(save_dir,save_name+'_actmap_abnormal'+extension), vis_abnormal)

    #open Image
    vis_abnormal=Image.open(os.path.join(save_dir,save_name+'_actmap_abnormal'+extension))
    plt.imshow(vis_abnormal)
    plt.show()

    original_img=Image.open(os.path.join(save_dir,save_name+'_original_image'+extension))
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


def get_actmap_using_all_model(model_root_dir , images , save_root_folder , extension='png'):

    print """ fundus_eval.py : def get_actmap_using_all_model """
    path,sub_dirs ,files=os.walk(model_root_dir).next()
    utils.make_dir(save_root_folder)
    for dir in sub_dirs:

        target_model_dir=os.path.join(model_root_dir , dir)
        target_save_dir=os.path.join(save_root_folder , dir)
        utils.make_dir(target_save_dir)
        for count,image in enumerate(images):
            print '####'
            print count
            print '####'

            save_file_path=os.path.join( target_save_dir, str(count)+'.'+extension)
            get_activation_map( target_model_dir , image , save_file_path )

    utils.make_dir(os.path.join(save_root_folder , 'merge'))#create save_root_folder/merge folder
    n_images=len(images)

    target_save_dir=os.path.join(save_root_folder,'merge')
    for i in range(n_images):
        for dir in sub_dirs:
            target_dir = os.path.join(save_root_folder, dir)
            img=Image.open(os.path.join(target_dir,str(i)+'_actmap_abnormal'+'.'+extension))
            img=img.convert('RGB')
            print np.shape(img)
            np_img=np.asarray(img)
            if i==0:
                merged_img=np_img
            else:
                merged_img+=np_img

        print np.shape(merged_img)
        merged_img=merged_img/n_images
        #merged_img=np.uint8(merged_img)
        merged_img=Image.fromarray(merged_img)
        target_filename='merged_'+str(i)+'.'+extension
        target_filepath=os.path.join(target_save_dir , target_filename)
        print np.shape(merged_img)
        plt.imsave(target_filepath , merged_img)


def eval(model_folder_path , images, labels=None):

    if not model_folder_path.endswith('/'):
        model_folder_path=model_folder_path+'/'

    sess = tf.Session()
    try:
        saver = tf.train.import_meta_graph(model_folder_path+'best_acc.ckpt.meta')
        saver.restore(sess, model_folder_path+'best_acc.ckpt')
    except IOError as ioe:
        print 'in model folder path , there is no best_acc.ckpt or best_acc.ckpt.meta files'
        return

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
        print 'label data not assin '
        pred=sess.run([prediction] , feed_dict={x_:images ,phase_train: False})
        return pred

def eval_multiple_images(model_folder_path , images, labels=None , batch_size= 60):
    debug_flag_lv0=True
    if __debug__ ==debug_flag_lv0:
        print 'debug start | fundus_eval.py | eval_multiple_images '
        print 'input image shape :',np.shape(images)
        print 'input labels', labels
        print 'batch size:',batch_size
        merged_pred=[]
        list_imgs = utils.divide_images(images , batch_size = batch_size)
        for i, imgs in enumerate(list_imgs):
            pred = eval(model_folder_path, imgs)
            merged_pred.extend(pred)
        onehot_pred=np.argmax(merged_pred , axis=0)

        if labels is None:
            return merged_pred
        else:
            cls=np.argmax(labels , axis=0 )
            mean_acc=np.sum(cls==onehot_pred)/float(len(cls))
            return merged_pred , mean_acc








def eval_from_numpy_image(path_dir , model_dir):
    """
    usage:
    numpy file name has to be this shape , e.g ) normal_test_images.npy or normal_train_images.npy
    :param path_dir:
    :param model_dir:
    :return:
    """
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
    debug_flag = True
    if __debug__ == debug_flag:
        print '### debug mode | fundus_eval.py : ensemble | start ###'

    if len(np.shape(labels)) == 1:
        print '***critical error***'
        print 'labels rank one , this functions need onehot-vector'
        raise ValueError

    path, names, files = os.walk(model_root_dir).next()
    print 'the number of model:', len(names)
    count=0
    for name in names[:]:
        print 'model name:' , name
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
            for imgs , labs in list_imgs_labs:
                _ , tmp_pred = eval(target_model, imgs, labs)
                tot_pred.extend(tmp_pred)
            tot_cls=np.argmax(tot_pred , axis=1)
            cls=np.argmax(labels, axis=1)
            acc=np.mean(np.equal(cls, tot_cls))

            if count==0:
                sum_pred=np.asarray(tot_pred)
            else:
                sum_pred+=np.asarray(tot_pred)
            count+=1
    mean_pred=sum_pred/float(count)
    mean_pred=mean_pred.astype(np.float32)

    tot_cls = np.argmax(sum_pred, axis=1)
    cls = np.argmax(labels, axis=1)
    acc = np.mean(np.equal(cls, tot_cls))
    if __debug__ == debug_flag:
        print '### debug mode | fundus_eval.py : ensemble | end ###'
    return acc,mean_pred
    """
    for i, pred in enumerate(np_preds):
        if i == 0:
            pred_sum = pred
        else:
            pred_sum += pred
    pred_mean = pred_sum / len(np_preds)
    """


def ensemble_all(path_dir ,model_root_dir, *names):
    #usage : ensemble_all(args.path_dir , args.model_root_dir , 'cataract' , 'glaucoma' , 'retina' , 'normal')
    for name in names:
        imgs = np.load(os.path.join(path_dir, name+('_test_images.npy')))
        cls= np.load (os.path.join(path_dir ,  name+('_test_labels.npy')))
        labs = data.cls2onehot(cls, depth=2)
        print 'data :',name , '# image length',len(imgs)
        acc, pred = ensemble(model_root_dir, imgs, labs)
        assert len(imgs) == len(labs) == len(cls)
        print name+' predictions:', pred, '\n'+name+' accuracy', acc




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
    debug_flag=True
    if __debug__ == debug_flag:
        print '#####  main func start!  #####'

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_dir" , help='image folder to load')
    parser.add_argument("--model_dir" , help='model folder to load')
    parser.add_argument("--model_root_dir", help='model root folder that saved images')
    args = parser.parse_args()

    """ usage : ensemble_all """
    #ensemble_all(args.path_dir , args.model_root_dir , 'cataract' , 'glaucoma' , 'retina' , 'normal')

    """ usage : get_activation_map"""
    imgs=np.load('./normal_test_0.npy')
    print np.shape(imgs)
    pred=eval('./cnn_model/fundus/0' , imgs  )

    #get_actmap_using_all_model(args.model_root_dir , imgs , './FD_300_actmap' )
    #get_activation_map(args.model_dir , imgs[0]  , './sample_actmap.jpg')





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

    if __debug__ == debug_flag:
        print '#####  main function end!  #####'


