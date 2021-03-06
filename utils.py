import sys,os
import numpy as np
import matplotlib.pyplot as plt
import math
import glob
from PIL import Image
import random
import tensorflow as tf
import pickle
import urllib
import tarfile
import zipfile
"""
import Image

background = Image.open("bg.png")
overlay = Image.open("ol.jpg")

background = background.convert("RGBA")
overlay = overlay.convert("RGBA")

new_img = Image.blend(background, overlay, 0.5)
new_img.save("new.png","PNG")
"""


def concat_all(images , axis):
    for i,image in enumerate(images):
        image=np.asarray(image)
        print np.shape(image)
        if i==0:
            merged_images=image
        else:
            merged_images=np.concatenate([merged_images , image] , axis=axis)
    return merged_images

def make_dir(path_dir):
    debug_flag=True

    if __debug__ ==True:
        print "debug mode : utils.py | make_dir "

    if not os.path.isdir(path_dir):
        os.mkdir(path_dir)
        print path_dir, 'was created!'
    else:
        print path_dir,' is existed!'

def get_name(path):
    name = path.split('/')[-1].split('.')[0]
    return name
def check_overlay_paths(all_paths , src_paths):
    """
    return not overlay image btw param all_paths and src_paths
    :param all_paths:
    :param src_paths:
    :return:
    """
    return_paths=[]
    overlay_paths=[]
    src_names=map(get_name,src_paths)
    for path in all_paths:
        name=path.split('/')[-1].split('.')[0]
        if name in src_names:
            overlay_paths.append(path)
        else:
            return_paths.append(path)
    if __debug__ ==True:
        print 'the number of overlay images : ',len(overlay_paths)
    return return_paths
def compare_images(ori_img, target_img):
    plt.title('debuging')
    fig= plt.figure()
    a=fig.add_subplot(1,2,1)
    a.set_xlabel('original image')
    plt.imshow()

    a = fig.add_subplot(1, 2, 1)
    plt.imshow('changed image')
    plt.show()
def show_progress(i,max_iter):
    msg='\r progress {}/{}'.format(i, max_iter)
    sys.stdout.write(msg)
    sys.stdout.flush()
def plot_images(imgs , names=None , random_order=False , savepath=None):
    h=math.ceil(math.sqrt(len(imgs)))
    fig=plt.figure()

    for i in range(len(imgs)):
        ax=fig.add_subplot(h,h,i+1)
        if random_order:
            ind=random.randint(0,len(imgs)-1)
        else:
            ind=i
        img=imgs[ind]
        plt.imshow(img)
        if not names==None:
            ax.set_xlabel(names[ind])
    if not savepath is None:
        plt.savefig(savepath)
    plt.show()
def open_images(paths):
    imgs=[]
    for path in paths:
        if path.endswith('.npy'):
            img=np.load(path)
        else:
            img=Image.open(path)
            img=np.asarray(img)
        imgs.append(imgs)
    return imgs
def sorted_fundus(paths):
    imgs=open_images(paths)
    n,h,w,c=np.shape(imgs)
    center=(h/2, w/2)
    for img in imgs:
        l_sum=img[:,:center[1],:].sum()
        r_sum=img[:,center[1]:,:].sum()
    if l_sum > r_sum:
        plt.imshow(img)
        plt.show()

def change_mode(image, mode='RGB'):
    #image = Image.fromarray(image * 255)
    image = image.convert(mode)
    return image

def np2img(image):
    try:
        image=Image.fromarray(image)
        return image
    except:
        print 'input value isnt numpy type '
        return image
def np2images(images, save_folder=None , paths = None  , extension='png'):
    debug_flag_lv0=False
    if __debug__ == debug_flag_lv0:
        print 'start debug |utils.py| np2images '
    if len(images)==3:
        h,w,ch=np.shape(images)
        images=images.reshape([1,h,w,ch])
    if save_folder is None:
        images=map(np2img,images)
        plot_images(images)
    else:
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)

        for i,image in enumerate(images):
            plt.imshow(image)
            if paths is None:
                plt.imsave(os.path.join(save_folder,str(i)+'.png') , image)
            else:
                plt.imsave(paths[i],image)
    if __debug__ == debug_flag_lv0:
        print 'end debug |data.py| np2images '


def delete_char_from_paths(folder_path , del_char):
    folder_names=os.walk(folder_path).next()[1]
    for folder_name in folder_names:
        paths=glob.glob(folder_path+folder_name+'/*.png')
        for path in paths:
            new_path=path.replace(del_char ,'')
            os.rename(path , new_path)

    """
    *usage:
        delete_char_from_paths(folder_path='../fundus_data/cropped_macula/' , del_char='*')
        test*.txt --> test.txt
    """
def get_paths_from_text(text_locate):
    f=open(text_locate , 'r')
    lines=f.readlines()
    lines=map(lambda x: x.replace('\n' , '' ) , lines)

    return lines

def save_paths(paths , save_path):
    f=open(save_path)
    for path in paths:
        f.write(path+'\n')
    f.close()




"""
def overlay_images(front_image , back_image):
    try:
        front_image=Image.fromarray(front_image)
    except:
        pass
    try:
        back_image=Image.fromarray(back_image)
    except:
        pass

    back_image.paste(front_image, (0, 0), front_image)
    back_image.show()
    return b_image
"""



def make_log_txt(folder_path):
    f = open(folder_path+'/log.txt', 'a')
    return f

def write_acc_loss(f,train_acc,train_loss,test_acc,test_loss):
    f.write(str(train_acc)+'\t'+str(train_loss)+'\t'+str(test_acc)+'\t'+str(test_loss)+'\n')


def divide_images(images , batch_size):
    debug_flag_lv0=True
    debug_flag_lv1=True
    if __debug__ == debug_flag_lv0:
        print 'debug start | utils.py | divide_images'
    batch_img_list = []
    share = len(images) / batch_size
    # print len(images)
    # print len(labels)
    # print 'share :',share

    for i in range(share + 1):
        if i == share:
            imgs = images[i * batch_size:]
            # print i+1, len(imgs), len(labs)
            batch_img_list.append(imgs)
            if __debug__ == debug_flag_lv1:
                print "######utils.py: divide_images_from_batch debug mode#####"
                print 'total :', len(images), 'batch', i * batch_size, '-', len(images)
        else:
            imgs = images[i * batch_size:(i + 1) * batch_size]
            # print i , len(imgs) , len(labs)
            batch_img_list.append(imgs)
            if __debug__ == debug_flag_lv1:
                print "######utils.py: divide_images_from_batch debug mode######"
                print 'total :', len(images), 'batch', i * batch_size, ":", (i + 1) * batch_size
    return batch_img_list


def divide_images_labels_from_batch(images, labels ,batch_size):
    debug_flag=False

    batch_img_list=[]
    batch_lab_list = []
    share=len(labels)/batch_size
    #print len(images)
    #print len(labels)
    #print 'share :',share

    for i in range(share+1):
        if i==share:
            imgs = images[i*batch_size:]
            labs = labels[i*batch_size:]
            #print i+1, len(imgs), len(labs)
            batch_img_list.append(imgs)
            batch_lab_list.append(labs)
            if __debug__ ==debug_flag:
                print "######utils.py: divide_images_labels_from_batch debug mode#####"
                print 'total :', len(images), 'batch', i*batch_size ,'-',len(images)
        else:
            imgs=images[i*batch_size:(i+1)*batch_size]
            labs=labels[i * batch_size:(i + 1) * batch_size]
           # print i , len(imgs) , len(labs)
            batch_img_list.append(imgs)
            batch_lab_list.append(labs)
            if __debug__ == debug_flag:
                print "######utils.py: divide_images_labels_from_batch debug mode######"
                print 'total :', len(images) ,'batch' ,i*batch_size ,":",(i+1)*batch_size
    return batch_img_list , batch_lab_list
def plot_xs_ys(title,xs_title, ys_title , folder_path, xs ,*arg_ys ):
    plt.xlabel(xs_title)
    plt.ylabel(ys_title)
    plt.title(title)
    for ys in arg_ys:
        ys=list(ys)
        plt.plot(xs, ys)
        #folder_path = './graph/' + file_path.split('/')[-1].split('.')[0]
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    plt.savefig(folder_path +'/'+ys_title)
    plt.close()

def draw_grpah(file_pointer,save_folder ,check_point=50):
    if isinstance(file_pointer , str):
        file_path=file_pointer
        f=open(file_path,'r')

    else:
        f=file_pointer
    lines=f.readlines()
    train_acc_list=[];train_loss_list=[];val_acc_list=[];val_loss_list=[];step_list=[]

    for i,line in enumerate(lines):
        step=i*check_point
        step_list.append(step)
        train_acc, train_loss , val_acc , val_loss=line.split('\t')
        train_acc_list.append(train_acc);train_loss_list.append(train_loss);val_acc_list.append(val_acc);val_loss_list.append(val_loss)
    #folder_path = './graph/' + file_path.split('/')[-1].split('.')[0] #

    plot_xs_ys('Normal Vs Abnormal','Step','Train Accuracy',save_folder,step_list , train_acc_list)
    plot_xs_ys('Normal Vs Abnormal', 'Step', 'Train Loss', save_folder,step_list, train_loss_list )
    plot_xs_ys('Normal Vs Abnormal', 'Step', 'Validation Accuracy', save_folder,step_list, val_acc_list)
    plot_xs_ys('Normal Vs Abnormal', 'Step', 'Validation Loss', save_folder,step_list, val_loss_list)
    plot_xs_ys('Normal Vs Abnormal','Step','Train_Validation Accuracy ',save_folder,step_list, train_acc_list, val_acc_list)
    plot_xs_ys('Normal Vs Abnormal','Step','Train_Validation Loss ',save_folder,step_list, train_loss_list, val_loss_list)
    if __debug__==True:
        print 'the number of steps',len(step_list)
        print 'the number of train accuracy , loss',len(train_acc_list),len(train_loss_list)
        print 'the number of validation accuracy and loss',len(val_acc_list) , len(val_loss_list)
        print 'all graph was saved here',save_folder

def make_folder(root_folder_path , folder_name):
    """
    usage:
    :param root_folder_path:
    :param folder_name:
    :return:
    """
    if not os.path.isdir(root_folder_path+folder_name):
        os.mkdir(root_folder_path+folder_name)
        print root_folder_path+folder_name ,'is made'
    count=0
    w_flag=True
    while w_flag:
        if not os.path.isdir(root_folder_path+folder_name+str(count)):
            os.mkdir(root_folder_path+folder_name+str(count))
            print root_folder_path+folder_name+str(count),'is made'
            w_flag=False
        else:
           count+=1
    return root_folder_path+folder_name+str(count)+'/'


def get_acc(true , pred):
    assert np.ndim(true) == np.ndim(pred) , 'true shape : {} pred shape : {} '.format(np.shape(true) , np.shape(pred))
    if np.ndim(true) ==2:
        true_cls =np.argmax(true , axis =1)
        pred_cls = np.argmax(pred, axis=1)

    tmp=[true_cls == pred_cls]
    acc=np.sum(tmp) / float(len(true_cls))
    return acc


"""-----------------------------------------------------------------------------------------------
                                        TENSORFLOW UTILS
-----------------------------------------------------------------------------------------------"""




def make_saver():
    last_saver=tf.train.Saver(max_to_keep=1)
    best_saver = tf.train.Saver(max_to_keep=100)
    return last_saver , best_saver
def save_model(sess,max_acc, min_loss, acc, loss, step,model_dir , last_saver , best_saver):
    """
    model_dir/last
    model_dir/acc
    model_dir/loss

    :param sess:
    :param max_acc:
    :param min_loss:
    :param acc:
    :param loss:
    :param step:
    :param model_dir:
    :param last_saver:
    :param best_saver:
    :return:
    """
    last_dir=os.path.join(model_dir , 'last' )
    root_best_acc_dir = os.path.join(model_dir, 'acc' )
    root_best_loss_dir = os.path.join(model_dir, 'loss' )
    if not os.path.isdir(last_dir):
        print 'construct last model Saver!'
        os.makedirs(last_dir)
    if not os.path.isdir(root_best_acc_dir):
        print 'construct best acc model Saver!'
        os.makedirs(root_best_acc_dir)
    if not os.path.isdir(root_best_loss_dir):
        print 'construct best loss model Saver!'
        os.makedirs(root_best_loss_dir)

    # if training, acc, loss param not is changed , so onlt last model was saved
    if acc > max_acc:  # best acc
        max_acc = acc
        print 'max acc : {} , model_saved'.format(max_acc)
        best_acc_dir= os.path.join(root_best_acc_dir, 'step_{}_acc_{}'.format(step, max_acc))
        os.mkdir(best_acc_dir)
        best_saver.save(sess=sess,save_path=os.path.join(best_acc_dir,'model'))

    if loss < min_loss:  # best loss
        min_loss = loss
        print 'min loss : {}, model_saved'.format(min_loss)
        best_loss_dir = os.path.join(root_best_loss_dir, 'step_{}_loss_{}'.format(step, min_loss))
        os.mkdir(best_loss_dir)
        best_saver.save(sess=sess,save_path=os.path.join(best_loss_dir , 'model'))

    last_saver.save(sess, save_path=os.path.join(last_dir , 'model'), global_step=step)
    return max_acc, min_loss

def write_acc_loss(summary_writer ,prefix , loss , acc  , step):
    summary = tf.Summary(value=[tf.Summary.Value(tag='loss_{}'.format(prefix), simple_value=float(loss)),
                                tf.Summary.Value(tag='accuracy_{}'.format(prefix), simple_value=float(acc))])
    summary_writer.add_summary(summary, step)


def restore_model(saver,sess,ckpt_dir,type='last'):
    if type=='last':
        if tf.train.get_checkpoint_state(checkpoint_dir=ckpt_dir):
            last_ckpt_filename=tf.train.latest_checkpoint(ckpt_dir, latest_filename=None)
            global_step = int(os.path.basename(last_ckpt_filename).split('-')[1])
            saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
            print '*********************************************'
            print '*            Restore Model                  *'
            print '*           global step : {: <6}            *'.format(global_step)
            print '*********************************************'
        else:
            print 'No Model , initializing global step to 0'
            global_step=0
        return global_step
    elif type =='acc':
        #search best accuracy model at ckpt_dir
        path , subdirs , files =os.walk(ckpt_dir).next()
        max_subdir_name=''
        max_acc=0
        for subdir in (subdirs):
            acc=float(str(subdir).split('_')[-1])
            if acc > max_acc:
                max_subdir_name = subdir
                max_acc=acc
        best_model_path=os.path.join(path , max_subdir_name , 'model')
        saver.restore(sess ,save_path=best_model_path)
        print '*************************************'
        print '*            Best Model             *'
        print '*           acc : {:.4f}            *'.format(max_acc)
        print '*************************************'


    else:
        raise NotImplementedError




def cache(cache_path ,  fn , *args , **kwargs):
    if os.path.exists(cache_path):
            # Load the cached data from the file.
            with open(cache_path, mode='rb') as file:
                obj = pickle.load(file)

            print("- Data loaded from cache-file: " + cache_path)
    else:
        # The cache-file does not exist.
        # Call the function / class-init with the supplied arguments.
        obj = fn(*args, **kwargs)
        # Save the data to a cache-file.
        with open(cache_path, mode='wb') as file:
            pickle.dump(obj, file)
        print("- Data saved to cache-file: " + cache_path)
        return obj

def numpy2pickle(in_path , out_path):
    data = np.load(in_path)
    # Save the data using pickle.
    with open(out_path, mode='wb') as file:
        pickle.dump(data, file)


def donwload(url ,download_dir):
    def _print_download_progress(count, block_size, total_size):
        # Percentage completion.
        pct_complete = float(count * block_size) / total_size
        # Status-message. Note the \r which means the line should overwrite itself.
        msg = "\r- Download progress: {0:.1%}".format(pct_complete)
        # Print it.
        sys.stdout.write(msg)
        sys.stdout.flush()

    if not os.path.isdir(download_dir):
        os.makedirs(download_dir)
    filename = url.split('/')[-1]
    file_path = os.path.join(download_dir, filename)
    if not os.path.exists(file_path):
        print 'downloading ...'
        urllib.urlretrieve(url=url , filename=file_path ,reporthook=_print_download_progress);
        print 'Done'

def extract(file_path , out_dir_path):
    print 'extracting files....'
    if file_path.endswith(".zip"):
        zipfile.ZipFile(file=file_path , mode='r').extractall(out_dir_path)
    if file_path.endswith((".tar.gz" , ".tgz")):
        tarfile.open(name=file_path, mode="r:gz").extractall(out_dir_path)
    print("Done.")

"""----------------------------------------------------------------------------------------------------------------
                                                Tensorflow Record
----------------------------------------------------------------------------------------------------------------"""


def read_one_example( tfrecord_path  , resize ):
    filename_queue = tf.train.string_input_producer([tfrecord_path] , num_epochs=10)
    reader = tf.TFRecordReader()
    _ , serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'raw_image': tf.FixedLenFeature([], tf.string),
        'label' : tf.FixedLenFeature([] , tf.int64)
        })
    image = tf.decode_raw(features['raw_image'], tf.uint8)
    height= tf.cast(features['height'] , tf.int32)
    width = tf.cast(features['width'] , tf.int32)
    label = tf.cast(features['label'] , tf.int32)
    image_shape = tf.stack([height , width , 3 ])
    image=tf.reshape(image ,  image_shape)
    if not resize == None :
        resize_height , resize_width  = resize
        image_size_const = tf.constant((resize_height , resize_width , 3) , dtype = tf.int32)
        image = tf.image.resize_image_with_crop_or_pad(image=image,
                                               target_height=resize_height,
                                               target_width=resize_width)
#    images  = tf.train.shuffle_batch([image ] , batch_size =batch_size  , capacity =30 ,num_threads=3 , min_after_dequeue=10)
    return image,label
"""----------------------------------------------------------------------------------------------------------------
                                                Tensorflow Utils
----------------------------------------------------------------------------------------------------------------"""

def show_tensorflow_op():
    for op in tf.get_default_graph().get_operations():
        print op.name

def get_op_name(op):
    return op.name

def search_best_acc_model(self, model_dir):
    max_acc = 0;
    best_model_name = ''
    list_acc = []
    model_name_list = os.listdir(os.path.join(model_dir, 'best_acc'))
    for dir_name in model_name_list:
        acc = int(dir_name.split('_')[-1])
        if max_acc < acc:
            best_model_name = dir_name
            max_acc = acc
    return best_model_name


if __name__=='__main__':

    pred=[0,0,0,1]
    labels=[0,0,0,1]
    pred=np.array([[0.7 , 0.5],[0.7 , 0]])
    labels=np.array([[1. , 0 ],[1, 0 ]])
    print get_acc(pred,labels)

