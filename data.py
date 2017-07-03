import glob , os , sys
import numpy as np
import utils
from PIL import Image
import random
import matplotlib.pyplot as plt



def cls2onehot(cls, depth):
    labels=np.zeros([len(cls),2])
    for i,ind in enumerate(cls):

        labels[i][ind:ind+1]=1
    if __debug__==True:
        print 'show sample cls and converted labels'
        print cls[:10]
        print labels[:10]
        print cls[-10:]
        print labels[-10:]
    return labels
def make_paths(folder_path , extension , f_name):
    paths=glob.glob(folder_path+extension)
    if os.path.isfile('./'+f_name):
        print 'paths is already made. this function will be closed'
        f=open('./'+f_name, 'r')
        paths=[path.replace('\n','') for path in f.readlines()]
        return paths
    else:
        f=open('./'+f_name, 'w')
        for path in paths:
          f.write(path+'\n')
    return paths

def next_batch(imgs, labs , batch_size):
    indices=random.sample(range(np.shape(imgs)[0]) , batch_size)
    batch_xs=imgs[indices]
    batch_ys=labs[indices]
    return batch_xs , batch_ys

def make_numpy_images_labels(paths , label_num):
    n_paths=len(paths)
    labels=np.zeros([n_paths])
    labels.fill(label_num)
    tmp=[]
    for i,path in enumerate(paths):
        try:
            img=Image.open(path)
        except IOError as ioe:
            continue
        img=np.asarray(img)
        if i==0:
            print np.shape(np.shape(img))
        utils.show_progress(i, len(paths))
        #print np.shape(img)
        tmp.append(img)
    imgs=np.asarray(tmp)
    return imgs , labels
def get_train_test_images_labels(normal_images,abnormal_images, train_ratio=0.95 ):

    NORMAL_LABEL=0
    ABNORMAL_LABEL=1
    n_normal=len(normal_images)
    n_normal_train=int(n_normal*train_ratio)
    n_normal_test=n_normal-n_normal_train

    normal_indices=random.sample(range(n_normal) , n_normal)
    normal_train_indices=normal_indices[:n_normal_train]
    normal_test_indices = normal_indices[n_normal_train:]

    normal_train_images=normal_images[normal_train_indices]
    normal_test_images = normal_images[normal_test_indices]


    n_abnormal = len(abnormal_images)
    n_abnormal_train=int(n_abnormal*train_ratio)
    n_abnormal_test=n_abnormal-n_abnormal_train

    abnormal_indices = random.sample(range(n_abnormal),n_abnormal)
    abnormal_train_indices = abnormal_indices[:n_abnormal_train]
    abnormal_test_indices = abnormal_indices[n_abnormal_train:]

    abnormal_train_images = abnormal_images[abnormal_train_indices]
    abnormal_test_images = abnormal_images[abnormal_test_indices]


    train_images=np.concatenate((normal_train_images,abnormal_train_images))
    test_images = np.concatenate((normal_test_images, abnormal_test_images))

    labels_normal_train=np.zeros([n_normal_train])
    labels_normal_train.fill(NORMAL_LABEL)
    labels_normal_test = np.zeros([n_normal_test])
    labels_normal_test.fill(NORMAL_LABEL)

    labels_abnormal_train=np.zeros([n_abnormal_train])
    labels_abnormal_train.fill(ABNORMAL_LABEL)
    labels_abnormal_test = np.zeros([n_abnormal_test])
    labels_abnormal_test.fill(ABNORMAL_LABEL)


    train_labels = np.concatenate(labels_normal_train, labels_abnormal_train)
    test_labels = np.concatenate(labels_normal_test, labels_abnormal_test)
    train_labels=train_labels.astype(np.int32)
    test_labels=test_labels .astype(np.int32)
    train_labels=cls2onehot(train_labels ,2 )
    test_labels = cls2onehot(test_labels, 2)

    if __debug__==True:
        print 'the number of normal data',n_normal
        print 'the normal images shape', normal_images.shape
        print 'the number of normal train data',n_normal_train
        print 'the normal train images shape', normal_train_images.shape
        print 'the number of normal test data',n_normal_test
        print 'the normal test images shape ', normal_test_images.shape

        print 'the number of abnormal data',n_abnormal
        print 'the abnormal images shape', abnormal_images.shape
        print 'the number of abnormal train data',n_abnormal_train
        print 'the abnormal train images shape', abnormal_train_images.shape
        print 'the number of abnormal test data',n_abnormal_test
        print 'the abnormal test images shape', abnormal_test_images.shape

        print 'train images shape',train_images.shape
        print 'test images shape',test_images.shape
        print 'train_labels',train_labels.shape
        print 'test_labels', test_labels.shape

    return train_images, train_labels , test_images , test_labels

def eye_299x299():
    image_height = 299
    image_width = 299
    image_color_ch = 3
    n_classes = 2

    if os.path.isfile('./train_imgs.npy') and os.path.isfile('./train_labs.npy'):

        train_imgs=np.load('./train_imgs.npy')
        train_labs = np.load('./train_labs.npy')
        test_imgs = np.load('./test_imgs.npy')
        test_labs = np.load('./test_labs.npy')
        return image_height, image_width, image_color_ch, n_classes, train_imgs, train_labs, test_imgs, test_labs

    folder_path='/home/mediwhale-2/data/resize_eye/'
    cataract_paths=make_paths(folder_path+'abnormal/cataract/' ,'*.png' , 'cataract_paths')
    retina_paths = make_paths(folder_path+'/abnormal/retina/', '*.png', 'retina_paths')
    glaucoma_paths = make_paths(folder_path+'/abnormal/glaucoma/', '*.png', 'glaucoma_paths')
    normal_paths = make_paths(folder_path+'/normal/', '*.png', 'normal_paths')
    ########################################  setting here ###############################################
    abnormal_paths=[]
    abnormal_paths.extend(cataract_paths);
    abnormal_paths.extend(retina_paths)
    abnormal_paths.extend(glaucoma_paths[:len(retina_paths)])
    normal_paths=normal_paths[:len(abnormal_paths)+100]
    ####################################################################################################
    print 'Image Loading ....'
    abnormal_imgs, abnormal_labels = make_numpy_images_labels(abnormal_paths, label_num=1)
    normal_imgs, normal_labels = make_numpy_images_labels(normal_paths, label_num=0)
    train_imgs , train_labs , test_imgs , test_labs =get_train_test_images_labels(normal_images=normal_imgs , abnormal_images=abnormal_imgs)

    #cata_imgs, cata_labels = make_numpy_images_labels(cataract_paths, label_num=1)
    #cata_imgs , cata_labels= make_numpy_images_labels(cataract_paths, label_num=1)
    #retina_imgs, retina_labels = make_numpy_images_labels(retina_paths, label_num=1)
    #glaucoma_imgs,glaucoma_labels  = make_numpy_images_labels(glaucoma_paths , label_num=1)
    #normal_imgs,normal_labels  = make_numpy_images_labels(normal_paths[:12000] , label_num=0)

    return image_height, image_width, image_color_ch, n_classes, train_imgs, train_labs, test_imgs, test_labs

if __name__ == '__main__':
    image_height, image_width, image_color_ch, n_classes, train_imgs, train_labs, test_imgs, test_labs=eye_299x299()
    plt.imshow(train_imgs[0])
    plt.show()
    plt.imshow(train_imgs[-1])
    plt.show()

    plt.imshow(test_imgs[0])
    plt.show()
    plt.imshow(test_imgs[-1])
    plt.show()

    print train_labs
    print test_labs
    """ 
    cataract_paths=make_paths('/home/mediwhale/data/eye/resize_eye/abnormal/cataract/' ,'*.png' , 'cataract_paths')
    retina_paths = make_paths('/home/mediwhale/data/eye/resize_eye/abnormal/retina/', '*.png', 'retina_paths')
    glaucoma_paths = make_paths('/home/mediwhale/data/eye/resize_eye/abnormal/glaucoma/', '*.png', 'glaucoma_paths')
    normal_paths = make_paths('/home/mediwhale/data/eye/resize_eye/normal/', '*.png', 'normal_paths')
    cata_imgs , cata_labels= make_numpy_images_labels(cataract_paths, label_num=1)
    retina_imgs, retina_labels = make_numpy_images_labels(retina_paths, label_num=1)
    glaucoma_imgs,glaucoma_labels  = make_numpy_images_labels(glaucoma_paths , label_num=1)
    normal_imgs,normal_labels  = make_numpy_images_labels(normal_paths[:12000] , label_num=0)
    get_train_test_images_labels(cata_imgs , retina_imgs)
    """