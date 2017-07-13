import glob , os , sys
import numpy as np
import utils
from PIL import Image
import random
import matplotlib.pyplot as plt
from multiprocessing import Pool
import aug

def save_paths(src_paths,f_path):
    f= open(f_path , 'w')
    for path in src_paths:
        f.write(path+'\n')

def cls2onehot(cls, depth):
    debug_flag=False
    labels=np.zeros([len(cls),2])
    for i,ind in enumerate(cls):

        labels[i][ind:ind+1]=1
    if __debug__==debug_flag:
        print 'show sample cls and converted labels'
        print cls[:10]
        print labels[:10]
        print cls[-10:]
        print labels[-10:]
    return labels
def make_paths(folder_path , extension , f_name):
    paths=glob.glob(folder_path+extension)
    if os.path.isfile(f_name):
        print 'paths is already made. this function will be closed'
        f=open(f_name, 'r')
        paths=[path.replace('\n','') for path in f.readlines()]
        return paths
    else:
        f=open(f_name, 'w')
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
def open_Image(path):
    try:
        img = Image.open(path)
        img = np.asarray(img)
    except IOError as ioe:
        print str(ioe)
        img=None
    return img, path
def multiproc_make_numpy_images_labels(paths , label_num):

    count=0
    pool=Pool()
    n_paths=len(paths)
    labels=np.zeros([n_paths])
    labels.fill(label_num)
    tmp=[]


    for img,path in pool.imap(open_Image , paths):
        if img ==None:
            continue
        utils.show_progress(count, len(paths))
        tmp.append(img)
        count+=1
    print np.shape(tmp)
    imgs=np.asarray(tmp)
    pool.close()
    pool.join()
    return imgs , labels

def get_train_test_paths(test_ratio,*pathss):

    all_train_paths=[]
    all_test_paths=[]
    def fn(path):
        path=path.replace('\n', '')
        return path

    for i,paths in enumerate(pathss):
        f = open(paths)
        lines = f.readlines()
        n_lines = len(lines)
        lines = map(fn, lines) # erase 'n'
        random.shuffle(lines) # shuffle list
        n_test=int(n_lines*test_ratio)
        n_train=n_lines-n_test
        all_train_paths.extend(lines[:n_train])
        all_test_paths.extend(lines[n_train:])
        if __debug__ ==True:
            print 'sample type:', lines[0].split('/')[-2]
            print 'total paths:', n_lines, 'train:',n_train , 'test',n_test
            print '##########################################################'
    if __debug__==True:
        print 'all train paths :', len(all_train_paths)
        print 'all_test_paths :', len(all_test_paths)
        print ''
    return all_train_paths , all_test_paths
def get_train_test_images_labels(normal_images,abnormal_images, train_ratio=0.95):
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

    return image_height, image_width, image_color_ch, n_classes, train_imgs, train_labs, test_imgs, test_labs



def fundus_images(folder_path):
    debug_flag=False
    """
    usage:
    :param folder_path:
    :return:
    """

    cataract_paths=make_paths(folder_path+'cataract/', '*.png', folder_path+'cataract/'+'cataract_paths.txt') #no random shuffle
    retina_paths = make_paths(folder_path+'retina/', '*.png', folder_path+'retina/'+'retina_paths.txt')
    glaucoma_paths = make_paths(folder_path+'glaucoma/', '*.png', folder_path+'/glaucoma/'+'glaucoma_paths.txt')
    normal_paths = make_paths(folder_path+'normal/', '*.png', folder_path+'/normal/'+'normal_paths.txt')

    cata_train_paths, cata_test_paths = get_train_test_paths(0.5, folder_path+'cataract/'+'cataract_paths.txt') # random shuffle here
    glau_train_paths, glau_test_paths = get_train_test_paths(0.5, folder_path+'retina/'+'retina_paths.txt')
    retina_train_paths, retina_test_paths = get_train_test_paths(0.5, folder_path+'/glaucoma/'+'glaucoma_paths.txt')
    normal_train_paths, normal_test_paths = get_train_test_paths(0.5, folder_path+'/normal/'+'normal_paths.txt')

    save_paths(cata_train_paths,folder_path+'cataract/'+'cataract_train_paths.txt') ;save_paths(cata_test_paths,folder_path+'cataract/'+'cataract_test_paths.txt')
    save_paths(glau_train_paths , folder_path+'glaucoma/'+'glaucoma_train_paths.txt') ;save_paths(glau_test_paths,folder_path+'glaucoma/'+'glaucoma_test_paths.txt')
    save_paths(retina_train_paths,folder_path+'retina/'+'retina_train_paths.txt');save_paths(retina_test_paths,folder_path+'retina/'+'retina_test_paths.txt')
    save_paths(normal_train_paths,folder_path+'normal/'+'normal_train_paths.txt');save_paths(normal_test_paths,folder_path+'normal/'+'normal_test_paths.txt')

    ########################################  setting here ###############################################

    ####################################################################################################
    print 'Image Loading ....'
    cata_train = multiproc_make_numpy_images_labels(cata_train_paths , label_num=1)
    glau_train = multiproc_make_numpy_images_labels(glau_train_paths , label_num=1)
    retina_train = multiproc_make_numpy_images_labels(retina_train_paths, label_num=1)
    normal_train = multiproc_make_numpy_images_labels(normal_train_paths, label_num=0)

    cata_test = multiproc_make_numpy_images_labels(cata_test_paths , label_num=1)
    glau_test = multiproc_make_numpy_images_labels(glau_test_paths , label_num=1)
    retina_test = multiproc_make_numpy_images_labels(retina_test_paths, label_num=1)
    normal_test = multiproc_make_numpy_images_labels(normal_test_paths, label_num=0)


    if __debug__ ==debug_flag:
        print ''
        print '# cataract :', len(cataract_paths)
        print '# glaucoma :', len(glaucoma_paths)
        print '# retina :', len(retina_paths)
        print '# normal :', len(normal_paths)

        print '# cataract train , :', len(cata_train_paths) , '# cataract test :', len(cata_test_paths)
        print '# glaucoma train , :', len(glau_train_paths) , '# glaucoma test :', len(glau_test_paths)
        print '# retina train , :', len(retina_train_paths) , '# retina test :', len(retina_test_paths)
        print '# normal train , :', len(normal_train_paths) , '# normal test :', len(normal_test_paths)

        #print cata_train_paths
        print 'shape of cata_train_imgs' , cata_train[0].shape
        fig = plt.figure()
        a=fig.add_subplot(1,2,1)
        a.set_xlabel('')
        plt.imshow(cata_train[0][0])
        a = fig.add_subplot(1, 2, 2)
        a.set_xlabel('')
        plt.imshow(cata_train[0][1])
        plt.show()

    return [cata_train  , cata_test , cata_train_paths , cata_test_paths] ,\
            [glau_train , glau_test , glau_train_paths , glau_test_paths],\
            [retina_train , retina_test , retina_train_paths , retina_test_paths],\
            [normal_train , normal_test , normal_train_paths , normal_test_paths]

def macula_299x299():
    debug_flag=True

    n_classes=2
    cata,glau,retina,normal=fundus_images(folder_path='../fundus_data/cropped_macula/')
    train_imgs_labs=(cata[0], glau[0], retina[0], normal[0])
    test_imgs=np.concatenate((cata[1][0],glau[1][0],retina[1][0],normal[1][0]))
    test_labs = np.concatenate((cata[1][1], glau[1][1], retina[1][1], normal[1][1]))
    test_labs=test_labs.astype(np.int32)
    test_labs=cls2onehot(test_labs,2)
    image_height , image_width , image_color_ch=np.shape(train_imgs_labs[0][0][0])

    if __debug__==debug_flag:
        print 'image_height',image_height
        print 'image_weight', image_height
        print 'image_color_ch', image_color_ch
        print 'n classes' , n_classes
        print 'test_imgs shape', test_imgs.shape
        print 'test_labs shape', test_labs.shape

    return image_height, image_width, image_color_ch, n_classes, train_imgs_labs, test_imgs , test_labs


def optical_299x299():
    debug_flag=True
    n_classes=2
    cata,glau,retina,normal=fundus_images(folder_path='../fundus_data/cropped_optical/')
    train_imgs_labs=(cata[0], glau[0], retina[0], normal[0])
    test_imgs=np.concatenate((cata[1][0],glau[1][0],retina[1][0],normal[1][0]))
    test_labs = np.concatenate((cata[1][1], glau[1][1], retina[1][1], normal[1][1]))
    test_labs=test_labs.astype(np.int32)
    test_labs=cls2onehot(test_labs,2)
    image_height , image_width , image_color_ch=np.shape(train_imgs_labs[0][0][0])

    if __debug__==debug_flag:
        print 'image_height',image_height
        print 'image_weight', image_height
        print 'image_color_ch', image_color_ch
        print 'n classes' , n_classes
        print 'test_imgs shape', test_imgs.shape
        print 'test_labs shape', test_labs.shape

    return image_height, image_width, image_color_ch, n_classes, train_imgs_labs, test_imgs , test_labs


def make_train_batch(cata_train , glau_train , retina_train , normal_train):
    """

    :param cata_train = (cata_train_imgs , cata_train_labs)
    :param glau_train: = (glau_train_imgs , glau_train_labs)
    :param retina_train =  (retina_train_imgs , retina_train_labs)
    :param normal_train: =  (normal_train_imgs , normal_train_labs)
    :return:
    """
    debug_flag=False
    cata_batch=7
    glau_batch=13
    retina_batch=10
    normal_batch=30
    n_batch=cata_batch+glau_batch+retina_batch+normal_batch
    cata_xs, cata_ys=next_batch(cata_train[0],cata_train[1],cata_batch)
    glau_xs, glau_ys = next_batch(glau_train[0], glau_train[1], glau_batch)
    retina_xs, retina_ys=next_batch(retina_train[0],retina_train[1],retina_batch)
    normal_xs, normal_ys=next_batch(normal_train[0],normal_train[1],normal_batch)

    batch_xs =np.concatenate((cata_xs,glau_xs,retina_xs,normal_xs) , axis=0)
    batch_ys=np.concatenate((cata_ys, glau_ys, retina_ys, normal_ys) , axis=0)
    random_indices=random.sample(range(n_batch), n_batch)
    batch_xs = batch_xs[random_indices]
    batch_ys = batch_ys[random_indices]

    np.asarray(batch_ys)
    batch_ys=batch_ys.astype(np.int32)
    batch_ys=cls2onehot(batch_ys,2)
    if __debug__ == debug_flag:
        print '**** make_train_batch ****'
        print 'the number of batch',n_batch
        print 'the shape of batch xs ' , batch_xs.shape
        print 'the shape of batch ys ', batch_ys.shape


    return batch_xs , batch_ys



if __name__ == '__main__':
    image_height, image_width, image_color_ch, n_classes, train_imgs_labs, test_imgs, test_labs=macula_299x299()
    batch_xs , batch_ys= make_train_batch(train_imgs_labs[0] , train_imgs_labs[1],train_imgs_labs[2],train_imgs_labs[3])
    """
    #make_paths('./fundus_data/cropped_optical',)
    cata , glau , retina , normal =fundus_macula_images()
    batch_xs , batch_ys=make_train_batch(cata[0] , glau[0] , retina[0] , normal[0])
    batch_xs = map(aug.random_blur , batch_xs)
    batch_xs = map(aug.random_flip , batch_xs)
    batch_xs = map(aug.random_rotate, batch_xs)

    utils.plot_images(batch_xs)
    """

    """
    cata_train_paths, cata_test_paths = get_train_test_paths('./cataract_paths')
    normal_train_paths, normal_test_paths = get_train_test_paths('./normal_paths')
    glau_train_paths, glau_test_paths = get_train_test_paths('./glaucoma_paths')
    retina_train_paths, retina_test_paths = get_train_test_paths('./retina_paths')
    """


    """usage: get_train_test_paths"""

    """
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