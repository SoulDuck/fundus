#-*- coding:utf-8 -*-

import glob, os, sys
import numpy as np
import utils
from PIL import Image
import random
import matplotlib.pyplot as plt
from multiprocessing import Pool
import aug
import tensorflow as tf



def save_paths(src_paths, f_path):
    f = open(f_path, 'w')
    for path in src_paths:
        f.write(path + '\n')


def cls2onehot(cls, depth):
    debug_flag=False
    if not type(cls).__module__ == np.__name__:
        cls=np.asarray(cls)
    cls=cls.astype(np.int32)
    debug_flag = False
    labels = np.zeros([len(cls), depth] , dtype=np.int32)
    for i, ind in enumerate(cls):
        labels[i][ind:ind + 1] = 1
    if __debug__ == debug_flag:
        print '#### data.py | cls2onehot() ####'
        print 'show sample cls and converted labels'
        print cls[:10]
        print labels[:10]
        print cls[-10:]
        print labels[-10:]
    return labels


def make_paths(folder_path, extension, f_name):
    paths = glob.glob(folder_path + extension)
    if os.path.isfile(f_name):
        print 'paths is already made. this function will be closed'
        f = open(f_name, 'r')
        paths = [path.replace('\n', '') for path in f.readlines()]
        return paths
    else:
        f = open(f_name, 'w')
        for path in paths:
            f.write(path + '\n')
    return paths


def next_batch(imgs, labs, batch_size):
    indices = random.sample(range(np.shape(imgs)[0]), batch_size)
    if not type(imgs).__module__ == np.__name__:  # check images type to numpy
        imgs = np.asarray(imgs)
    imgs = np.asarray(imgs)
    batch_xs = imgs[indices]
    batch_ys = labs[indices]
    return batch_xs, batch_ys


def make_numpy_images_labels(paths, label_num):
    n_paths = len(paths)
    labels = np.zeros([n_paths])
    labels.fill(label_num)
    tmp = []
    for i, path in enumerate(paths):
        try:
            img = Image.open(path)
        except IOError as ioe:
            continue
        img = np.asarray(img)
        if i == 0:
            print np.shape(np.shape(img))
        utils.show_progress(i, len(paths))
        # print np.shape(img)
        tmp.append(img)
    imgs = np.asarray(tmp)
    return imgs, labels


def open_Image(path):
    try:
        img = Image.open(path)
        img = np.asarray(img)
    except Exception as e:
        print str(e)
        img = 'None'
    return img, path


def multiproc_make_numpy_images_labels(paths, label_num):
    count = 0
    pool = Pool()
    n_paths = len(paths)
    labels = np.zeros([n_paths])
    labels.fill(label_num)
    h, w, ch = np.shape(Image.open(paths[0]))
    images = []

    for img, path in pool.imap(open_Image, paths):
        if img is 'None':
            continue
        utils.show_progress(count, len(paths))
        images.append(img)
        count += 1
    print ''
    print 'images shape', np.shape(images)
    pool.close()
    pool.join()
    return images, labels


def get_train_test_paths(test_ratio, *pathss):
    all_train_paths = []
    all_test_paths = []

    def fn(path):
        path = path.replace('\n', '')
        return path

    for i, paths in enumerate(pathss):
        f = open(paths)
        lines = f.readlines()
        n_lines = len(lines)
        lines = map(fn, lines)  # erase 'n'
        random.shuffle(lines)  # shuffle list
        n_test = int(n_lines * test_ratio)
        n_train = n_lines - n_test
        all_train_paths.extend(lines[:n_train])
        all_test_paths.extend(lines[n_train:])
        if __debug__ == True:
            print 'sample type:', lines[0].split('/')[-2]
            print 'total paths:', n_lines, 'train:', n_train, 'test', n_test
            print '##########################################################'
    if __debug__ == True:
        print 'all train paths :', len(all_train_paths)
        print 'all_test_paths :', len(all_test_paths)
        print ''
    return all_train_paths, all_test_paths


def get_train_test_images_labels(normal_images, abnormal_images, train_ratio=0.95):
    NORMAL_LABEL = 0
    ABNORMAL_LABEL = 1
    n_normal = len(normal_images)
    n_normal_train = int(n_normal * train_ratio)
    n_normal_test = n_normal - n_normal_train

    normal_indices = random.sample(range(n_normal), n_normal)
    normal_train_indices = normal_indices[:n_normal_train]
    normal_test_indices = normal_indices[n_normal_train:]

    normal_train_images = normal_images[normal_train_indices]
    normal_test_images = normal_images[normal_test_indices]

    n_abnormal = len(abnormal_images)
    n_abnormal_train = int(n_abnormal * train_ratio)
    n_abnormal_test = n_abnormal - n_abnormal_train

    abnormal_indices = random.sample(range(n_abnormal), n_abnormal)
    abnormal_train_indices = abnormal_indices[:n_abnormal_train]
    abnormal_test_indices = abnormal_indices[n_abnormal_train:]

    abnormal_train_images = abnormal_images[abnormal_train_indices]
    abnormal_test_images = abnormal_images[abnormal_test_indices]

    train_images = np.concatenate((normal_train_images, abnormal_train_images))
    test_images = np.concatenate((normal_test_images, abnormal_test_images))

    labels_normal_train = np.zeros([n_normal_train])
    labels_normal_train.fill(NORMAL_LABEL)
    labels_normal_test = np.zeros([n_normal_test])
    labels_normal_test.fill(NORMAL_LABEL)

    labels_abnormal_train = np.zeros([n_abnormal_train])
    labels_abnormal_train.fill(ABNORMAL_LABEL)
    labels_abnormal_test = np.zeros([n_abnormal_test])
    labels_abnormal_test.fill(ABNORMAL_LABEL)

    train_labels = np.concatenate(labels_normal_train, labels_abnormal_train)
    test_labels = np.concatenate(labels_normal_test, labels_abnormal_test)
    train_labels = train_labels.astype(np.int32)
    test_labels = test_labels.astype(np.int32)
    train_labels = cls2onehot(train_labels, 2)
    test_labels = cls2onehot(test_labels, 2)

    if __debug__ == True:
        print 'the number of normal data', n_normal
        print 'the normal images shape', normal_images.shape
        print 'the number of normal train data', n_normal_train
        print 'the normal train images shape', normal_train_images.shape
        print 'the number of normal test data', n_normal_test
        print 'the normal test images shape ', normal_test_images.shape

        print 'the number of abnormal data', n_abnormal
        print 'the abnormal images shape', abnormal_images.shape
        print 'the number of abnormal train data', n_abnormal_train
        print 'the abnormal train images shape', abnormal_train_images.shape
        print 'the number of abnormal test data', n_abnormal_test
        print 'the abnormal test images shape', abnormal_test_images.shape

        print 'train images shape', train_images.shape
        print 'test images shape', test_images.shape
        print 'train_labels', train_labels.shape
        print 'test_labels', test_labels.shape

    return train_images, train_labels, test_images, test_labels


def fundus_images(folder_path, reload_folder_path=None ,extension='png',\
                  names=['cataract','glaucoma','retina','retina_glaucoma','retina_cataract','cataract_glaucoma','normal'],\
                  n_tests=[100,100,100,5,5,5,330]
                  ,labels=[0,0,0,0,0,0,1],
                  n_trains=[None , None ,None , None , None ,None ,None]):
    """
    usage:
    :param
    :folder_path: e.g) ../fundus_data/cropped_optical/
    :reload_paths_folder: glaucoma_test_images.npy saved!
    :extension:
    :n_tests:
    :labels:

    :return:

    :have to read!!
    """
    debug_flag_lv0 = True
    debug_flag_lv1=True
    if __debug__ ==debug_flag_lv0:
        print 'start : fundus | data | fundus_images'

    assert len(names)==len(n_tests)==len(labels) == len(n_trains)
    if extension.startswith('.'):
        extension=extension.replace('.','')

    test_list_file_paths = []
    train_list_file_paths = []
    test_list_imgs_labs = [];
    train_list_imgs_labs = []
    if reload_folder_path is None:
        path,_,files=os.walk(folder_path).next() #e.g)cataract
        folder_paths=map(lambda name : os.path.join(path, name), names) # ./cropped_300x300/cataract
        list_file_paths=map(lambda folder_path : glob.glob(os.path.join(folder_path , '*.'+extension)) , folder_paths)# ./cropped_300x300/cataract/*.png
        for i in range(len(names)): #len(names) ==> 7

            file_paths=list_file_paths[i]
            train_file_paths=file_paths[n_tests[i]:]
            test_file_paths=file_paths[:n_tests[i]]
            test_list_file_paths.append(test_file_paths)
            train_list_file_paths.append(train_file_paths)

            if not n_trains[i] == None:
                indices = random.sample(range(len(train_file_paths)), n_trains[i])
                tmp_lines = []
                for ind in indices:
                    line = train_file_paths[ind]
                    tmp_lines.append(line)
                train_file_paths= tmp_lines


            test_imgs_labs=multiproc_make_numpy_images_labels(test_file_paths, label_num=labels[i])
            train_imgs_labs = multiproc_make_numpy_images_labels(train_file_paths, label_num=labels[i])
            test_list_imgs_labs.append(test_imgs_labs)
            train_list_imgs_labs.append(train_imgs_labs)
            if __debug__ == debug_flag_lv1:

                print 'name :',names[i],'the # of list of train file paths', len(train_file_paths)
                print 'name :',names[i],'the # of list of test file paths' , len(test_file_paths)
    else:

        for i,name in enumerate(names):
            try:
                f_train_paths=open(os.path.join(reload_folder_path,name+'_train_paths.txt'))
                f_test_paths = open(os.path.join(reload_folder_path, name + '_test_paths.txt'))
                train_lines=f_train_paths.readlines()
                test_lines = f_test_paths.readlines()

                if not n_trains[i] == None:
                    indices=random.sample(range(len(train_lines)) , n_trains[i] )
                    tmp_lines=[]
                    for ind in indices:
                        line=train_lines[ind]
                        tmp_lines.append(line)
                    train_lines=tmp_lines


                train_file_paths=map(lambda line: line.replace('\n','') , train_lines)
                test_file_paths = map(lambda line: line.replace('\n', ''), test_lines)
                train_imgs_labs = multiproc_make_numpy_images_labels(train_file_paths, label_num=labels[i])
                test_imgs_labs = multiproc_make_numpy_images_labels(test_file_paths, label_num=labels[i])
                test_list_file_paths.append(test_file_paths)
                train_list_file_paths.append(train_file_paths)
                test_list_imgs_labs.append(test_imgs_labs)
                train_list_imgs_labs.append(train_imgs_labs)

                if __debug__ == debug_flag_lv1:
                    print 'name :', names[i],', the # of list of train file paths :', len(train_file_paths)
                    print 'name :', names[i],', the # of list of test file paths :', len(test_file_paths)

            except IOError as ioe:
                print 'cannot find folder or files'
                break;
    if __debug__ == debug_flag_lv1:
        print '# data type :',len(train_list_file_paths)
        for i,name in enumerate(names):
            print 'name',name , '#train' , len(train_list_file_paths[i]) ,'#test :' , len(test_list_file_paths[i])
            train_imgs , train_labs=train_list_imgs_labs[i]
            test_imgs, test_labs = test_list_imgs_labs[i]
            print 'name',name,  ' #train image shape', np.shape(train_imgs), '#train label shape :',np.shape(train_labs)
            print 'name', name, ' #train image shape', np.shape(test_imgs), '#test label shape :', np.shape(test_labs)


        print 'end : fundus | data | fundus_images'


    return train_list_imgs_labs, test_list_imgs_labs, train_list_file_paths, test_list_file_paths, names



def eye_299x299():
    image_height = 299
    image_width = 299
    image_color_ch = 3
    n_classes = 2

    if os.path.isfile('./train_imgs.npy') and os.path.isfile('./train_labs.npy'):
        train_imgs = np.load('./train_imgs.npy')
        train_labs = np.load('./train_labs.npy')
        test_imgs = np.load('./test_imgs.npy')
        test_labs = np.load('./test_labs.npy')
        return image_height, image_width, image_color_ch, n_classes, train_imgs, train_labs, test_imgs, test_labs

    folder_path = '/home/mediwhale-2/data/resize_eye/'
    cataract_paths = make_paths(folder_path + 'abnormal/cataract/', '*.png', 'cataract_paths')
    retina_paths = make_paths(folder_path + '/abnormal/retina/', '*.png', 'retina_paths')
    glaucoma_paths = make_paths(folder_path + '/abnormal/glaucoma/', '*.png', 'glaucoma_paths')
    normal_paths = make_paths(folder_path + '/normal/', '*.png', 'normal_paths')
    ########################################  setting here ###############################################
    abnormal_paths = []
    abnormal_paths.extend(cataract_paths);
    abnormal_paths.extend(retina_paths)
    abnormal_paths.extend(glaucoma_paths[:len(retina_paths)])
    normal_paths = normal_paths[:len(abnormal_paths) + 100]
    ####################################################################################################
    print 'Image Loading ....'
    abnormal_imgs, abnormal_labels = make_numpy_images_labels(abnormal_paths, label_num=1)
    normal_imgs, normal_labels = make_numpy_images_labels(normal_paths, label_num=0)
    train_imgs, train_labs, test_imgs, test_labs = get_train_test_images_labels(normal_images=normal_imgs,
                                                                                abnormal_images=abnormal_imgs)

    return image_height, image_width, image_color_ch, n_classes, train_imgs, train_labs, test_imgs, test_labs



def fundus_300x300(folder_path='../fundus_data/cropped_original_fundus_300x300/' ,reload_folder_path=None,extension='png',\
    names = ['cataract', 'glaucoma', 'retina', 'retina_glaucoma','retina_cataract', 'cataract_glaucoma', 'normal'], \
    n_tests = [100, 100, 100, 5, 5, 5, 330], labels = [0, 0, 0, 0, 0, 0, 1] , n_trains=[None, None , None , None ,None, None,None] ):

    """
    dir tree
        fundus_data
        |
        fundus
            |
            cropped_macula #299x299
            cropped_optical #299x299
            cropped_original_fundus_300x300 #300x300
    :return:  image_height, image_width, image_color_ch, n_classes, train_list_imgs_labs, test_list_imgs_labs,train_list_file_paths, test_list_file_paths

    """
    debug_flag = True
    train_list_imgs_labs, test_list_imgs_labs, train_list_file_paths, test_list_file_paths, names=\
            fundus_images(folder_path,reload_folder_path ,extension,names,n_tests,labels , n_trains=n_trains)
    n,h,w,ch=np.shape(train_list_imgs_labs[0][0])

    n_classes = 2
    image_height=h
    image_width=w
    image_color_ch=ch

    assert image_height==300 and image_height==300


    if __debug__ == debug_flag:
        print 'image_height', image_height
        print 'image_weight', image_height
        print 'image_color_ch', image_color_ch
        print 'n classes', n_classes
    return image_height, image_width, image_color_ch, n_classes, \
               train_list_imgs_labs, test_list_imgs_labs,train_list_file_paths, test_list_file_paths,names


def macula_299x299():
    debug_flag = True

    n_classes = 2
    cata, glau, retina, normal = fundus_images(folder_path='../fundus_data/cropped_macula/')
    train_imgs_labs = (cata[0], glau[0], retina[0], normal[0])
    test_imgs = np.concatenate((cata[1][0], glau[1][0], retina[1][0], normal[1][0]))
    test_labs = np.concatenate((cata[1][1], glau[1][1], retina[1][1], normal[1][1]))
    test_labs = test_labs.astype(np.int32)
    test_labs = cls2onehot(test_labs, 2)
    image_height, image_width, image_color_ch = np.shape(train_imgs_labs[0][0][0])

    if __debug__ == debug_flag:
        print 'image_height', image_height
        print 'image_weight', image_height
        print 'image_color_ch', image_color_ch
        print 'n classes', n_classes
        print 'test_imgs shape', test_imgs.shape
        print 'test_labs shape', test_labs.shape

    return image_height, image_width, image_color_ch, n_classes, train_imgs_labs, test_imgs, test_labs


def optical_299x299():
    debug_flag = True
    n_classes = 2
    cata, glau, retina, normal = fundus_images(folder_path='../fundus_data/cropped_optical/')
    train_imgs_labs = (cata[0], glau[0], retina[0], normal[0])
    test_imgs = np.concatenate((cata[1][0], glau[1][0], retina[1][0], normal[1][0]))
    test_labs = np.concatenate((cata[1][1], glau[1][1], retina[1][1], normal[1][1]))
    test_labs = test_labs.astype(np.int32)
    test_labs = cls2onehot(test_labs, 2)
    image_height, image_width, image_color_ch = np.shape(train_imgs_labs[0][0][0])

    if __debug__ == debug_flag:
        print 'image_height', image_height
        print 'image_weight', image_height
        print 'image_color_ch', image_color_ch
        print 'n classes', n_classes
        print 'test_imgs shape', test_imgs.shape
        print 'test_labs shape', test_labs.shape

    return image_height, image_width, image_color_ch, n_classes, train_imgs_labs, test_imgs, test_labs

def make_batch(list_imgs_labs ,  nx , names ):
    """
    :param cata_train = (cata_train_imgs , cata_train_labs)
    :param glau_train: = (glau_train_imgs , glau_train_labs)
    :param retina_train =  (retina_train_imgs , retina_train_labs)
    :param normal_train: =  (normal_train_imgs , normal_train_labs)
    :return:
    """
    debug_flag_lv0=False
    debug_flag_lv1=False
    if __debug__ == debug_flag_lv0:
        print 'start : data.py | make_batch '
    try:
        assert len(list_imgs_labs) == len(nx) == len(names)
    except AssertionError:
        print 'the number of params ard different!'
        print len(list_imgs_labs)
        print len(nx)
        print len(names)

    for i,name in enumerate(names):
        train_imgs , train_labs = list_imgs_labs[i]
        train_imgs=np.asarray(train_imgs)
        train_labs=np.asarray(train_labs)
        indices=random.sample(range(len(train_imgs)) , nx[i] )
        #print indices
        if i ==0:
            batch_xs=train_imgs[indices]
            batch_ys=train_labs[indices]
        else:
            batch_xs=np.concatenate([batch_xs , train_imgs[indices]] , axis=0)
            batch_ys=np.concatenate([batch_ys , train_labs[indices]] , axis=0)

    np.asarray(batch_ys)
    batch_ys = batch_ys.astype(np.int32)
    batch_ys = cls2onehot(batch_ys, 2)
    if __debug__ == debug_flag_lv1:
        print '**** make_train_batch ****'
        print 'the number of batch', len(batch_xs)
        print 'the shape of batch xs ', batch_xs.shape
        print 'the shape of batch ys ', batch_ys.shape

    return batch_xs, batch_ys


def get_paths_from_file(filepath):
    f = open(filepath)
    lines = f.readlines()
    newlines = []
    for line in lines:
        line = line.replace('\n', '')
        newlines.append(line)
    return newlines



def reconstruct_tfrecord_rawdata(tfrecord_path, resize=(299, 299)):
    print 'now Reconstruct Image Data please wait a second'
    reconstruct_image = []
    # caution record_iter is generator

    record_iter = tf.python_io.tf_record_iterator(path=tfrecord_path)

    ret_img_list = []
    ret_lab_list = []
    ret_fnames = []
    for i, str_record in enumerate(record_iter):
        example = tf.train.Example()
        example.ParseFromString(str_record)

        height = int(example.features.feature['height'].int64_list.value[0])
        width = int(example.features.feature['width'].int64_list.value[0])

        raw_image = (example.features.feature['raw_image'].bytes_list.value[0])
        label = int(example.features.feature['label'].int64_list.value[0])
        filename = example.features.feature['filename'].bytes_list.value[0]
        filename = filename.decode('utf-8')
        image = np.fromstring(raw_image, dtype=np.uint8)
        image = image.reshape((height, width, -1))
        ret_img_list.append(image)
        ret_lab_list.append(label)
        ret_fnames.append(filename)
    ret_imgs = np.asarray(ret_img_list)

    if np.ndim(ret_imgs) == 3:  # for black image or single image ?
        b, h, w = np.shape(ret_imgs)
        h_diff = h - resize[0]
        w_diff = w - resize[1]
        ret_imgs = ret_imgs[h_diff / 2: h_diff / 2 + resize[0], w_diff / 2: w_diff / 2 + resize[1], :]
    elif np.ndim(ret_imgs) == 4:  # Image Up sacle(x) image Down Scale (O)
        b, h, w, ch = np.shape(ret_imgs)
        h_diff = h - resize[0]
        w_diff = w - resize[1]
        ret_imgs = ret_imgs[:, h_diff / 2: h_diff / 2 + resize[0], w_diff / 2: w_diff / 2 + resize[1], :]
    ret_labs = np.asarray(ret_lab_list)
    ret_imgs = np.asarray(ret_imgs)
    ret_fnames = np.asarray(ret_fnames)
    return ret_imgs, ret_labs, ret_fnames

def type1(tfrecords_dir, onehot=True, resize=(299, 299)):
    """type1  데이터 확인 완료 함 """
    # type1 은 cataract_glaucoma , retina_catarct  , retina_glaucoma을 각각의 카테고리에 맞는 곳에 넣었다
    # 늑 cataract_glacucoma 는 cataract , glaucoma 에 넣었다

    images, labels, filenames = [], [], []
    names = ['normal_0', 'glaucoma', 'cataract', 'retina', 'cataract_glaucoma', 'retina_cataract', 'retina_glaucoma']
    for name in names:
        for type in ['train', 'test']:
            imgs, labs, fnames = reconstruct_tfrecord_rawdata(
                tfrecord_path=tfrecords_dir + '/' + name + '_' + type + '.tfrecord', resize=resize)
            print type, ' ', name
            print 'image shape', np.shape(imgs)
            print 'label shape', np.shape(labs)
            images.append(imgs);
            labels.append(labs), filenames.append(fnames)

    n = len(names)
    train_images, train_labels, train_filenames = [], [], []
    test_images, test_labels, test_filenames = [], [], []

    for i in range(n):
        train_images.append(images[i * 2]);
        train_labels.append(labels[i * 2]);
        train_filenames.append(filenames[i * 2])
        test_images.append(images[(i * 2) + 1]);
        test_labels.append(labels[(i * 2) + 1]);
        test_filenames.append(filenames[(i * 2) + 1])

    train_images, train_labels, train_filenames, test_images, test_labels, test_filenames = \
        map(lambda x: np.asarray(x),
            [train_images, train_labels, train_filenames, test_images, test_labels, test_filenames])

    def _fn1(x, a, b):
        x[a] = np.concatenate([x[a], x[b]], axis=0)  # cata_glau을  cata에 더한다
        return x

    """
    4번은 cataract glaucoma 
    5번은 retina cataract 
    6번은 retina glaucoma 
    1번은 glaucoma
    2번은 cataract
    3번은 retina
    """
    train_images, train_labels, train_filenames = map(lambda x: _fn1(x, 1, 4),
                                                      [train_images, train_labels, train_filenames])
    test_images, test_labels, test_filenames = map(lambda x: _fn1(x, 1, 4), [test_images, test_labels, test_filenames])
    train_images, train_labels, train_filenames = map(lambda x: _fn1(x, 2, 4),
                                                      [train_images, train_labels, train_filenames])
    test_images, test_labels, test_filenames = map(lambda x: _fn1(x, 2, 4), [test_images, test_labels, test_filenames])

    train_images, train_labels, train_filenames = map(lambda x: _fn1(x, 2, 5),
                                                      [train_images, train_labels, train_filenames])  # retina cataract을
    test_images, test_labels, test_filenames = map(lambda x: _fn1(x, 2, 5), [test_images, test_labels, test_filenames])
    train_images, train_labels, train_filenames = map(lambda x: _fn1(x, 3, 5),
                                                      [train_images, train_labels, train_filenames])
    test_images, test_labels, test_filenames = map(lambda x: _fn1(x, 3, 5), [test_images, test_labels, test_filenames])

    train_images, train_labels, train_filenames = map(lambda x: _fn1(x, 1, 6),
                                                      [train_images, train_labels, train_filenames])
    test_images, test_labels, test_filenames = map(lambda x: _fn1(x, 1, 6), [test_images, test_labels, test_filenames])
    train_images, train_labels, train_filenames = map(lambda x: _fn1(x, 3, 6),
                                                      [train_images, train_labels, train_filenames])
    test_images, test_labels, test_filenames = map(lambda x: _fn1(x, 3, 6), [test_images, test_labels, test_filenames])

    for i in range(4):
        print '#', np.shape(train_images[i])
    for i in range(4):
        print '#', np.shape(test_images[i])

    train_labels = train_labels[:4]
    train_filenames = train_filenames[:4]

    test_images = test_images[:4]
    test_labels = test_labels[:4]
    test_filenames = test_filenames[:4]

    train_images, train_labels, train_filenames, test_images, test_labels, test_filenames = \
        map(lambda x: np.concatenate([x[0], x[1], x[2], x[3]], axis=0), \
            [train_images, train_labels, train_filenames, test_images, test_labels, test_filenames])

    print 'train images ', np.shape(train_images)
    print 'train labels ', np.shape(train_labels)
    print 'train fnamess ', np.shape(train_filenames)
    print 'test images ', np.shape(test_images)
    print 'test labels ', np.shape(test_labels)
    print 'test fnames ', np.shape(test_filenames)
    n_classes = 2
    if onehot:
        train_labels = cls2onehot(train_labels, depth=n_classes)
        test_labels = cls2onehot(test_labels, depth=n_classes)

    return train_images, train_labels, train_filenames, test_images, test_labels, test_filenames

def type2(tfrecords_dir, onehot=True, resize=(299, 299) , random_shuffle = True ,limits = [3000 , 1000 , 1000 , 1000] , save_dir_name=None ):
    # normal : 3000
    # glaucoma : 1000
    # retina : 1000
    # cataract : 1000
    train_images, train_labels, train_filenames = [], [], []
    test_images, test_labels, test_filenames = [], [], []

    names = ['normal_0', 'glaucoma', 'cataract', 'retina', 'cataract_glaucoma', 'retina_cataract', 'retina_glaucoma']
    for ind , name in enumerate(names):
        for type in ['train', 'test']:
            imgs, labs, fnames = reconstruct_tfrecord_rawdata(
                tfrecord_path=tfrecords_dir + '/' + name + '_' + type + '.tfrecord', resize=resize)
            print type, ' ', name
            print 'image shape', np.shape(imgs)
            print 'label shape', np.shape(labs)

            if type =='train':
                random_indices = random.sample(range(len(labs)),
                                               len(labs))  # normal , glaucoma , cataract , retina 만 random shuffle 을 한다
                if random_shuffle and ind < 4:
                    print 'random shuffle On : {} limit : {}'.format(name , limits[ind])
                    limit =limits[ind]
                else :
                    limit = None
                train_images.append(imgs[random_indices[:limit]]);
                train_labels.append(labs[random_indices[:limit]]);
                train_filenames.append(fnames[random_indices[:limit]]);
            else :
                test_images.append(imgs);
                test_labels.append(labs);
                test_filenames.append(fnames);
    def _fn1(x, a, b):
        x[a] = np.concatenate([x[a], x[b]], axis=0)  # cata_glau을  cata에 더한다
        return x

    train_images, train_labels, train_filenames = map(lambda x: _fn1(x, 1, 4),
                                                      [train_images, train_labels, train_filenames])
    test_images, test_labels, test_filenames = map(lambda x: _fn1(x, 1, 4), [test_images, test_labels, test_filenames])
    train_images, train_labels, train_filenames = map(lambda x: _fn1(x, 2, 4),
                                                      [train_images, train_labels, train_filenames])
    test_images, test_labels, test_filenames = map(lambda x: _fn1(x, 2, 4), [test_images, test_labels, test_filenames])

    train_images, train_labels, train_filenames = map(lambda x: _fn1(x, 2, 5),
                                                      [train_images, train_labels, train_filenames])  # retina cataract을
    test_images, test_labels, test_filenames = map(lambda x: _fn1(x, 2, 5), [test_images, test_labels, test_filenames])
    train_images, train_labels, train_filenames = map(lambda x: _fn1(x, 3, 5),
                                                      [train_images, train_labels, train_filenames])
    test_images, test_labels, test_filenames = map(lambda x: _fn1(x, 3, 5), [test_images, test_labels, test_filenames])

    train_images, train_labels, train_filenames = map(lambda x: _fn1(x, 1, 6),
                                                      [train_images, train_labels, train_filenames])
    test_images, test_labels, test_filenames = map(lambda x: _fn1(x, 1, 6), [test_images, test_labels, test_filenames])
    train_images, train_labels, train_filenames = map(lambda x: _fn1(x, 3, 6),
                                                      [train_images, train_labels, train_filenames])
    test_images, test_labels, test_filenames = map(lambda x: _fn1(x, 3, 6), [test_images, test_labels, test_filenames])

    for i in range(4):
        print '#', np.shape(train_images[i])
    for i in range(4):
        print '#', np.shape(test_images[i])

    train_labels = train_labels[:4]
    train_filenames = train_filenames[:4]

    test_images = test_images[:4]
    test_labels = test_labels[:4]
    test_filenames = test_filenames[:4]

    train_images, train_labels, train_filenames, test_images, test_labels, test_filenames = \
        map(lambda x: np.concatenate([x[0], x[1], x[2], x[3]], axis=0), \
            [train_images, train_labels, train_filenames, test_images, test_labels, test_filenames])

    print 'train images ', np.shape(train_images)
    print 'train labels ', np.shape(train_labels)
    print 'train fnamess ', np.shape(train_filenames)
    print 'test images ', np.shape(test_images)
    print 'test labels ', np.shape(test_labels)
    print 'test fnames ', np.shape(test_filenames)
    n_classes = 2
    if onehot:
        train_labels = cls2onehot(train_labels, depth=n_classes)
        test_labels = cls2onehot(test_labels, depth=n_classes)
    if not os.path.isdir('./type2'):
        os.mkdir('./type2')
    if not save_dir_name is None:
        os.mkdir(os.path.join('./type2', save_dir_name))
    count=0
    while True:

        if save_dir_name == None:
            f_path='./type2/{}'.format(count)
        else:
            f_path = os.path.join('./type2',save_dir_name, '{}'.format(count))

        if not os.path.isdir(f_path):
            os.mkdir(f_path)
            break;
        else:
            count += 1



    np.save(os.path.join(f_path , 'train_imgs.npy') , train_images)
    np.save(os.path.join(f_path, 'train_labs.npy'), train_labels)
    np.save(os.path.join(f_path, 'train_fnames.npy'), train_filenames)
    return train_images, train_labels, train_filenames, test_images, test_labels, test_filenames

def type3(tfrecords_dir, onehot=True, resize=(299, 299) , random_shuffle = True ,limits = [6000 , 2000 , 2000 , 2000]):
    return type2(tfrecords_dir, onehot=onehot, resize=resize, random_shuffle = random_shuffle ,limits = limits)



if __name__ == '__main__':
    train_images, train_labels, train_filenames, test_images, test_labels, test_filenames=type2('./fundus_300_debug')

