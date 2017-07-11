from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import PIL
import time
import argparse
import random
import glob , sys, os
import utils
from multiprocessing import Pool


def crop_resize_fundus(path):
    debug_flag=False
    """
    file name =1002959_20130627_L.png
    """
    name = path.split('/')[-1]
    start_time = time.time()
    im = Image.open(path)  # Can be many different formats.
    np_img = np.asarray(im)

    mean_pix = np.mean(np_img)

    pix = im.load()
    height, width = im.size  # Get the width and hight of the image for iterating over
    # pix[1000,1000] #Get the RGBA Value of the a pixel of an image
    c_x, c_y = (int(height / 2), int(width / 2))

    for y in range(c_y):
        if sum(pix[c_x, y]) > mean_pix:
            left = (c_x, y)
            break;

    for x in range(c_x):
        if sum(pix[x, c_y]) > mean_pix:
            up = (x, c_y)
            break;

    crop_img = im.crop((up[0], left[1], left[0], up[1]))

    #plt.imshow(crop_img)

    diameter_height = up[1] - left[1]
    diameter_width = left[0] - up[0]

    crop_img = im.crop((up[0], left[1], left[0] + diameter_width, up[1] + diameter_height))
    end_time = time.time()

    if __debug__ == debug_flag:
        print end_time - start_time
        print np.shape(np_img)

    return crop_img ,path

def crop_specify_point_and_resize(x,start_pos , end_pos , resize_ =None):
    """
    :param x: x shape has to be 4 dimension
    :param start_pos:
    :param end_pos:
    :return:
    """
    x=np.asarray(x)
    cropped_x=x[start_pos[0] : end_pos[0], start_pos[1]: end_pos[1] , :]

    cropped_x=Image.fromarray(cropped_x)
    if resize_ !=None:
        cropped_x=cropped_x.resize(resize_ , PIL.Image.ANTIALIAS)
    cropped_x = np.asarray(cropped_x)
    return cropped_x
def macula_crop(path):

    if path.endswith('.npy'):
        img=Image.fromarray(np.load(path))
    else:
        img = Image.open(path)

    if 'L' in path:
        #img=crop_specify_point_and_resize(img,(400,500),(1250,1350) , resize_=(299,299))
        img = crop_specify_point_and_resize(img, (250, 400), (600, 750))  # 750_750 Image cropped_original-image

    else:
        #img=crop_specify_point_and_resize(img, (400,1150),(1250,2000), resize_=(299, 299))
        img = crop_specify_point_and_resize(img, (250, 50), (600, 400))  # 750_750 Image cropped_original-image
    return img , path

def optical_crop(path):
    if path.endswith('.npy'):
        img=Image.fromarray(np.load(path))
    else:
        img = Image.open(path)
    #print np.shape(img)
    #print plt.imshow(img)
    #print plt.show()
    try:
        if 'L' in path:
            #img = crop_specify_point_and_resize(img, (400, 100), (1150, 850), resize_=(1150-400, 850-100))  # cropped_original-image
            #img = crop_specify_point_and_resize(img, (400, 1150), (1250, 2000), resize_=(299, 299)) #original-image
            img = crop_specify_point_and_resize(img, (250, 50), (600, 400))  # 750_750 Image cropped_original-image
            pass
        else:
            #img = crop_specify_point_and_resize(img, (400, 750), (1150, 1500), resize_=(1150-400, 1500-750)) # cropped_original_image
            #img = crop_specify_point_and_resize(img, (400, 500), (1250, 1350), resize_=(299, 299)) #original-image
            img = crop_specify_point_and_resize(img, (250, 400), (600, 750))  # 750_750 Image cropped_original-image
    except Exception as e :
        print e
        print '*error path*:',path

    return img, path

def find_optical(path):
    img = Image.open(path)


def show_progress(i,max_iter):
    msg='\r Progress {0}/{1}'.format(i,max_iter)
    sys.stdout.write(msg)
    sys.stdout.flush()
def save_img(img, save_folder , extension):
    name = path.split('/')[-1].split('.')[0]
    if extension == '.npy':
        np.save(save_folder + name + extension, img)
    else :
        img = Image.fromarray(img)
        plt.imshow(img)
        plt.imsave(save_folder + name + extension, img)

if __name__ == '__main__':
    """
    path='./sample_image/original_images/43203_20140121_L.png'
    path='./sample_image/original_images/43203_20140121_R.png'

    ori_img=Image.open(path)
    cropped_img,cropped_path=optical_crop(path)
    fig= plt.figure()
    a=fig.add_subplot(1,2,1)
    plt.imshow(ori_img)
    a=fig.add_subplot(1,2,2)
    plt.imshow(cropped_img)
    plt.show()
    """


    """usage: fundus optical crop"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir" , help='folder to preprocessing')
    parser.add_argument("--save_dir" , help='folder to save')
    parser.add_argument("--extension" , help='extension')
    parser.add_argument("--limit_paths", help='limit to paths for multiprocessing')
    args = parser.parse_args()


    if args.dir:
        folder_path=args.dir
    else:
        folder_path='../fundus_data/cropped_original_fundus/'

    if args.save_dir:
        save_folder=args.save_dir
    else:
        save_folder='../fundus_data/cropped_optical/'

    if args.extension:
        extension = args.extension
    else:
        extension = '*.png'

    if args.limit_paths:
        limit_paths=args.limit_paths
    else:
        limit_paths=3000


    folder_names=os.walk(folder_path).next()[1]
    saved_extension='.png'
    for folder_name in folder_names:
        target_folder_path=folder_path+folder_name+'/'
        target_save_folder_path = save_folder + folder_name + '/'
        if not os.path.isdir(target_save_folder_path):
            os.mkdir(target_save_folder_path)
            print target_save_folder_path, 'is made'

        paths=glob.glob(target_folder_path+extension)
        saved_paths = glob.glob(target_save_folder_path + '*' + saved_extension)
        paths = utils.check_overlay_paths(paths, saved_paths)  # check overlay paths
        paths = paths[:limit_paths]
        if __debug__ == True:
            print ''
            print '################################ '
            print 'folder_path:', target_folder_path
            print 'save_folder:', target_save_folder_path
            print 'number of paths' , len(paths)
            print 'extension', extension
            print 'saved extension', saved_extension

        pool=Pool()
        count=0
        for img , path in pool.imap(optical_crop ,paths):
            save_img(img, target_save_folder_path , saved_extension) #save image ==> save_folder+name+extension
            utils.show_progress(count , len(paths))
            count+=1



    """usage: fundus macula crop"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", help='folder to preprocessing')
    parser.add_argument("--save_dir", help='folder to save')
    parser.add_argument("--extension", help='extension')
    parser.add_argument("--limit_paths", help='limit to paths for multiprocessing')
    args = parser.parse_args()

    if args.dir:
        folder_path = args.dir
    else:
        folder_path = '../fundus_data/cropped_original_fundus/'

    if args.save_dir:
        save_folder = args.save_dir
    else:
        save_folder = '../fundus_data/cropped_macula/'

    if args.extension:
        extension = args.extension
    else:
        extension = '*.png'

    if args.limit_paths:
        limit_paths=args.limit_paths
    else:
        limit_paths=3000

    folder_names = os.walk(folder_path).next()[1]
    saved_extension = '.png'
    for folder_name in folder_names:

        target_folder_path = folder_path + folder_name + '/'
        target_save_folder_path = save_folder + folder_name + '/'
        if not os.path.isdir(target_save_folder_path):
            os.mkdir(target_save_folder_path)
            print target_save_folder_path, 'is made'

        paths = glob.glob(target_folder_path + extension)
        saved_paths = glob.glob(target_save_folder_path + '*' + saved_extension)
        paths = utils.check_overlay_paths(paths, saved_paths)  # check overlay paths
        paths = paths[:limit_paths]
        if __debug__ == True:
            print ''
            print '################################ '
            print 'folder_path:', target_folder_path
            print 'save_folder:', target_save_folder_path
            print 'number of paths', len(paths)
            print 'extension', extension
            print 'saved extension', saved_extension

        pool = Pool()
        count = 0
        for img, path in pool.imap(macula_crop, paths):
            save_img(img, target_save_folder_path, saved_extension)  # save image ==> save_folder+name+extension
            utils.show_progress(count, len(paths))
            count += 1

    #########   usage : crop_reisize_fundus   #########
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", help='folder to preprocessing')
    parser.add_argument("--save_dir", help='folder to save')
    parser.add_argument("--extension", help='extension')
    parser.add_argument("--limit_paths" , help='limit to paths for multiprocessing')
    args = parser.parse_args()

    if args.dir:
        folder_path = args.dir
    else:
        folder_path = '../fundus_data/original_fundus/'

    if args.save_dir:
        save_folder = args.save_dir
    else:
        save_folder = '../fundus_data/cropped_original_fundus/'

    if args.extension:
        extension = args.extension
    else:
        extension = '*.png'

    if args.limit_paths:
        limit_paths=args.limit_paths
    else:
        limit_paths=3000


    folder_names = os.walk(folder_path).next()[1]
    saved_extension = '.png'

    for folder_name in folder_names:
        target_folder_path = folder_path + folder_name + '/'
        target_save_folder_path = save_folder + folder_name + '/'
        if not os.path.isdir(target_save_folder_path):
            os.mkdir(target_save_folder_path)
            print target_save_folder_path,'is made'

        paths = glob.glob(target_folder_path + extension)
        saved_paths = glob.glob(target_save_folder_path + '*' + saved_extension)
        paths = utils.check_overlay_paths(paths, saved_paths)  # check overlay paths

        paths=paths[:limit_paths]
        pool = Pool()
        count = 0

        if __debug__ == True:
            print ''
            print '################################ '
            print 'folder_path:', target_folder_path
            print 'save_folder:', target_save_folder_path
            print 'number of paths', len(paths)
            print 'extension', extension
            print 'saved extension', saved_extension
        if len(paths)==0:
            continue;
        for img, path in pool.imap(crop_resize_fundus, paths):
            utils.show_progress(count,len(paths))
            name = path.split('/')[-1]
            save_path = os.path.join(target_save_folder_path, name)
            reshape_img_size = (750, 750)
            img = img.resize(reshape_img_size, PIL.Image.ANTIALIAS)
            img.save(save_path + saved_extension)
            count+=1
    print 'fundus_processing.py out'

"""
