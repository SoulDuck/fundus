from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import PIL
import time
import argparse
import random
import glob , sys, os

from multiprocessing import Pool


def crop_resize_fundus(path):
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
    print np.shape(np_img)
    #plt.imshow(crop_img)

    diameter_height = up[1] - left[1]
    diameter_width = left[0] - up[0]

    crop_img = im.crop((up[0], left[1], left[0] + diameter_width, up[1] + diameter_height))
    end_time = time.time()

    if __debug__ == True:
        print end_time - start_time

    return crop_img ,path

def crop_specify_point_and_resize(x,start_pos , end_pos , resize_):
    """
    :param x: x shape has to be 4 dimension
    :param start_pos:
    :param end_pos:
    :return:
    """
    x=np.asarray(x)
    cropped_x=x[start_pos[0] : end_pos[0], start_pos[1]: end_pos[1] , :]

    cropped_x=Image.fromarray(cropped_x)
    cropped_x=cropped_x.resize(resize_ , PIL.Image.ANTIALIAS)
    cropped_x = np.asarray(cropped_x)
    return cropped_x
def optical_crop(path):
    name=path.split('/')[-1].split('.')[0]
    img=Image.open(path)
    if 'L' in path:
        img=crop_specify_point_and_resize(img,(400,500),(1250,1350) , resize_=(299,299))
    else:
        img=crop_specify_point_and_resize(img, (400,1150),(1250,2000), resize_=(299, 299))
    return img , path

def macula_crop(path):
    img = Image.open(path)
    if 'L' in path:
        img = crop_specify_point_and_resize(img, (400, 1150), (1250, 2000), resize_=(299, 299))
    else:
        img = crop_specify_point_and_resize(img, (400, 500), (1250, 1350), resize_=(299, 299))
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


    """usage: fundus optical crop """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir" , help='folder to preprocessing')
    parser.add_argument("--save_dir" , help='folder to save')
    parser.add_argument("--extension" , help='extension')
    args = parser.parse_args()

    if args.dir:
        folder_path=args.dir
    else:
        folder_path='./sample_image/original_images/'

    if args.save_dir:
        save_folder=args.save_dir
    else:
        save_folder='./sample_image/processed_images/'

    if args.extension:
        extension = args.extension
    else:
        extension = '*.png'

    paths=glob.glob(folder_path + extension)
    saved_extension=extension.replace('*' ,'')
    if __debug__ == True:
        print 'folder_path:', folder_path
        print 'save_folder:', save_folder
        print 'number of paths' , len(paths)
        print 'extension', extension
        print 'saved extension', saved_extension

    pool=Pool()
    for img , path in pool.imap(macula_crop ,paths):
        save_img(img, save_folder , extension)

    """
    #########   usage : crop_reisize_fundus   #########

    target_folder='./sample_image/'
    extension='*.png'
    paths=glob.glob(target_folder+extension)
    pool=Pool()
    count=0
    for img , path in pool.imap(crop_resize_fundus , paths):
        print np.shape(img)
        save_folder='./'
        name=path.split('/')[-1]
        save_path=os.path.join(save_folder , name)
        reshape_img_size=(228,228)
        img =img.resize(reshape_img_size, PIL.Image.ANTIALIAS)
        img.save(save_path)
    """
