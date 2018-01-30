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


def add_padding(images , padded_images_height ,padded_images_width ):

    """

     _______________
    |Back    ^      |
    |ground  |      |
    |   _________   |
    |<-|         |->|
    |  | ori_img |  |
    |  |_________|  |
    |        |      |
    |        V      |
    |_______________|


    :param images: shape : 4D , (batch , height , width , channel)
    :return: numpy type of images

    """

    b , ori_img_h , ori_img_w , ch =np.shape(images)
    assert (ori_img_h < padded_images_height )and (ori_img_w < padded_images_width)
    bg=np.zeros(shape=(b , padded_images_height  , padded_images_width , ch))

    h_gap = (padded_images_height - ori_img_h) / 2
    w_gap = (padded_images_width - ori_img_w) / 2
    #element wise sum bg with images
    bg[: , h_gap : h_gap+ori_img_h , w_gap : w_gap +ori_img_w , :] = images

    return bg






def dense_crop(image , crop_height , crop_width , lr_flip =False, ud_flip=False):
    """
     _________________
    | ____       ___  |
    ||    |-->->|   | |
    ||____|     |___| |

            ...
      ____       ____
    ||    |-->->|    ||
    ||____|     |____||
    |_________________|



    :param image:
    :param crop_height:
    :param crop_width:
    :param lr_flip:
    :param ud_flip:
    :return:
    """
    cropped_images=[]
    img_h,img_w,ch=np.shape(image)
    n_h_move = img_h - crop_height + 1
    n_w_move = img_w - crop_width + 1
    coords=[]
    for h in range(n_h_move):
        for w in range(n_w_move):
            cropped_images.append(image[ h : h+crop_height , w : w+crop_width , :])
            coords.append([w,h,w+crop_width , h+crop_height])

    ori_cropped_images=np.asarray(cropped_images)
    #indices = random.sample(range(5900), 60)

    if lr_flip:
        lr_flip_cropped_images=np.flip(ori_cropped_images[:, ] , axis=2)
        #utils.plot_images(lr_flip_cropped_images[:60], random_order=True)
        cropped_images=np.vstack((ori_cropped_images , lr_flip_cropped_images))

    if ud_flip:
        ud_flip_cropped_images = np.flip(ori_cropped_images[:, ], axis=1)
        #utils.plot_images(ud_flip_cropped_images[:60], random_order=True)
        cropped_images = np.vstack((cropped_images, ud_flip_cropped_images))

    if lr_flip and ud_flip:
        lr_ud_flip_cropped_images = np.flip(ud_flip_cropped_images, axis=2)
        cropped_images = np.vstack((cropped_images, lr_ud_flip_cropped_images))

    return np.asarray(cropped_images)


def sparse_crop(image , crop_height , crop_width ,lr_flip =False, ud_flip=False):
    """
     _________________
    | ____       ___  |
    ||    |     |   | |
    ||____| ____|___| |
    |      |    |     |
    |      |____|     |
    | ____       ____ |
    ||    |     |    ||
    ||____|     |____||
    |_________________|

    :return:
    """
    h,w,ch=np.shape(image)
    h_gap = (h - crop_height) / 2
    w_gap = (w - crop_width) / 2

    ori_cropped_images=[]
    up_left_crop_image = image[:crop_height, :crop_width, :]
    up_right_crop_image = image[:crop_height, -crop_width: , :]
    up_middle_crop_image = image[:crop_height, w_gap : w_gap + crop_width , :]

    central_left_crop_image = image[h_gap : h_gap + crop_height  ,:crop_width , :  ]
    central_right_crop_image = image[h_gap: h_gap + crop_height, -crop_width:, :]
    central_crop_image = image[ h_gap : h_gap + crop_height , w_gap : w_gap + crop_width , :]

    down_left_crop_image = image[-crop_height:, :crop_width, :]
    down_right_crop_image = image[-crop_height:, -crop_width:, :]
    down_middle_crop_image = image[-crop_height: , w_gap: w_gap + crop_width, :]

    ori_cropped_images.append(up_left_crop_image )
    ori_cropped_images.append(up_middle_crop_image)
    ori_cropped_images.append(up_right_crop_image)

    ori_cropped_images.append(central_left_crop_image)
    ori_cropped_images.append(central_crop_image)
    ori_cropped_images.append(central_right_crop_image)

    ori_cropped_images.append(down_left_crop_image)
    ori_cropped_images.append(down_middle_crop_image)
    ori_cropped_images.append(down_right_crop_image)

    ori_cropped_images=np.asarray(ori_cropped_images)
    if lr_flip:
        lr_flip_cropped_images=np.flip(ori_cropped_images[:, ] , axis=2)
        #utils.plot_images(lr_flip_cropped_images[:60], random_order=True)
        cropped_images=np.vstack((ori_cropped_images , lr_flip_cropped_images))

    if ud_flip:
        ud_flip_cropped_images = np.flip(ori_cropped_images[:, ], axis=1)
        #utils.plot_images(ud_flip_cropped_images[:60], random_order=True)
        cropped_images = np.vstack((cropped_images, ud_flip_cropped_images))

    if lr_flip and ud_flip:
        lr_ud_flip_cropped_images = np.flip(ud_flip_cropped_images , axis=2)
        cropped_images = np.vstack((cropped_images , lr_ud_flip_cropped_images))

    if lr_flip == False and ud_flip ==False:
        cropped_images= ori_cropped_images
    return cropped_images


if __name__ == '__main__':
    img=Image.open('./debug/0.png')
    img=np.asarray(img)
    #images=dense_crop(img , 224 ,224 , lr_flip=True  , ud_flip=True )
    images=sparse_crop(img , 224 ,224 , lr_flip=True  , ud_flip=True )

    print 'image shape : {}'.format(np.shape(images))
    indices=random.sample(range(len(images)), 60)
    utils.plot_images(images[indices] , random_order=True)














def red_free_image(image):
    debug_flag = False
    # if not type(imgs).__module__ == np.__name__:
    try:
        if not type(image).__moduel__ == __name__:
            image=np.asarray(image)
    except AttributeError as attr_error:
        #print attr_error
        image = np.asarray(image)
    h,w,ch = np.shape(np.asarray(image))

    image_r = np.zeros([h,w])
    image_r.fill(0)
    image_g = image[:, :, 1]
    image_b = image[:, :, 2]

    image_r=image_r.reshape([h,w,1])
    image_g = image_g.reshape([h, w, 1])
    image_b = image_b.reshape([h, w, 1])


    image=np.concatenate([image_r , image_g, image_b] , axis=2)
    if __debug__ == debug_flag:
        print 'red_free_image debugging mode '
        print 'image red shape',np.shape(image_r)
        print 'red channel mean',image[:,:,0].mean()
        print 'image shape' , np.shape(np.asarray(image))
    return image



def green_free_image(image):
    # if not type(imgs).__module__ == np.__name__:
    try:
        if not type(image).__moduel__ == __name__:
            image=np.asarray(image)
    except AttributeError as attr_error:
        print attr_error
        image = np.asarray(image)
    h,w,ch = np.shape(np.asarray(image))

    image_r = image[:, :, 0]
    #image_g = image[:, :, 1]
    image_g = np.zeros([h,w])
    image_g.fill(0)
    image_b = image[:, :, 2]

    image_r=image_r.reshape([h,w,1])
    image_g = image_g.reshape([h, w, 1])
    image_b = image_b.reshape([h, w, 1])


    image=np.concatenate([image_r , image_g, image_b] , axis=2)
    if __debug__ == True:
        print 'image green shape',np.shape(image_g)
        print image[:,:,0].mean()
    return image


def blue_free_image(image):
    # if not type(imgs).__module__ == np.__name__:
    try:
        if not type(image).__moduel__ == __name__:
            image=np.asarray(image)
    except AttributeError as attr_error:
        print attr_error
        image = np.asarray(image)
    h,w,ch = np.shape(np.asarray(image))

    image_r = image[:, :, 0]
    image_g = image[:, :, 1]
    #image_b = image[:, :, 2]
    image_b = np.zeros([h,w])
    image_b.fill(0)

    image_r = image_r.reshape([h ,w ,1])
    image_g = image_g.reshape([h, w, 1])
    image_b = image_b.reshape([h, w, 1])


    image=np.concatenate([image_r , image_g, image_b] , axis=2)
    if __debug__ == True:
        print 'image blue shape',np.shape(image_b)
        print image[:,:,0].mean()
        print image[:, :, 1].mean()
        print image[:, :, 2].mean()
    return image

def get_redfree_images(images):
    debug_flag= False
    if __debug__ ==debug_flag:
        print "get_redfree_images debug mode"
        print "image shape is ",np.shape(images)
    imgs = map(red_free_image, images)
    return imgs

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
    try:
        if 'L' in path:
            #img=crop_specify_point_and_resize(img,(400,500),(1250,1350) , resize_=(299,299))
            img = crop_specify_point_and_resize(img, (250, 400), (600, 750))  # 750_750 Image cropped_original-image
        else:
            #img=crop_specify_point_and_resize(img, (400,1150),(1250,2000), resize_=(299, 299))
            img = crop_specify_point_and_resize(img, (250, 50), (600, 400))  # 750_750 Image cropped_original-image
    except Exception as e :
        print e
        print '*error path*:',path
        img = None
        return img , path

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
        img = None
        return img, path

    return img, path

def find_optical(path):
    img = Image.open(path)


"""
def save_img(img, save_folder , extension):
    name = path.split('/')[-1].split('.')[0]
    if extension == '.npy':
        np.save(save_folder + name + extension, img)
    else :
        img = Image.fromarray(img)
        plt.imshow(img)
        plt.imsave(save_folder + name + extension, img)
"""
def image_resize(path):
    try:
        img=Image.open(path)
        img=img.resize((300,300) , PIL.Image.ANTIALIAS)
    except IOError:
        print path ,'has some problem'
        img =None
    return img , path

def overlaps(window_coord , foreground_coord):
    w_x1, w_y1, w_x2, w_y2 = window_coord
    fg_x1,fg_y1,fg_x2,fg_y2 , =foreground_coord
    o_x1 = max(w_x1, fg_x1)  # o --> overlap
    o_x2 = min(w_x2, fg_x2)
    o_y1 = max(w_y1, fg_y1)
    o_y2 = min(w_y2, fg_y2)
    #print window_coord
    #print foreground_coord
    #print o_x1 , o_y1 , o_x2 , o_y2
    o_w = (o_x2 - o_x1)
    o_h = (o_y2 - o_y1)

    if o_w > 0 and o_h > 0:
        area = o_w * o_h
    else:
        area=None
    return area


def get_width_height(coord):
    x1, y1, x2, y2 = coord
    w = (x2 - x1)
    h = (y2 - y1)

    return w , h



if __name__ == '__main__':
    pass;