import numpy as np
from PIL import Image
import utils
import random

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
    for h in range(n_h_move):
        for w in range(n_w_move):
            cropped_images.append(image[ h : h+crop_height , w : w+crop_width , :])

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

def central_eval(image , crop_height , crop_width ,lr_flip =False, ud_flip=False ):
    cropped_images=[]
    up_left_crop_image=image[:crop_height , :crop_width , :]
    up_right_crop_image = image[:crop_height, : -crop_width, :]
    cropped_images.append(up_left_crop_image)
    cropped_images.append(up_right_crop_image)
    cropped_images=np.asarray(cropped_images)
    utils.plot_images(cropped_images)

def sparse_eval(image , crop_height , crop_width ,lr_flip =False, ud_flip=False ):
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
    ori_cropped_images=[]
    up_left_crop_image = image[:crop_height, :crop_width, :]
    up_right_crop_image = image[:crop_height, -crop_width: , :]
    h_gap=(h - crop_height) / 2
    w_gap = (w - crop_width) / 2
    central_crop_image = image[ h_gap : h_gap + crop_height , w_gap : w_gap + crop_width , :]
    down_left_crop_image = image[-crop_height:, :crop_width, :]
    down_right_crop_image = image[-crop_height:, -crop_width:, :]

    ori_cropped_images.append(up_left_crop_image )
    ori_cropped_images.append(up_right_crop_image)
    ori_cropped_images.append(central_crop_image)
    ori_cropped_images.append(down_left_crop_image)
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
    utils.plot_images(cropped_images)
    return cropped_images


if __name__ == '__main__':
    img=Image.open('./debug/0.png')
    img=np.asarray(img)
    #images=dense_crop(img , 224 ,224 , lr_flip=True  , ud_flip=True )
    images=sparse_eval(img , 224 ,224 , lr_flip=True  , ud_flip=True )

    print 'image shape : {}'.format(np.shape(images))
    indices=random.sample(range(len(images)), 60)
    utils.plot_images(images[indices] , random_order=True)



