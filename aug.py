import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import random
from PIL import ImageFilter
import aug
import tensorflow as tf

def check_type_numpy(a):
    if type(a).__module__ ==np.__name__:
        return True
    else:
        return False

def random_rotate(img):
    debug_flag=False
    if check_type_numpy(img):
        if np.max(img)<=1:
            img=img*255.
        img=Image.fromarray(img.astype('uint8'))
    ### usage: map(random_rotate , images) ###
    ind=random.randint(0,180)
    minus = random.randint(0,1)
    minus=bool(minus)
    if minus==True:
        ind=ind*-1
    img=img.rotate(ind)
    img=np.asarray(img)


    #image type is must be PIL
    if __debug__ == debug_flag:
        print ind
        plt.imshow(img)
        plt.show()
    img=img/255.
    return img

def random_rotate_images(images):
    images=np.asarray(map(lambda image : random_rotate(image) , images))
    return images

def random_flip(image):
    debug_flag = False
    if not check_type_numpy(image):
        image=np.asarray(image)
    flipud_flag=bool(random.randint(0,1))
    fliplr_flag = bool(random.randint(0, 1))

    if flipud_flag== True:
        image=np.flipud(image)
    if fliplr_flag==True:
        image = np.fliplr(image)

    if __debug__==debug_flag:
        print 'flip lr ', str(fliplr_flag)
        print 'flip ud ', str(flipud_flag)
    return image
def random_blur(image):
    if check_type_numpy(image):
        image=Image.fromarray(image)
    ind=random.randint(0,10)
    blurred_image = image.filter(ImageFilter.GaussianBlur(radius=ind))
    blurred_image=np.asarray(blurred_image)

    return blurred_image
def aug_level_1(imgs):
    imgs = map(random_blur , imgs)
    imgs = map(random_flip , imgs)
    imgs = map(random_rotate, imgs)
    return imgs

def aug_tensor_images(images , phase_train , img_size_cropped , color_aug=True):
    num_channels=int(images.get_shape()[-1])
    print num_channels


    # This function takes a single image as input,
    # and a boolean whether to build the training or testing graph.

    def _training(image):
        # For training, add the following to the TensorFlow graph.
        # Randomly crop the input image.
        print img_size_cropped
        image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, num_channels])
        print image

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)

        # Randomly adjust hue, contrast and saturation.
        if color_aug:
            print 'hue augmentation On'
            print 'random_contrast augmentation On'
            print 'random_brightness augmentation On'
            print 'random_saturation augmentation On'
            image = tf.image.random_hue(image, max_delta=0.05)
            image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
            image = tf.image.random_brightness(image, max_delta=0.2)
            image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

        # Some of these functions may overflow and result in pixel
        # values beyond the [0, 1] range. It is unclear from the
        # documentation of TensorFlow 0.10.0rc0 whether this is
        # intended. A simple solution is to limit the range.

        # Limit the image pixels between [0, 1] in case of overflow.
        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)
        return image
    def _eval(image):
        print image
        # For training, add the following to the TensorFlow graph.

        # Crop the input image around the centre so it is the same
        # size as images that are randomly cropped during training.
        image = tf.image.resize_image_with_crop_or_pad(image,
                                                       target_height=img_size_cropped,
                                                       target_width=img_size_cropped)
        return image


    # Use TensorFlow to loop over all the input images and call
    # the function above which takes a single image as input.
    #    logit = tf.cond(phase_train, lambda: affine('fc', x, out_ch=self.n_classes, keep_prob=0.5), \
    #                    lambda: affine('fc', x, out_ch=self.n_classes, keep_prob=1.0))

    #images=tf.map_fn(lambda image : _training(image) , images)
    #print images
    ##images = tf.map_fn(lambda image: _pre_process_image(image, phase_train ), images)
    images = tf.map_fn(lambda image : tf.cond(phase_train ,  lambda: _training(image)  , lambda :_eval(image)), images)
    return images



#==== histogram equalization
def histo_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = cv2.equalizeHist(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized


# CLAHE (Contrast Limited Adaptive Histogram Equalization)
#adaptive histogram equalization is used. In this, image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these blocks are histogram equalized as usual. So in a small area, histogram would confine to a small region (unless there is noise). If noise is there, it will be amplified. To avoid this, contrast limiting is applied. If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram equalization. After equalization, to remove artifacts in tile borders, bilinear interpolation is applied
#create a CLAHE object (Arguments are optional).
def clahe_equalized(img):
    assert (len(img.shape)==3)  #4D arrays
    img=img.copy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if img.shape[-1] ==3: # if color shape
        for i in range(3):
            img[:, :, i]=clahe.apply(np.array(img[:,:,i], dtype=np.uint8))
    elif img.shape[-1] ==1: # if Greys,
        img = clahe.apply(np.array(img, dtype = np.uint8))
    return img

# ===== normalize over the dataset
def dataset_normalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized


def adjust_gamma(imgs, gamma=1.0):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i,0] = cv2.LUT(np.array(imgs[i,0], dtype = np.uint8), table)
    return new_imgs




if __name__ == '__main__':
    img=Image.open('./debug/0.png').convert('RGB')
    img=np.asarray(img)
    plt.imshow(img)
    plt.show()
    img=clahe_equalized(img)
    plt.imshow(img)
    plt.show()

    np_img = np.asarray(img)
    img=random_rotate(img)
    np_img=np.asarray(img)

