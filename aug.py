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

def random_rotate(image):
    debug_flag=False
    if check_type_numpy(image):
        img=Image.fromarray(image)
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
    return img

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

def get_redfree_images(images):
    debug_flag= False
    if __debug__ ==debug_flag:
        print "get_redfree_images debug mode"
        print "image shape is ",np.shape(images)
    imgs = map(red_free_image, images)
    return imgs
if __name__ == '__main__':
    img=Image.open('./data/rion.png')
    img=random_rotate(img)
    img=random_flip(img)
    img=random_blur(img)
    #print np.shape(img)
    #img=img.rotate(45)
    #print np.shape(img)
    plt.imshow(img)
    plt.show()
    """
    img=cv2.imread('./data/rion.png',0)
    rows, cols=img.shape
    rotated_img=cv2.getRotationMatrix2D((cols/2, rows/2),90,1)
    img=np.asarray(img )
    img=img/255.
    print img.shape
    plt.imshow(img)
    plt.show()
    plt.imshow(rotated_img)
    plt.show()
    """

"""usage:red free image"""
"""
extension='png'
src_root_folder='../fundus_data/cropped_original_fundus_300x300/'
target_root_folder='../fundus_data/cropped_original_fundus_redfree/'
root_folder, sub_folder_names, file_list=os.walk(src_root_folder).next()
for sub_folder_name in sub_folder_names:
    src_folder=os.path.join(src_root_folder, sub_folder_name)
    saved_folder=os.path.join(target_root_folder , sub_folder_name)
    if not os.path.isdir(saved_folder):
        os.mkdir(saved_folder)
        print saved_folder+'is made'
    paths=glob.glob(src_folder +'/*.'+extension)
    images=map(Image.open , paths[:])
    names=map(lambda x : x.split('/')[-1].split('.')[0] ,paths[:3])
    start_time=time.time()
    redFree_images = map(red_free_image, images[:60])
    print np.shape(redFree_images)
    print  time.time() - start_time
"""


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

    #images =tf.cond(phase_train , tf.map_fn(lambda image : _training(image) , images ) ,\
    #        tf.map_fn(lambda image : _eval(image) , images ))