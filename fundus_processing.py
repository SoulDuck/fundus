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
    print np.shape(images)
    bg[: , h_gap : h_gap+ori_img_h , w_gap : w_gap +ori_img_w , :] = images
    print np.shape(bg)
    #print np.shape(padded_images)

    return bg

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


def sparse_crop(image , crop_height , crop_width ,lr_flip =False, ud_flip=False ):
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




if __name__ == '__main__':

    """
    img = Image.open('./normal/43203_20140121_L.png')
    img=red_free_image(img)
    plt.imshow(img/255.)
    plt.show()
    img = Image.open('./normal/43203_20140121_L.png')
    img = green_free_image(img)
    plt.imshow(img / 255.)
    plt.show()
    img = Image.open('./normal/43203_20140121_L.png')
    img = blue_free_image(img)
    plt.imshow(img / 255.)
    plt.show()
    """
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
    """
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
        f=open(target_save_folder_path+'broken_images.txt' , 'w')
        for img , path in pool.imap(optical_crop ,paths):
            if img==None:
                f.write(path+'\n')
                continue;
            save_img(img, target_save_folder_path , saved_extension) #save image ==> save_folder+name+extension
            utils.show_progress(count , len(paths))
            count+=1
    """

    """usage: fundus macula crop"""
    """
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
            if img == None:
                f.write(path + '\n')
                continue;
            save_img(img, target_save_folder_path, saved_extension)  # save image ==> save_folder+name+extension
            utils.show_progress(count, len(paths))
            count += 1
    """
    #########   usage : crop_reisize_fundus   #########
    """
    dir --- subDir_1
                |- aaa.jpg
                |- bbb.jpg
                ...ccc.jpg
                
            subDir_2
            subDir_3
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", help='folder to preprocessing')
    parser.add_argument("--save_dir", help='folder to save')
    parser.add_argument("--extension", help='extension') #'.png'
    parser.add_argument("--limit_paths" , help='limit to paths for multiprocessing')
    args = parser.parse_args()

    folder_path = args.dir
    save_folder = args.save_dir
    extension = args.extension
    limit_paths=args.limit_paths

    folder_names = os.walk(folder_path).next()[1]
    print folder_names
    saved_extension = extension.replace('*','.') #extension = '.jpg'

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
        print len(paths)
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
            reshape_img_size = (600, 600)
            img = img.resize(reshape_img_size, PIL.Image.ANTIALIAS)
            img.save(save_path + saved_extension)
            count+=1
    print 'fundus_processing.py out'


    """usage : fundus resize"""
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", help='folder to preprocessing')
    parser.add_argument("--save_dir", help='folder to save')
    parser.add_argument("--extension", help='extension') #'.png'
    parser.add_argument("--limit_paths" , help='limit to paths for multiprocessing')
    args = parser.parse_args()

    if args.dir:
        folder_path = args.dir
    else:
        folder_path = '../fundus_data/cropped_original_fundus/'

    if args.save_dir:
        save_folder = args.save_dir
    else:
        save_folder = '../fundus_data/cropped_original_fundus_300x300/'

    if args.extension:
        extension = args.extension
    else:
        extension = '*.png'

    if args.limit_paths:
        limit_paths=args.limit_paths
    else:
        limit_paths=3000


    folder_names = os.walk(folder_path).next()[1]
    print folder_names
    saved_extension = extension.replace('*','.') #extension = '.jpg'

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
        print len(paths)
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
        for img, path in pool.imap(image_resize , paths):
            if img ==None:
                continue
            utils.show_progress(count,len(paths))
            name = path.split('/')[-1]
            save_path = os.path.join(target_save_folder_path, name)
            img.save(save_path + saved_extension)
            count+=1
    print 'fundus_processing.py out'
    """