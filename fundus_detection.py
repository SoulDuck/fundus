#-*- coding: utf-8 -*-
from PIL import Image
import PIL
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def get_roi_from_mask_image(ori_image , mask_image):
    print 'a'
    print np.shape(mask_image)

    masked_image=np.zeros([1001,1001,3])
    ori_image_r= ori_image[:,:,0] * mask_image
    ori_image_g = ori_image[:, :, 1] * mask_image
    ori_image_b = ori_image[:, :, 2] * mask_image
    #tmp_img = np.concatenate([ori_image_r.reshape([1,1001, 1001]), ori_image_g.reshape([1,1001, 1001]),
    #                               ori_image_b.reshape([1,1001, 1001])], axis=0)

    masked_image[:, :, 0] = ori_image_r
    masked_image[:, :, 1] = ori_image_g
    masked_image[:, :, 2] = ori_image_b

    masked_image=np.reshape(masked_image/255. , (1001,1001,3))
    print np.shape(masked_image)

    plt.imshow(masked_image)
    plt.show()
    exit()
    print np.shape(ori_image_r)
    #ori_image_g = ori_image[:, :, 1:2] * mask_image
    #ori_image_b= ori_image[:, :, 2:3] * mask_image

    #ori_img=np.concatenate((ori_image_r , ori_image_g , ori_image_b) , axis=2)
    plt.imshow(ori_image_r , cmap='Greys')
    plt.show()

def get_roi_from_csv(ori_image, mask_csv_path, root_data_dir):
    """
    함수 설명:

    get roi and saved masekd image
    :param ori_image:
    :param mask_csv_path:
    :param data_dir
    :return:
    """
    f = open(mask_csv_path)
    for i, line in enumerate(f.readlines()):
        if i == 0:
            continue
        roi = map( int ,map(float, line.replace('\r\n', '').split(','))) # string -->float -->int
        roi_class = int(roi[0])
        roi_coord = roi[1:]
        start_w , start_h= roi_coord[0],roi_coord[1]
        end_w, end_h = roi_coord[2], roi_coord[3]
        gap_h = roi_coord[2] - roi_coord[0]
        gap_w = roi_coord[3] - roi_coord[1]
        cropped_img=ori_image[start_h : end_h , start_w : end_w ,:]
        dir_path=os.path.join(root_data_dir,str(roi_class))
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        count=0;
        while(True):
            if gap_w==0 or gap_h ==0:
                break;
            try:
                filename='{}.png'.format(count)
                filepath=os.path.join(dir_path ,filename )
                if not os.path.exists(filepath):
                    plt.imsave(filepath,cropped_img )
                    break;
                count+=1
            except Exception as e :
                print e

def get_all_roi_from_csv(csv_dir, data_dir):
    """

    :param csv_dir: csv 파일이 저장된 장소
    :param data_dir: roi 가 저장될 장소
    :return:
    """
    csv_list=glob.glob(os.path.join(csv_dir,'*.csv')) # get csv list from data_dir
    #./roi/diabetic_roi/csv/4052347_20160920_L.csv --> 4052347_20160920_L
    print 'the number of csv : {}'.format(len(csv_list))
    for csv in csv_list:
        #print csv
        filename=csv.split('/')[-1].split('.')[0] #
        print 'filename : {}'.format(filename)
        ori_img = np.asarray(Image.open('./roi/diabetic_images/{}.png'.format(filename)))
        #data_dir='./roi/masked_images'
        get_roi_from_csv(ori_img, csv ,data_dir) #data_dir 에 저장된다

#get_all_roi_from_csv(csv_dir='./roi/diabetic_roi/csv/' , data_dir='./roi/masked_images' )
"""
def get_normal_image(csv_dir ,data_dir):
    csv_list = glob.glob(csv_dir + '*.csv')  # get csv list from data_dir
    print 'the number of csv : {}'.format(len(csv_list))
    for csv in csv_list:
"""
def get_rectangle_size(data_dir):
    """
    :param data_dir: root dir saved masked images
    :return:
    """
    roi_size_list=[]
    dir_path , subdirs , _=os.walk(data_dir).next()
    subdir_paths = map(lambda subdir: os.path.join(data_dir, subdir), subdirs)
    for path in subdir_paths:
        img_path_list=glob.glob(os.path.join(path, '*.png'))
        roi_size_list.extend( map(lambda path : np.shape(np.asarray(Image.open(path)))[:2] , img_path_list))
        roi_size_list=list(set(roi_size_list))

    for roi_coord in roi_size_list:
        x = roi_coord[0]
        y = roi_coord[1]
        plt.scatter(x, y)
    plt.show()

    return roi_size_list

def merge_roi_size(roi_size_list , n_roi_size):
    pass;
#roi_size_list=get_rectangle_size(data_dir='./roi/masked_images')

csv_dir='/Users/seongjungkim/PycharmProjects/fundus/roi/csv'
data_dir='/Users/seongjungkim/PycharmProjects/fundus/roi/masked_images'
#get_all_roi_from_csv(csv_dir , data_dir)
roi_size_list=get_rectangle_size(data_dir='./roi/masked_images')


