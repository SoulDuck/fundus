#-*- coding:utf-8 -*-
import numpy as np
import utils
from fundus_processing import overlaps , get_width_height , sparse_crop
from PIL import Image
import matplotlib.pyplot as plt
import os
import glob
import matplotlib.patches as patches
import random
def dense_crop(image , crop_height , crop_width , lr_flip =False, ud_flip=False ):
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
    coords=[]
    cropped_images=[]
    img_h,img_w,ch=np.shape(image)
    n_h_move = img_h - crop_height + 1
    n_w_move = img_w - crop_width + 1
    for h in range(n_h_move):
        for w in range(n_w_move):
            x1 = w
            y1 = h
            x2 = w + crop_width
            y2 = h + crop_height
            coords.append((x1,y1,x2,y2))
            cropped_images.append(image[ h : h+crop_height , w : w+crop_width , :])
    assert len(cropped_images) == len(coords)
    return cropped_images , coords



class Preprocessing(object):
    def __init__(self , csv_dir , img_dir):
        self.csv_dir  = csv_dir
        self.img_dir = img_dir

        # 전체 csv paths 을 training 과 test csv 로 나눈다
        self.csv_paths = glob.glob(os.path.join(self.csv_dir, '*.csv'))
        self.n_paths = len(self.csv_paths)
        self.n_test_paths = int(self.n_paths * 0.1)
        self.n_train_paths = self.n_paths - self.n_test_paths
        self.test_csv_paths = self.csv_paths[:self.n_test_paths]
        self.train_csv_paths = self.csv_paths[self.n_test_paths:]

        # save train path , test path to ./train_path.txt , ./test_path.txt
        self.save_csv_paths()
        self.all_labels=self._get_all_coords() # get all_labels
        self.crop_size=75
#        self.fg_images=self.get_rois(roi_num=1)
#        self.bg_images = self.get_bg(roi_num=1 , num_bg=30)



        #self._get_cropped()

    def get_coords(self, path):
        labels = {}
        f = open(path, 'r')
        lines = f.readlines()
        for i, line in enumerate(lines[1:]):
            label, x1, y1, x2, y2 = map(float, line.split(','))
            if not label in labels.keys():
                labels[int(label)] = [[x1, y1, x2, y2]]  #
            else:
                labels[int(label)].append([x1, y1, x2, y2])  # )
        return labels

    def _get_all_coords(self):
        assert len(self.csv_paths) > 0 , 'the number of csv path {} '.format(len(self.csv_paths))
        self.all_labels={}
        for path in self.train_csv_paths:
            labels=self.get_coords(path)
            for key in labels.keys():
                if not key in self.all_labels.keys():
                    self.all_labels[key] = labels[key]
                else:
                    self.all_labels[key].extend(labels[key])
        return self.all_labels

    def get_rois(self , roi_num , csv_paths):
        roi_coords={}
        roi_images = []
        for path in csv_paths[:]:
            name = os.path.split(path)[1]
            name = os.path.splitext(name)[0]
            labels = self.get_coords(path)  # csv별 roi을 가져온다. 예시 [4] : [[x1,y1 x2, y2] ...[x1,y1 x2, y2]]
            img = np.asarray(Image.open(os.path.join(self.img_dir, name + '.png')))
            for k in labels.keys():
                if k == roi_num:
                     for i, coord in enumerate(labels[k]):
                        if not  name in roi_coords.keys():
                            roi_coords[name]=[coord]
                        else:
                            roi_coords[name].append(coord)
        return roi_coords

    def get_fg(self , roi_num , csv_paths,show=False):
        fg_images=[]
        for path in csv_paths[:]:
            name=os.path.split(path)[1]
            name=os.path.splitext(name)[0]
            print name
            labels = self.get_coords(path)  # csv별 roi을 가져온다. 예시 [4] : [[x1,y1 x2, y2] ...[x1,y1 x2, y2]]
            img=np.asarray(Image.open(os.path.join(self.img_dir, name + '.png')))
            for k in labels.keys():
                if k ==roi_num:
                    for i,fg_coord in enumerate(labels[k]):
                        try:
                            print 'foreground coord : {} , {}'.format(fg_coord,i)

                            fg_x1, fg_y1, fg_x2, fg_y2 = map(int, fg_coord)
                            fg_w = fg_x2 - fg_x1
                            fg_h = fg_y2 - fg_y1
                            fg_area = fg_w * fg_h
                            if fg_area < 225*225:
                                roi_img=img[fg_y1 : fg_y2 , fg_x1: fg_x2]
                                roi_h,roi_w,ch=np.shape(roi_img)
                                if not roi_h > self.crop_size or roi_w > self.crop_size:
                                    h_ = max(roi_h, self.crop_size)
                                    w_ = max(roi_w, self.crop_size)
                                    resized_roi=np.asarray(Image.fromarray(roi_img).resize((h_,w_) , Image.ANTIALIAS))
                                    fg_croppped_imgs = sparse_crop(resized_roi, self.crop_size, self.crop_size)
                                else:
                                    fg_croppped_imgs = sparse_crop(
                                        img[fg_y1 - 10:fg_y2 + 10, fg_x1 - 10L:fg_x2 + 10], self.crop_size, self.crop_size)

                                fg_images.extend(list(fg_croppped_imgs))
                                if show:
                                    utils.plot_images(fg_croppped_imgs)
                                    plt.close()
                                    plt.title(name)
                                    plt.imshow(roi_img)
                                    plt.show()
                                    plt.close()
                        except Exception as e:
                            print 'error coord {}'.format([fg_x1, fg_y1, fg_x2, fg_y2])
        return np.asarray(fg_images)
    def get_bg(self ,roi_num , num_bg , csv_paths):
        bg_images=[]

        for path in csv_paths:
            name=os.path.split(path)[1]
            name=os.path.splitext(name)[0]
            labels = self.get_coords(path)  # csv별 roi을 가져온다. 예시 [4] : [[x1,y1 x2, y2] ...[x1,y1 x2, y2]]
            img=np.asarray(Image.open(os.path.join(self.img_dir, name + '.png')))
            img_h, img_w, img_ch = np.shape(img)
            #이미지에서 임의의 좌표를 찍고 self.crop_size 너비와 높이를 가지는 사각형을 crop한다 , 원래 이미지를 넘어가지 않게 좌표를 잘조정한다
            #하지만 그 좌표가 fg와 겹치면 pass한다

            for i in range(num_bg):
                overlap_flag = False
                bg_y1=random.randint(0 , img_h-self.crop_size)
                bg_x1 = random.randint(0, img_w - self.crop_size)
                bg_y2 = bg_y1 + self.crop_size
                bg_x2=bg_x1+self.crop_size
                bg_coord=[bg_x1, bg_y1, bg_x2, bg_y2]

                for k in labels.keys():
                    if k ==roi_num:
                        for i,fg_coord in enumerate(labels[k]):
                            if not overlaps(bg_coord , fg_coord ) == None:
                                overlap_flag=True

                if not overlap_flag:
                    bg_images.append(list(img[bg_y1:bg_y2, bg_x1:bg_x2]))
        return np.asarray(bg_images)
    def save_csv_paths(self , save_path='./'):
        #save train_csv and test_csv to path

        f=open(save_path+'train_path.txt' ,'w')
        for path in self.train_csv_paths:
            f.write(path+'\n')
        f.close()

        f = open(save_path + 'test_path.txt', 'w')
        for path in self.test_csv_paths:
            f.write(path+'\n')
        f.close()












"""
    
    def _get_cropped(self):
        root_root_roi_dir = './data/roi'
        root_root_fg_dir = './data/fg'
        root_root_bg_dir = './data/bg'

        if not os.path.exists(root_root_roi_dir):
            os.makedirs(root_root_roi_dir)
            print 'a'
        if not os.path.exists(root_root_fg_dir):
            os.makedirs(root_root_fg_dir)
        if not os.path.exists(root_root_bg_dir):
            os.makedirs(root_root_bg_dir)
        for path in self.train_csv_paths:
            name=os.path.split(path)[1]
            name = os.path.splitext(name)[0]
            labels = self.get_coords(path) # csv별 roi을 가져온다. 예시 [4] : [[x1,y1 x2, y2] ...[x1,y1 x2, y2]]

            # root_root_roi , fg_dir , bg_dir 에서 csv 이름이 있는 폴더를 생성한다
            # E.G) './data/roi'-->'./data/roi/140352'
            root_roi_dir, root_fg_dir, root_bg_dir = map(lambda path: os.path.join(path, name),
                                                         [root_root_roi_dir, root_root_fg_dir, root_root_bg_dir])
            if not os.path.isdir(root_roi_dir):
                map(lambda path: os.makedirs(os.path.join(path)),[root_roi_dir, root_fg_dir, root_bg_dir])
            print 'name:',name
            for k in labels.keys():
                if not k ==1 : # 1번 라벨에 대해서만 fg을 얻는다
                    continue
                # make dir
                roi_dir, fg_dir, bg_dir = map(lambda path: os.path.join(path, str(k)),
                                              [root_roi_dir, root_fg_dir, root_bg_dir])
                if not os.path.isdir(roi_dir):
                    map(lambda path: os.makedirs(os.path.join(path)), [roi_dir, fg_dir, bg_dir])
                # load image
                img = np.asarray(Image.open(os.path.join(self.img_dir, name + '.png')))
                # crop image
                cropped_images, cropped_coords = dense_crop(img, 75, 75)  # 100 , 100 으로 모든 이미지를 자른다

                # save background image and foreground image to each folder
                for fg_coord in labels[k]:
                    try:
                        print 'foreground coord : {}'.format(fg_coord)
                        fg_x1, fg_y1, fg_x2, fg_y2=map(int , fg_coord)
                        fg_w = fg_x2 - fg_x1
                        fg_h = fg_y2 - fg_y1
                        fg_area = fg_w * fg_h

                        fg_croppped_imgs,fg_croppped_coords=dense_crop(img[fg_y1-10 :fg_y2+10 ,fg_x1-10L:fg_x2+10],75,75)
                        #print np.shape(fg_croppped_imgs)
                        fig = plt.figure()
                        ax1 = fig.add_subplot(1,2,1)
                        ax1.imshow(img)
                        rect=patches.Rectangle((fg_x1, fg_y1), fg_w, fg_h , facecolor=None , linewidth=1 , edgecolor='r' ,fill=False)
                        ax1.add_patch(rect)
                        ax2 = fig.add_subplot(1, 2, 2)
                        ax2.imshow(fg_croppped_imgs[0])
                        plt.show()
                        plt.close()
                        np.save(os.path.join(fg_dir, 'fg.npy'),fg_croppped_imgs)
                    except Exception as e:
                        print 'error coordinate ',fg_x1, fg_y1, fg_x2, fg_y2
"""

if __name__ =='__main__':
    img_dir ='/Users/seongjungkim/data/detection/resize'
    csv_dir='/Users/seongjungkim/data/detection/csv'
    model=Preprocessing(csv_dir , img_dir)
    rois=model.get_rois(1,model.test_csv_paths)
    print len(rois.keys())
    #print np.shape(model.fg_images)
    #print np.shape(model.bg_images)


    """
    for k in model.train_labels.keys():
        print len(model.train_labels[k])
    print model.train_labels[2]
    width_height=map(lambda coord : get_width_height(coord) , model.train_labels[1])

    count =0
    for w,h in width_height:
        if w < 100 and h < 100 :
            count +=1
            print w ,h
            plt.scatter(w,h)
        print count


    plt.show()
    exit()
    
    img=Image.open('../debug/0.png').convert('RGB')
    np_img=np.asarray(img)

    print np.shape(np_img)
    cropped_images , coords=dense_crop(np_img,30,30)

    for coord in coords:
        cropped_images = img.crop(coord)
        plt.imshow(cropped_images)
        plt.show()

    """
