#-*- coding:utf-8 -*-
import numpy as np
from fundus_processing import overlaps , get_width_height
from PIL import Image
import matplotlib.pyplot as plt
import os
import glob
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



class preprocessing(object):
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
        self.train_labels =self._get_coords()
        self._get_cropped()


    def _get_coords(self):
        assert len(self.csv_paths) > 0 , 'the number of csv path {} '.format(len(self.csv_paths))
        train_labels = {}
        for path in self.train_csv_paths:
            f=open(path , 'r')
            lines=f.readlines()
            for i,line in enumerate(lines[1:]):
                label , x1 , y1 ,x2 , y2 =map(float , line.split(','))
                if not label in train_labels.keys():
                    train_labels[int(label)]=[[x1,y1,x2,y2]] #
                else:
                    train_labels[int(label)].append([x1,y1,x2,y2]) #)

        return train_labels


    def _get_cropped(self):
        for path in self.train_csv_paths:
            name=os.path.split(path)[1]
            name = os.path.splitext(name)[0]
            img=np.asarray(Image.open(os.path.join(self.img_dir , name+'.png')))
            #overlaps(img)
            #roi #foreground #backgound

            roi_dir='./data/roi'
            fg_dir='./data/fg'
            bg_dir='./data/bg'

            if not os.path.exists(roi_dir):
                os.makedirs(roi_dir)
            if not os.path.exists(fg_dir):
                os.makedirs(fg_dir)
            if not os.path.exists(bg_dir):
                os.makedirs(bg_dir)



if __name__ =='__main__':
    img_dir ='/Users/seongjungkim/data/detection/margin_cropped_image'
    csv_dir='/Users/seongjungkim/data/detection/csv'
    model=preprocessing(csv_dir , img_dir)
    exit()
    for k in model.train_labels.keys():
        print len(model.train_labels[k])
    print model.train_labels[1]
    width_height=map(lambda coord : get_width_height(coord) , model.train_labels[1])

    for w,h in width_height:
        print w ,h
        plt.scatter(w,h)
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


