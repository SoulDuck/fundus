import numpy as np
"""
import os  , sys ,glob

src_path='../fundus_data/cropped_original_fundus_300x300/'
folder_names=os.walk(src_path).next()[1]
for name in folder_names:
    src_folder=os.path.join(src_path , name)
    print src_folder
    paths=glob.glob(src_folder+'/*.png')
    for src_path in paths:
        print src_path
        new_path=src_path.replace('.png.png..png' , '.png')
        os.rename(src_path , new_path)
"""


list_=np.array([[1,2],[3,4],[5,6]])
list_=list_.reshape(-1)
print list_
