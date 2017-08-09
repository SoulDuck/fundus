import os , glob
import numpy as np
from PIL import Image
import PIL


path,names,files=os.walk('../fundus_data/cropped_original_fundus_300x300/').next()
paths=map(lambda name : os.path.join(path , name) , names)
list_paths=map(lambda path : glob.glob(path+'/*.png.png..png') ,paths)
for paths in list_paths:
    paths=map(lambda path : os.rename(path , path.replace('.png.png..png', '.png') ) , paths)


