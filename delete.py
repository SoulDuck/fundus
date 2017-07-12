import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
print 'a'
"""
img=Image.open('../fundus_data/cropped_original_fundus/cataract/955512_20160808_R.png.png')
plt.imshow(img)
plt.savefig('./sample.png')
plt.show()
"""
list_=[1,2,3,4,5,6,7,8,9,10]
list_=np.asarray(list_)
list__=[1,2]
print list_[list__]
print random.sample(list_,10)