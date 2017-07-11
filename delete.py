import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
print 'a'

img=Image.open('../fundus_data/cropped_original_fundus/cataract/955512_20160808_R.png.png')
plt.imshow(img)
plt.savefig('./sample.png')
plt.show()
