import numpy as np
import matplotlib.pyplot as plt
print 'a'

img=np.load('../fundus_data/cropped_original_fundus/cataract_glaucoma/8261421_20160614_R.npy')
plt.imshow(img)
plt.show()