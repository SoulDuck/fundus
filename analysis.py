import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
sample_img=Image.open('./debug/1.png').convert('RGB')
sample_img=np.asarray(sample_img)
print np.shape(sample_img)
r_ch=sample_img[0]
g_ch=sample_img[1]
b_ch=sample_img[2]

img_reduced=pca.fit_transform(sample_img)
print np.shape(img_reduced)




