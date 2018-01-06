import dicom
import numpy as np
import matplotlib.pyplot as plt
def tmp(a,b,c):
    print a
    print b
    print c
dc=dicom.read_file('./tmp/SC.1.2.392.200036.9135.5000.1.19876.3500.1501636323.536.dcm')
print dc.pixel_array.shape
#tmp(3,(*(dc.pixel_array.shape[:])))



ds_image = np.reshape(dc.pixel_array,[2592, 3872,3])

np_dc=np.swapaxes(np.asarray(dc.pixel_array) , 0,1 )
print np.shape(np_dc)
np_dc=np.swapaxes(np_dc , 1,2 )
print np.shape(np_dc)
plt.imshow(np_dc)
plt.show()




