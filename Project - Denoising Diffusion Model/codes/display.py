''' This is a simple script to display the output of the generator for your reference.
'''
import cv2
import numpy as np

filename = 'test_run/samples_30x128x128x3.npz'


outputs = np.load(filename)['arr_0']
for i in range(outputs.shape[0]):
    img_i = outputs[i][..., ::-1] # BGR -> RGB
    name = './vis_{:05d}.png'.format(i)
    cv2.imwrite(name, img_i)
    print(name, img_i.shape)