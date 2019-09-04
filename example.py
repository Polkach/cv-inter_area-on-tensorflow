import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import cv2
from inter_area import *

im = cv2.imread('banana.jpg')/255.
print('-------------- Source image shape:', im.shape)

tarhet_h, target_w = 123, 456
print('-------------- Target image shape:', (tarhet_h, target_w, 3))

im_cv = cv2.resize(im, (target_w, tarhet_h), interpolation=cv2.INTER_AREA)

image = tf.placeholder(tf.float64,shape=[None,None,None,3])
resized = inter_area_batch(image,im.shape[0],im.shape[1],tarhet_h,target_w)

sess = tf.Session()
im_tf = sess.run(resized,feed_dict={image:np.expand_dims(im,0)})

print('-------------- Max difference between TF-resized and cv2-resized images:',np.max(np.abs(im_tf[0]-im_cv)))

print('-------------- Calculating gradients using TensorFlow and NumPy')
grads_out = np.random.uniform(0.,1.,(1,tarhet_h,target_w,3))
grads = tf.gradients(resized,image,grad_ys=[grads_out])

grads_tf = sess.run(grads,feed_dict={image:np.expand_dims(im,0)})[0][0]
grads_np = grad_form_area(grads_out[0],im.shape[0],im.shape[1])

print('-------------- Max difference between TF gradients and NumPy gradients:',np.max(np.abs(grads_tf-grads_np)))
