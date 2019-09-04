import numpy as np
import tensorflow as tf

def inter_area_batch(im_inp,h,w,hs,ws):
	'''Create a TensorFlow differntiable graph that evaluates resize function 
	equal to the cv2.inter_area()
	
	im_inp: rank 4 image tensor and shape [N, h, w, 3]
	Input tensor has to be float64!
	h: int, source height
	w: int, source width
	hw: int, target height
	ws: int, target width
	
	return: rank 4 resized image tensor with format NHWC and shape [N, hs, ws, 3]
	'''
	
	print('============ Input tensor has to be float64 to work exactly like inter_area from cv! ============')
	
	b_size = tf.shape(im_inp)[0]
	
	split_y = np.tile(np.expand_dims(np.linspace(0,h,hs+1),1),[1,ws+1])
	split_x = np.tile(np.expand_dims(np.linspace(0,w,ws+1),0),[hs+1,1])
	split = np.stack([split_y,split_x],axis=2)
	split_floor = np.floor(split).astype(np.int32)
	split_ceil = np.ceil(split).astype(np.int32)
	split_int = np.concatenate([split_ceil-1,split_floor+1],axis=2)
	
	y_cumsum = tf.cumsum(im_inp,axis=1)
	cumsum = tf.pad(tf.cumsum(y_cumsum,axis=2),[[0,0],[1,0],[1,0],[0,0]])
	y_cumsum = tf.pad(y_cumsum,[[0,0],[1,0],[0,0],[0,0]])
	x_cumsum = tf.pad(tf.cumsum(im_inp,axis=2),[[0,0],[0,0],[1,0],[0,0]])
	
	numer = tf.range(b_size,dtype=tf.int32)
	numer = tf.expand_dims(tf.expand_dims(tf.expand_dims(numer,1),1),1)
	numer = tf.tile(numer,[1,hs,ws,1])
	def gth(arr,b_size):
		arr = np.expand_dims(arr.copy(),0)
		arr = tf.tile(arr,[b_size,1,1,1])
		return tf.concat([numer,arr],axis=3)
		
	floor_floor = tf.gather_nd(cumsum,gth(split_int[1:,1:,:2],b_size))
	ceil_ceil = tf.gather_nd(cumsum,gth(split_int[:-1,:-1,2:],b_size))
	floor_ceil = tf.gather_nd(cumsum,gth(split_int[1:,:-1,0:4:3],b_size))
	ceil_floor = tf.gather_nd(cumsum,gth(split_int[:-1,1:,2:0:-1],b_size))
	whole = floor_floor+ceil_ceil-floor_ceil-ceil_floor
	
	whole += (tf.gather_nd(y_cumsum,gth(split_int[1:,1:,:2],b_size))-\
				tf.gather_nd(y_cumsum,gth(split_int[:-1,1:,2:0:-1],b_size)))*\
					np.expand_dims(split[1:,1:,1]-split_int[1:,1:,1],2)
	whole += (tf.gather_nd(x_cumsum,gth(split_int[1:,1:,:2],b_size))-\
				tf.gather_nd(x_cumsum,gth(split_int[1:,:-1,0:4:3],b_size)))*\
					np.expand_dims(split[1:,1:,0]-split_int[1:,1:,0],2)
	whole += (tf.gather_nd(y_cumsum,gth(np.stack([split_int[1:,:-1,0],split_floor[1:,:-1,1]],axis=2),b_size))-\
				tf.gather_nd(y_cumsum,gth(np.stack([split_int[:-1,:-1,2],split_floor[:-1,:-1,1]],axis=2),b_size)))*\
					np.expand_dims(split_int[:-1,:-1,3]-split[:-1,:-1,1],2)
	whole += (tf.gather_nd(x_cumsum,gth(np.stack([split_floor[:-1,1:,0],split_int[:-1,1:,1]],axis=2),b_size))-\
				tf.gather_nd(x_cumsum,gth(np.stack([split_floor[:-1,:-1,0],split_int[:-1,:-1,3]],axis=2),b_size)))*\
					np.expand_dims(split_int[:-1,:-1,2]-split[:-1,:-1,0],2)
					
	whole += tf.gather_nd(im_inp,gth(split_int[1:,1:,:2],b_size))*\
				np.expand_dims(split[1:,1:,1]-split_int[1:,1:,1],2)*\
					np.expand_dims(split[1:,1:,0]-split_int[1:,1:,0],2)
	whole += tf.gather_nd(im_inp,gth(np.stack([split_int[1:,:-1,0],split_floor[1:,:-1,1]],axis=2),b_size))*\
				np.expand_dims(split_int[:-1,:-1,3]-split[:-1,:-1,1],2)*\
					np.expand_dims(split[1:,1:,0]-split_int[1:,1:,0],2)
	whole += tf.gather_nd(im_inp,gth(np.stack([split_floor[:-1,1:,0],split_int[:-1,1:,1]],axis=2),b_size))*\
				np.expand_dims(split[1:,1:,1]-split_int[1:,1:,1],2)*\
					np.expand_dims(split_int[:-1,:-1,2]-split[:-1,:-1,0],2)
	whole += tf.gather_nd(im_inp,gth(split_floor[:-1,:-1],b_size))*\
				np.expand_dims(split_int[:-1,:-1,3]-split[:-1,:-1,1],2)*\
					np.expand_dims(split_int[:-1,:-1,2]-split[:-1,:-1,0],2)

	whole /= np.prod(split[1,1])
	
	return whole
	
def grad_form_area(grads,h,w):
	'''Propagate gradients from the output of inter_area resize function 
	to the input using numpy only
	
	grads: rank 3 np.array with format HWC of gradients values on the output
	h: height of the source image
	w: width of the source image
	
	return: np.array of gradients values on the input
	'''
	
	hs = grads.shape[0]
	ws = grads.shape[1]
	
	h_split = np.linspace(0,h,hs+1)
	w_split = np.linspace(0,w,ws+1)
	
	area = h_split[1]*w_split[1]
	new_grad = np.zeros((h,w,3),dtype=np.float64)
	
	h_range = np.tile(np.expand_dims(np.arange(h),1),[1,w])
	w_range = np.tile(np.expand_dims(np.arange(w),0),[h,1])
	
	h_ind = (h_range//h_split[1]).astype(int)
	w_ind = (w_range//w_split[1]).astype(int)
	
	h_ind2 = np.minimum(h_ind+1,hs-1)
	w_ind2 = np.minimum(w_ind+1,ws-1)
	
	h_next = h_split[h_ind+1]
	w_next = w_split[w_ind+1]
	
	h_over = np.expand_dims(np.clip(h_next-h_range,None,1.),2)
	w_over = np.expand_dims(np.clip(w_next-w_range,None,1.),2)
	
	new_grad = grads[(h_ind.ravel(),w_ind.ravel())].reshape((h,w,3))*h_over*w_over+\
				grads[(h_ind2.ravel(),w_ind.ravel())].reshape((h,w,3))*(1.-h_over)*w_over+\
				grads[(h_ind.ravel(),w_ind2.ravel())].reshape((h,w,3))*h_over*(1.-w_over)+\
				grads[(h_ind2.ravel(),w_ind2.ravel())].reshape((h,w,3))*(1.-h_over)*(1.-w_over)
	return new_grad/h/w*hs*ws

