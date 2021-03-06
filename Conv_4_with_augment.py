#Title:Convolutional Neural Network forcasting, architecture testing
#Created by:Abetharan Antony
#Institution:Imperial College London
#Msc Thesis: Improving Tropical Cyclone Forecasting Using Neural Networks
#
#Objective: Convolutional neural network for learning classification of tropical cyclones
#Architecture: 4 Convolutional layers including pooling 2x2 with stides of 2
#reduces image size down to 12x12 
#Includes 2 hidden layers with softmax output.
#Regularizers: l2 and dropout 
#=================================================================================


#libraries 
import tensorflow as tf
import numpy as np
import math as math
import random as rand
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.decomposition import PCA
import random as rand
import matplotlib.pyplot as plt
import time 
from sklearn.utils import shuffle
import pickle 
from sklearn.utils import shuffle
from skimage import transform, filters, exposure

def pre_process(data,mean):
	'''
	Purpose: Centre image globally and locally 
	args.
	    Data: Input data for centering shape [n_sampels,features]
		dtype: Type of data output expected. 
	returns.
			centered data 

	'''
	
	#Zero-Centre
	data-=mean
	
	#Standardize Centering
	std=np.std(data, axis = 0)
	if any(std)==0:
		index=np.where(std==0)
		std[index]=1
	data /= std

	
	return data


def cross_validation_split(eval_data,eval_labels,training_data,training_labels):
	'''
		Purpose: Splitting Eval-data randomly such to receive validation and test data
				and one-hot encodes the labels 
		
		args.

			Eval_Data: Input pixel data for splitting into train,validation,test data.
			Eval_labels: Input label data for splitting into train,validation,test data.

		return.
			Training_data,Training_labels,Validation_data,Validation_label,Test_data,Test_label
	'''
	

	validation_data,testing_data,validation_labels,testing_labels=train_test_split(
		eval_data,eval_labels,test_size=0.5,random_state=10)

	label_binarizer = sklearn.preprocessing.LabelBinarizer()
	label_binarizer.fit((np.unique(training_labels)))
	
	training_labels=label_binarizer.transform(training_labels)
	validation_labels=label_binarizer.transform(validation_labels)
	testing_labels=label_binarizer.transform(testing_labels)


	print("training data sizes",np.shape(training_data),np.shape(training_labels))
	print("validation data sizes",np.shape(validation_data),np.shape(validation_labels))
	print("testing data sizes",np.shape(testing_data),np.shape(testing_labels))
	
	return training_data,training_labels,validation_data,validation_labels,testing_data,testing_labels

def data_augment(data, y):
	''' Modified version for tensorflow and my data of T Florian Muellerkleins deep learning challenge data augmentation code, from: 
	   http://florianmuellerklein.github.io/cnn_streetview/
	'''

	'''
	Data augmentation batch iterator for feeding images into CNN.
	rotate all images in a given batch between -10 and 10 degrees
	random translations between -10 and 10 pixels in all directions.
	random zooms between 1 and 1.3.
	random shearing between -25 and 25 degrees.
	randomly applies sobel edge detector to 1/4th of the images in each batch.
	randomly inverts 1/2 of the images in each batch.
	'''

	pixels=192

	
	max_values=np.amax(data,axis=1)
	data=data=np.divide(data,max_values[:,None])
	X_batch = np.asarray(data,dtype=np.float32)
	y_batch = y
	# set empty copy to hold augmented images so that we don't overwrite
	X_batch_aug = np.empty(shape = (X_batch.shape[0],pixels, pixels),
	                       dtype = 'float32')

	# random rotations betweein -10 and 10 degrees
	dorotate = rand.randint(-5,5)


	# random translations
	trans_1 = rand.randint(-10,10)
	trans_2 = rand.randint(-10,10)

	# random zooms
	zoom = rand.uniform(1, 1.3)


	# set the transform parameters for skimage.transform.warp
	# have to shift to center and then shift back after transformation otherwise
	# rotations will make image go out of frame
	center_shift   = np.array((pixels, pixels)) / 2. - 0.5
	tform_center   = transform.SimilarityTransform(translation=-center_shift)
	tform_uncenter = transform.SimilarityTransform(translation=center_shift)

	tform_aug = transform.AffineTransform(rotation = np.deg2rad(dorotate),
	                                      scale =(1/zoom, 1/zoom),	
	                                      translation = (trans_1, trans_2)
	                                      )

	tform=tform_aug+tform_center+tform_uncenter

	# images in the batch do the augmentation
	for j in range(X_batch.shape[0]):
		warping_image=np.reshape(X_batch[j],[192,192])
		X_batch_aug[j,:,:] = transform.warp(warping_image, tform)
	                                  
	X_batch_aug=np.reshape(X_batch_aug,[-1,192*192])
	
	return X_batch_aug,y_batch

def train(epoch_max):
	'''
	Purpose: Neural network body, has all the functions required to generate my graph, run training and test model.
	'''

	#Import data where eval_data is to be split into validation data and testing data.
	#This was done as for this project the training data went through re-balancing using Adasyn 
	#Feel free to modify this section as see fit. 
	data=np.load()
	labels=np.load()
	
	eval_data=np.load()
	eval_labels=np.load()
	
	training_data,training_labels,validation_data,validation_labels,testing_data,testing_labels=cross_validation_split(eval_data,eval_labels,data,labels)
	
	train_mean=np.mean(training_data,dtype=training_data.dtype)
	
	training_data=pre_process(training_data,train_mean)
	validation_data=pre_process(validation_data,train_mean)
	testing_data=pre_process(testing_data,train_mean)

	train_mean=np.mean(training_data,dtype=training_data.dtype)
	
	training_data=pre_process(training_data,train_mean)
	validation_data=pre_process(validation_data,train_mean)
	testing_data=pre_process(testing_data,train_mean)


	#Hyper parameters
	#
	beta=0.001
	batch_size=128
	learning_rate=0.01
	steps=170
	avg_loss_list=[]
	validation_list=[]
	mini_batch_list=[]
	full_loss=[]
	image_size=192
	no_channels=1
	graph=tf.Graph()
	config = tf.ConfigProto(intra_op_parallelism_threads=20, inter_op_parallelism_threads=20, \
                        allow_soft_placement=True, device_count = {'CPU': 1})
	sess = tf.Session(config=config)
	

	with tf.name_scope('input'):
		#Shape == [batch_size,number of data]
		x=tf.placeholder(tf.float32,shape=[None,image_size*image_size],name='x-input') 
		y_=tf.placeholder(tf.float32,shape=[None,16],name='y_input')
		phase = tf.placeholder(tf.bool, name='phase')
		keep_prob=tf.placeholder(tf.float32)
	with tf.name_scope('image'):
		image_shaped=tf.reshape(x,[-1,image_size,image_size,no_channels])
		tf.summary.image('input',image_shaped,16)


	def variable_summaries(var):
		'''	
		Purpose: Initilize tensorboard monitors
		args.
			var: Any tensorboard variable of interest
		'''

		with tf.name_scope('summaries'):
			mean=tf.reduce_mean(var)
			tf.summary_scalar('mean',mean)
			with tf.name_scope('stddev'):
				stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
			tf.summary.scalar('stddev',stddev)
			tf.summary.scalar('max',tf.reduce_max(var))
			tf.summary.scalar('min',tf.reduce_min(var))
			tf.summary.histogram('histogram',var)

	def batch_norm(input_tensor,phase):
		
		with tf.name_scope('Batch_normalise_input'):	
				normalized=tf.layers.batch_normalization(input_tensor,
	                                          center=True, scale=True, 
	                                          training=phase
	                                        	 )
		return normalized
		


	def conv_nn_layer(input_tensor,layer_name,NoFilters,phase):
		'''	
		Purpose: Implement convolutional layer followed by relu activation with max pooling.
		args.
			input_tensor: Input data with shape [batch_size,image_size,image_size,filters]
			layer_name: scope name
			NoFilters: No of filters to apply
		return.
			pool: max pooled layer 
	
		'''
		with tf.name_scope(layer_name):

			
			with tf.name_scope('conv'):
				conv= tf.layers.conv2d(
							      		inputs=input_tensor,
							     		filters=NoFilters,
							      		kernel_size=[3, 3],
							     	    padding="same",
							      	    activation=tf.nn.relu,
							      	    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=beta)
							     	    )
			tf.summary.histogram('pre-norm-activation',conv)

			with tf.name_scope('Batch_normalise'):	
				conv=tf.layers.batch_normalization(conv,
	                                          center=True, scale=True, 
	                                          training=phase
	                                        	 )
			tf.summary.histogram('after-norm-activation',conv)

			with tf.name_scope('pool'):
				pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[2, 2], strides=2)

		return pool

	
	def conv_nn_no_pool(input_tensor,layer_name,NoFilters,phase):
		'''
		Purpose: Implement convolutional layer followed by relu activation with batch normalization.
		Args.
			input_tensor: Input data with shape [batch_size,image_size,image_size,filters]
			layer_name: scope name
			NoFilters: No of filters to apply
		return.
			pool: convolution outputs after batch normalizing 

		'''
		with tf.name_scope(layer_name):
				
			with tf.name_scope('conv_no_pool'):
					
				conv= tf.layers.conv2d(
							      		inputs=input_tensor,
							     		filters=NoFilters,
							      		kernel_size=[3, 3],
							     	    padding="same",
							      	    activation=tf.nn.relu,
							      	    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=beta)

							     	    )
			tf.summary.histogram('pre-norm_no_pool-activation',conv)
		
		return conv

	def dense_layer(input_tensor,layer_name,units,phase,act=tf.nn.relu):
		'''	
		Purpose: Create fully connected layer 
		args.
			input_tensor: Input data with shape [batch_size,image_size,image_size,filters]
			layer_name: scope name
			units: No of neurons 
		return.
			dense: full connected output after activation
	
		'''
		with tf.name_scope(layer_name):
			dense=tf.layers.dense(inputs=input_tensor,units=units,activation=act)
			
			dense=tf.layers.batch_normalization(dense, 
                                          center=True, scale=True, 
                                          training=phase
                                        	 )
		return dense
	def dropout(input_tensor,layer_name,keep_prob):
		
		with tf.name_scope('layer_name'):
			dropped=tf.nn.dropout(input_tensor,keep_prob)
		return dropped



	#Reshape batch shape [batch_Size,image_size*image_size*filters] to [batch_size,image_size,image_size,filters]
	feature_map=tf.reshape(x,[-1,192,192,1])
	
	#Normalising data using batch-norm
	
	#feature_normal=batch_norm(feature_map,phase=phase)

	#First Convolution+pooling layer
	#Input size [batch_size,192,192,1]
	#Output size [batch_size,96,96,48]
	conv_pool_1=conv_nn_layer(feature_map,'conv_pool_1',48,phase=phase)

	#conv1_norm_batch=norm_batch(conv_pool_1,phase,'conv_batch_norm')

	#Second convolutional layer
	#Input shape [batch_size,96,96,48]
	#Output shape [batch_size,48,48,128]

	conv_pool_2=conv_nn_layer(conv_pool_1,'conv_pool_2',128,phase=phase)

	#Third convolutiona layer
	#Input shape [batch_size,48,48,128]
	#Output shape [batch_size,48,48,192]

	conv3=conv_nn_no_pool(conv_pool_2,'conv_no_pool_3',192,phase=phase)
	
	#Fourth convolutional layer
	#Input shape [batch_size,48,48,192]
	#Output shape [batch_size,24,24,128]

	conv_pool_4=conv_nn_layer(conv3,'conv_no_pool_4',128,phase=phase)


	#Flatten for dense layer
	pool4_flat=tf.reshape(conv_pool_4,[-1,24*24*128])

	#First FC
	dense=dense_layer(pool4_flat,'dense_layer_1',2048,phase=phase)
	dropped=dropout(dense,layer_name='dropped_1',keep_prob=keep_prob)
	
	#Second FC
	dense2=dense_layer(dropped,'dense_layer_2',2048,phase=phase)
	dropped2=dropout(dense2,layer_name='dropped_2',keep_prob=keep_prob)
	#Softmax layer

	y=dense_layer(dropped2,'final_layer',16,phase=phase,act=tf.identity)


	with tf.name_scope('Cross_entropy'):
		diff=tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y)
		with tf.name_scope('total'):
			cross_entropy=tf.reduce_mean(diff)
	tf.summary.scalar('cross_entropy',cross_entropy)

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		train_step=tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

	with tf.name_scope('confusion_matrix'):
		confusion_matrix=tf.confusion_matrix(tf.argmax(y,1),tf.argmax(y_,1))

	with tf.name_scope('accuracy'):
		with tf.name_scope('correct_prediction'):
			correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
		with tf.name_scope('accuracy'):
			accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	tf.summary.scalar('accuracy',accuracy)

	with tf.name_scope('top-5'):
		with tf.name_scope('predictions'):
			labels = tf.argmax(y_, 1)
		with tf.name_scope('top-5'):
			topFive = tf.nn.in_top_k(y, labels, 5)
	

	merged=tf.summary.merge_all()
	train_writer=tf.summary.FileWriter('useful_name',sess.graph) ##Replace useful_name with Name of the tensorboard file and location 
	test_writer=tf.summary.FileWriter('useful_name')
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()

    

	def feed_dict(mode,step,data,labels):

		if  mode=='train':
			if i%10==1:
				print('Training')
			# Generate a minibatch.
			offset = (step * batch_size) % (labels.shape[0] - batch_size)		#offset added to introduce stochastic gradient learning 
		 	#Generate a minibatch.
			step_1=offset+batch_size
			xs=data[offset:step_1,:]
			ys = labels[offset:step_1,:]
			xs,ys=data_augment(xs,ys)
			k=0.6
			phases=1
			#xs = tf.image.random_contrast(xs, lower=0.5, upper=1.0)
			

		elif mode=='validate':
			print('Validating')
			max_number=(np.shape(data)[0]-100)
			random_number=rand.randint(0,max_number)
			xs=data[random_number:random_number+100,:]
			ys=labels[random_number:random_number+100,:]
	
			k=1.0
			phases=0
		elif mode=='test':
			print('Testing')
			xs=data
			ys=labels
			k=1.0
			phases=0

		return {x:xs,y_:ys,keep_prob:k,phase:phases}

	for epoch in range(epoch_max):
		start_time = time.time()

		

		max_number=(np.shape(training_data)[0]-batch_size*steps)
		random_number=rand.randint(0,max_number)
		X=training_data[random_number:random_number+batch_size*steps]
		Y=training_labels[random_number:random_number+batch_size*steps]
		
		training_data,training_labels=shuffle(X,  Y, random_state=0)

		if epoch>0:
			avg_loss_list.append(np.mean(loss_list))
			print(avg_loss_list)
			full_loss.append(loss_list)
			pickle.dump(avg_loss_list,open('avg_loss_conv4_16_label_augmented.p','wb'))
			pickle.dump(mini_batch_list,open('mini_batch_list_16_label_augmented.p','wb'))
			pickle.dump(validation_list,open('validation_list_conv4_16_label_augmented.p','wb'))
			pickle.dump(full_loss,open('full_loss_conv4_16_label_augmented.p','wb'))

			if avg_loss_list[epoch-2]>avg_loss_list[epoch-1]:
				saver.save(sess, 'Conv4_deep_net_full_label_partial_augment_epoch_20',global_step=epoch-1) #Name network save file 
			
			
			
			
		loss_list=[]
		for i in range(steps):

			if i %10 ==0:
				#print validation accuracy for feedback on how the network is training
				summary,acc=sess.run([merged,accuracy],feed_dict=feed_dict('validate',i,validation_data,validation_labels))
				test_writer.add_summary(summary,i)
				print('Validation accuracy at epoch {} step {}: {}'.format(epoch,i,acc))
				validation_list.append(acc)
		
			else:
				if i%200 ==99 and epoch==epoch_max-1:
					#save the metagraph down for future use
					run_options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
					run_metadata=tf.RunMetadata()
					summary,_,acc,loss=sess.run([merged,train_step,accuracy,cross_entropy],feed_dict=feed_dict('train',i,training_data,training_labels),
											options=run_options,run_metadata=run_metadata)
					train_writer.add_run_metadata(run_metadata,'step',i)
					train_writer.add_summary(summary,i)
					print('Adding run metadata for',i)	
					mini_batch_list.append(acc)
					loss_list.append(loss)
					

				else:
					summary,_,acc,loss=sess.run([merged,train_step,accuracy,cross_entropy],feed_dict=feed_dict('train',i,training_data,training_labels))
					train_writer.add_summary(summary,i)
					print('mini_batch accuracy at epoch {} step {}: {}'.format(epoch,i,acc))
					print('mini batch loss at epoch {} step {}: {}'.format(epoch,i,loss))
					loss_list.append(loss)
					mini_batch_list.append(acc)

		if epoch==epoch_max-1:
			acc,cm,topFive=sess.run([accuracy,confusion_matrix,topFive],feed_dict=feed_dict('test',i,testing_data,testing_labels))
			print(cm)
			print(acc)
			print('Top-5 score: {}'.format(topFive))
			avg_loss_list.append(np.mean(loss_list))
			print(avg_loss_list)

	sess.close()
	train_writer.close()
	test_writer.close()
	print("--- %s seconds ---" % (time.time() - start_time))
	return avg_loss_list,mini_batch_list,validation_list,full_loss

epoch_max=6
avg_loss_list,mini_batch_list,validation_list,full_loss=train(epoch_max)	

pickle.dump(avg_loss_list,open('avg_loss_conv4_16_label_augmented.p','wb'))
pickle.dump(mini_batch_list,open('mini_batch_list_16_label_augmented.p','wb'))
pickle.dump(validation_list,open('validation_list_conv4_16_label_augmented.p','wb'))
pickle.dump(full_loss,open('full_loss_conv4_16_label_augmented.p','wb'))











