#Title:Balancing Pre-Processing module for Tropical cyclone forecasting
#Creatd by:Abetharan Antony
#Institution:Imperial College London
#Msc Thesis: Improving Tropical Cyclone Forecasting Using Neural Networks
#
#Objective: Convolutional neural network for learning classification of tropical cyclones
#Architecture: Based on Krizhevsky et al. ImageNet Classification with Deep Convolutional Neural Networks
#modified by changing filter size as well as remove local response normalization in favour of batch
#normalization.
#Includes 2 hidden layers with softmax output.
#Regularizers: l2 and dropout 
#=================================================================================


#libraries 
import tensorflow as tf
import numpy as np
import math as math
import random as rand
import sklearn.preprocessing
import time as t
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.decomposition import PCA
import random as rand
import matplotlib.pyplot as plt
import matplotlib as mp
import pickle

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


def train(epoch_max):
	'''
	Purpose: Neural network body, has all the functions required to generate my graph, run training and test model.
	'''

	data=np.load("/disk1/rtproj17/Neural_net_for_cyclone/data_192/pixel_array_ADASYN_192.npy")
	labels=np.load("/disk1/rtproj17/Neural_net_for_cyclone/data_192/labels_ADASYN_192.npy")
	
	eval_data=np.load("/disk1/rtproj17/Neural_net_for_cyclone/data_192/eval_data_ADASYN_192_for_split.npy")
	eval_labels=np.load("/disk1/rtproj17/Neural_net_for_cyclone/data_192/eval_labels_ADASYN_192_for_split.npy")
	
	training_data,training_labels,validation_data,validation_labels,testing_data,testing_labels=cross_validation_split(eval_data,eval_labels,data,labels)
	
	train_mean=np.mean(training_data,dtype=training_data.dtype)
	
	training_data=pre_process(training_data,train_mean)
	validation_data=pre_process(validation_data,train_mean)
	testing_data=pre_process(testing_data,train_mean)


	#Hyper parameters
	#
	beta=0.01
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
	config = tf.ConfigProto(intra_op_parallelism_threads=18, inter_op_parallelism_threads=18, \
                        allow_soft_placement=True, device_count = {'CPU': 1})
	sess = tf.Session(config=config)
	

	with tf.name_scope('input'):
		#Shape == [batch_size,number of data]
		x=tf.placeholder(tf.float32,shape=[None,image_size*image_size],name='x-input') 
		y_=tf.placeholder(tf.float32,shape=[None,13],name='y_input')
		phase = tf.placeholder(tf.bool, name='phase')
		keep_prob=tf.placeholder(tf.float32)
	with tf.name_scope('image'):
		image_shaped=tf.reshape(x,[-1,image_size,image_size,no_channels])
		tf.summary.image('input',image_shaped,13)


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
							      	    activation=None,
							      	    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=beta)

							     	    )

			with tf.name_scope('Batch_normalise_input'):	
				conv=tf.layers.batch_normalization(conv,
				                              center=True, scale=True, 
				                              training=phase)
			tf.summary.histogram('pre-relu-activation',conv)

			with tf.name_scope('Activation'):
				activation=tf.nn.relu(conv,name='relu-activation')


			with tf.name_scope('pool'):
				pool = tf.layers.max_pooling2d(inputs=activation, pool_size=[3, 3], strides=2)

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
	#if step%10!=0:
	#	feature_map = tf.image.random_contrast(feature_map, lower=0.5, upper=1.0)
	#print(feature_map.get_shape().as_list())
	
	#Normalising data using batch-norm
	
	#feature_normal=batch_norm(feature_map,phase=phase)

	#First Convolution+pooling layer
	#Input size [batch_size,192,192,1]
	#Output size [batch_size,95,95,48]
	conv_pool_1=conv_nn_layer(feature_map,'conv_pool_1',48,phase=phase)
	
	#conv1_norm_batch=norm_batch(conv_pool_1,phase,'conv_batch_norm')

	#Second convolutional layer
	#Input shape [batch_size,95,95,48]
	#Output shape [batch_size,47,47,128]

	conv_pool_2=conv_nn_layer(conv_pool_1,'conv_pool_2',128,phase=phase)

	#Third convolutiona layer
	#Input shape [batch_size,47,47,128]
	#Output shape [batch_size,47,47,192]

	conv3=conv_nn_no_pool(conv_pool_2,'conv_no_pool_3',192,phase=phase)

	#Fourth convolutional layer
	#Input shape [batch_size,47,47,192]
	#Output shape [batch_size,47,47,192]

	conv4=conv_nn_no_pool(conv3,'conv_no_pool_4',192,phase=phase)
	
	#Firth convolutional layer
	#Input shape [batch_size,47,47,192]
	#Output shape [batch_size,23,23,128]

	conv_pool_5=conv_nn_layer(conv4,'conv_pool_5',128,phase=phase)


	#Flatten for dense layer
	pool5_flat=tf.reshape(conv_pool_5,[-1,23*23*128])

	#First FC
	dense=dense_layer(pool5_flat,'dense_layer_1',2048,phase=phase)
	dropped=dropout(dense,layer_name='dropped_1',keep_prob=keep_prob)
	
	#Second FC
	dense2=dense_layer(dropped,'dense_layer_2',2048,phase=phase)
	dropped2=dropout(dense2,layer_name='dropped_2',keep_prob=keep_prob)
	#Softmax layer

	y=dense_layer(dropped2,'final_layer',13,phase=phase,act=tf.identity)


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
	
	with tf.name_scope('top-5'):
		with tf.name_scope('predictions'):
			labels = tf.argmax(y_, 1)
		with tf.name_scope('top-5'):
			topFive = tf.nn.in_top_k(y, labels, 5)

	tf.summary.scalar('accuracy',accuracy)
	merged=tf.summary.merge_all()
	train_writer=tf.summary.FileWriter('/disk1/rtproj17/Neural_net_for_cyclone/tensorboard_results/AlexNET_192',sess.graph)
	test_writer=tf.summary.FileWriter('/disk1/rtproj17/Neural_net_for_cyclone/tensorboard_results/AlexNET_192')
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()

    

	def feed_dict(mode,step):

		if  mode=='train':
			if i%10==1:
				print('Training')
			# Generate a minibatch.
			offset = (step * batch_size) % (training_labels.shape[0] - batch_size)		#offset added to introduce stochastic gradient learning 
		 	#Generate a minibatch.
			step_1=offset+batch_size
			xs=training_data[offset:step_1,:]
			ys = training_labels[offset:step_1,:]
			k=0.8
			phases=1

		elif mode=='validate':
			print('Validating')
			max_number=(np.shape(validation_data)[0]-100)
			random_number=rand.randint(0,max_number)
			xs=validation_data[random_number:random_number+100,:]
			ys=validation_labels[random_number:random_number+100,:]

			k=1.0
			phases=0
		elif mode=='test':
			print('Testing')
			xs=testing_data
			ys=testing_labels
			k=1.0
			phases=0

		return {x:xs,y_:ys,keep_prob:k,phase:phases}

	for epoch in range(epoch_max):
		max_number=(np.shape(training_data)[0]-batch_size*steps)
		random_number=rand.randint(0,max_number)
		training_data=training_data[random_number:random_number+batch_size*steps]
		
		if epoch>0:
			avg_loss_list.append(np.mean(loss_list))
			print(avg_loss_list)
			if avg_loss_list[epoch-2]>avg_loss_list[epoch-1]:
				saver.save(sess, 'Alex_net_deep',global_step=epoch)
			print('loss_list_update_on_length:',len(loss_list))
			full_loss.append(loss_list)
			print('full_list_update_on_length',len(full_loss))

		loss_list=[]
		for i in range(steps):
			if i %10 ==0:
				summary,acc=sess.run([merged,accuracy],feed_dict=feed_dict('validate',i))
				test_writer.add_summary(summary,i)
				print('Validation accuracy at epoch {} step {}: {}'.format(epoch,i,acc))
				validation_list.append(acc)
			else:
				if i%200 ==99 and epoch==epoch_max-1:
					run_options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
					run_metadata=tf.RunMetadata()
					summary,_,acc,loss=sess.run([merged,train_step,accuracy,cross_entropy],feed_dict=feed_dict('train',i),
											options=run_options,run_metadata=run_metadata)
					train_writer.add_run_metadata(run_metadata,'step',i)
					train_writer.add_summary(summary,i)
					print('Adding run metadata for',i)	
					mini_batch_list.append(acc)
					loss_list.append(loss)
					

				else:
					summary,_,acc,loss=sess.run([merged,train_step,accuracy,cross_entropy],feed_dict=feed_dict('train',i))
					train_writer.add_summary(summary,i)
					print('mini_batch accuracy at epoch {} step {}: {}'.format(epoch,i,acc))
					print('mini batch loss at epoch {} step {}: {}'.format(epoch,i,loss))
					loss_list.append(loss)
					mini_batch_list.append(acc)

		if epoch==epoch_max-1:
			acc,cm,topFive=sess.run([accuracy,confusion_matrix,topFive],feed_dict=feed_dict('test',i))
			print(cm)
			print(acc)
			print('Top-5 score: {}'.format(topFive))
	
	sess.close()
	train_writer.close()
	test_writer.close()
	return avg_loss_list,mini_batch_list,validation_list,full_loss

epoch_max=25
avg_loss_list,mini_batch_list,validation_list,full_loss=train(epoch_max)	

pickle.dump(avg_loss,open('/data_192/model_output_data/AlexNet/avg_loss.p','wb'))
pickle.dump(mini_batch_list,open('/data_192/model_output_data/AlexNet/mini_batch_list.p','wb'))
pickle.dump(validation_list,open('/data_192/model_output_data/AlexNet/validation_list.p','wb'))
pickle.dump(full_loss,open('/data_192/model_output_data/AlexNet/full_loss.p','wb'))











