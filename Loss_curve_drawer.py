#Title:Convolutional Neural Network forcasting, architecture testing
#Created by:Abetharan Antony
#Institution:Imperial College London
#Msc Thesis: Improving Tropical Cyclone Forecasting Using Neural Networks
#
#Objective: plots loss curves for all networks that ran that have saved out
#loss for analysis.
#=================================================================================


import numpy as np
import matplotlib.pyplot as plt
import os

def plotter(validation_data,mini_batch_data,model_name):

	validation_data_range=np.arange(0,np.shape(validation_data),10)
	mini_batch_data_range=np.arange(0,np.shape(mini_batch_data),1)

	fig = plt.figure()
	ax = plt.subplot(111)
	w = 1
	validation=ax.plot(validation_data_range,validation_data)
	mini_batch=ax.plot(mini_batch_data_range,mini_batch_data)
	ax.set_ylabel('Loss')
	ax.set_xlabel('Step')
	ax.set_title('\n'.join(wrap('Loss trend for '+model_name,60)))

	ax.legend( (validation[0], mini_batch[0]), ('Prediction', 'Truth') )

	fig.savefig(model_name+'_frequency_histogram')
	
	

directory='path to data files' #Replace with path to data files

for item in os.lidir(directory)
	
	
	prediction_full_name=prediction_full_file[i]
	label_full_name=labels_full_file[i]

	if i<6:
		model_name='Conv'+prediction_full_name[16:]
	else:
		model_name='Conv_aug'+prediction_full_name[26:]
	predictions_full=np.load(directory+prediction_full_name+'.npy')	
	labels_full=np.load(directory+label_full_name+'.npy')	
	plotter(labels_full,predictions_full,model_name)
	
