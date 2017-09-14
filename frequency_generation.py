#Title:Convolutional Neural Network forcasting, architecture testing
#Created by:Abetharan Antony
#Institution:Imperial College London
#Msc Thesis: Improving Tropical Cyclone Forecasting Using Neural Networks
#
#Objective: Find frequency of occurence of different labels
#=================================================================================

from __future__ import division
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import os 
import time
import pickle
def frequency_test(labels,predictions,model_name):
	'''
	Purpose: Plots a frequency histogram for truth labels and predictions to compare. 
	inputs: predictions  - prediction output of the  networks 
			labels- truth labels of all images tested 

	'''
	unique_list=np.unique(predictions)
	print(unique_list)
	##Filtering out all anamalous dvorak values i.e any value not ending in 0 or 5. 
	modified=False
	print('Data set shape {}'.format(Counter(labels)))
	for unique in unique_list:
		if unique%5!=0 or unique>85:
			modified=True
			print("Removing,",unique)
			index=np.where(predictions==unique)
			print(index)
			for index_ in index[0]:
				predictions=np.delete(predictions,index_,0)



	no_predictions=Counter(predictions)
	no_labels=Counter(labels)
	print(no_predictions)
	no_labels_list=[]
	no_predictions_list=[]

	unique_list=np.unique(labels)
	print(unique_list)
	print(len(unique_list))
	for unique in unique_list:
		no_labels_list.append(no_labels[unique])
		no_predictions_list.append(no_predictions[unique])
	#if len(unique_list)>8:		
	#	unique_list=np.multiply((unique_list+1),5)
	#else:
	#	unique_list=np.multiply((unique_list+1),10)

	no_predictions_list=np.divide(no_predictions_list,np.sum(no_predictions_list),dtype=np.float32)*100
	no_labels_list=np.divide(no_labels_list,np.sum(no_labels_list),dtype=np.float32)*100
	print(unique_list)

	fig = plt.figure()
	ax = plt.subplot(111)
	w = 1
	prediction=ax.bar(unique_list, no_predictions_list,width=w,color='b',align='center')
	label=ax.bar(unique_list+w, no_labels_list,width=w,color='g',align='center')
	ax.set_ylabel('Frequency(%)')
	ax.set_xlabel('Dvorak CI numbers scaled up by 10')
	ax.set_title('Distribution of data for '+model_name)
	ax.set_xticks(unique_list+w)
	ax.set_xticklabels(unique_list)

	ax.legend( (prediction[0], label[0]), ('Prior Adasyn', 'After Adasyn') )
	#plt.show()
	fig.savefig(model_name+' distribution')

labels=np.load('labels_file_name') 		#Numpy file of labels
predictions=np.load('predicitions_file_name') #Numpy file of Predictions
frequency_test(labels,predictions,'Title')





