#Title:Balancing Pre-Processing module for Tropical cyclone forecasting
#Creatd by:Abetharan Antony
#Institution:Imperial College London
#Msc Thesis: Improving Tropical Cyclone Forecasting Using Neural Networks
#
#Objective: Testing if data is unbalanced and correcting the data such to remove the 
#bias present using ADASYN.
#Current Algorithm utilises ADASYN to syntically oversample and undersample the normal 
#way and clean up use Edited nearest neighbourgh. 
#=================================================================================


import numpy as np 
import matplotlib.pyplot as plt
import time as t
from random import randint
from collections import Counter
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split
labels=np.load('labels_192.npy')
data=np.load("pixel_array_192.npy")

#print('Original Data set shape {}'.format(Counter(labels)))

unique_list=np.unique(labels)
print(unique_list)
##Filtering out all anamalous dvorak values i.e any value not ending in 0 or 5. 
modified=False
print('Data set shape {}'.format(Counter(labels)))
for unique in unique_list:
	if unique%5!=0 or unique>85:
		modified=True
		print("Removing,",unique)
		index=np.where(labels==unique)
		print(index)
		for index_ in index[0]:
			labels=np.delete(labels,index_,0)
			data=np.delete(data,index_,0)
if modified==True:
	print('Modified Data set shape {}'.format(Counter(labels)))

training_data, eval_data,training_labels, eval_labels = train_test_split(
	data, labels, test_size=0.2, random_state=150)

np.save("eval_data_ADASYN_192",eval_data)
np.save("eval_labels_ADASYN_192",eval_labels)

validation_data,testing_data,validation_labels,testing_labels=train_test_split(
		eval_data,eval_labels,test_size=0.5,random_state=10)
print('Training data shape {}'.format(Counter(training_labels)))
print('Validation Data shape {}'.format(Counter(validation_labels)))
print('Testing Data shape {}'.format(Counter(testing_labels)))




index=np.where(validation_labels==80)
training_data=np.append(training_data,validation_data[index[0]],axis=0)
training_labels=np.append(training_labels,validation_labels[index[0]],axis=0)

print('Over sampled Training data shape {}'.format(Counter(training_labels)))

print('Training data shape {} Training label shape {}'.format(np.shape(training_data),np.shape(training_labels)))


ada=ADASYN(random_state=42,n_jobs=18)
print('Starting ADASYN')
X_res,y_res=ada.fit_sample(training_data,training_labels)
print('Resampled Data set shape {}'.format(Counter(y_res)))

##Save for use in neural network
np.save("labels_ADASYN_192",y_res)												 
np.save("pixel_array_ADASYN_192",X_res)









