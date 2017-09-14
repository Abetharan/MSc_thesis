#Title:Gathering results. 
#Created by:Abetharan Antony
#Institution:Imperial College London
#Msc Thesis: Improving Tropical Cyclone Forecasting Using Neural Networks
#
#Objective: Utilising predictions and truth labels gathered from the various neural network architectures
#gather meaningful statistics to understand the networks better. 
#The stastical tests were based on contingency table statistics, where the 
#predictions were converted into a strengthening/weakening table. 
#Numerous tests were conducted as can be seen at the end of the script.

#=================================================================================
from __future__ import division
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import os 
import time
import pickle
from textwrap import wrap

def binary_conversion(all_labels,counts,file_name_path):

	'''
	Purpose: Converts CI numbers into a binary scheme of weakening and strengthening, if its stationary i.e not doing anything those images are removed.
	Inputs: predictions  - prediction output of the  networks 
			labels- truth labels of all images tested 
			file_name_path - Where pictures are located in the file structure of image downloader.  
	How: By looking at the number of images associated with one cyclone slices the labels to match the length. 
		 Then it does a difference operation i.e subtracting neighbourhs and any numbers which are postive= strengthening
		 negative=weakening. 

	'''
	no_images=0
	


	label_conversion_array=[]
	i=0
	for file_name in os.listdir(file_name_path):
		
		
		if i==0:
			start=0
			end=counts[i]
			label_conversion_array.append(np.ndarray.tolist(np.diff(all_labels[start:end])))

		else:
			start=end
			end=counts[i]+end
			label_conversion_array.append(np.ndarray.tolist(np.diff(all_labels[start:end])))
		
		i+=1



	k=0 #To keep track of number of zeros. 
	for i in range(len(label_conversion_array)):
		for j in range(len(label_conversion_array[i])):

			check=label_conversion_array[i][j]
			if check<0:
				variable=-1
				label_conversion_array[i][j]=variable
			elif check>0:
				variable=1
				label_conversion_array[i][j]=variable

	return label_conversion_array

def contingency_table_data(truth_labels,prediction_labels,no_tropical_cyclones):
	hit=0
	miss=0
	false_alarm=0
	correct_negative=0
	true_counter=0
	fail_counter=0

	for i in range(no_tropical_cyclones):
		truth_pre=truth_labels[i]
		predictions_pre=prediction_labels[i]
		for j in range(len(truth_pre)): 
			truth=truth_pre[j]
			predictions=predictions_pre[j]
			##-1 represents weakening and 0 constant, both=no  and 1 strengthening=yes. 
			if truth==predictions:
				true_counter+=1
			else:
				fail_counter+=1

			if truth==1:
				if predictions==1:
					hit+=1
				elif  predictions==-1: #predictions==0 or
					miss+=1
			if truth==-1 : #or truth==0
				if predictions==1:
					false_alarm+=1
				elif predictions==-1 : #or predictions==0
					correct_negative+=1
	total=hit+miss+false_alarm+correct_negative
	
	return hit,miss,false_alarm,correct_negative,total,true_counter,fail_counter

def gilbert_skill_score(hits,misses,false_alarm,total):
	hits_random=((hits+misses)*(hits+false_alarm))/total
	ETS=(hits-hits_random)/(hits+misses+false_alarm-hits_random)
	return ETS

def Hansen_Kupen_discriminant(hits,misses,false_alarm,correct_negative):
	HK=(hits/(hits+misses))-(false_alarm/(false_alarm+correct_negative))
	return HK

def Heidke_skill_score(hits,misses,false_alarm,correct_negative):
	HSS=(2*(hits*correct_negative-misses*false_alarm))/((hits+miss)*(miss+correct_negative)+(hits+false_alarm)*(false_alarm+correct_negative))
	return HSS

def odds_ratio(hits,misses,false_alarm,correct_negative):
	OR=(hits*correct_negative)/(misses*false_alarm)

	return OR

def Yules_Q(hits,misses,false_alarm,correct_negative):
	ORSS=(hits*correct_negative-misses*false_alarm)/(hits*correct_negative+misses*false_alarm)

	return ORSS

predictions_file_names=['test_predictions_3_16','test_predictions_4_8','test_predictions_4_16','test_augmented_predictions_4_8','test_augmented_predictions_4_16',]

labels_file_name=['test_labels_3_16','test_labels_4_8','test_labels_4_16','test_augmented_labels_4_8','test_augmented_labels_4_16']

prediction_full_file=['1704_predictions_2_8','1704_predictions_2_16','1704_predictions_3_8','1704_predictions_3_16','1704_predictions_4_8','1704_predictions_4_16','1704_augmented_predictions_4_8','1704_augmented_predictions_4_16']

labels_full_file=['1704_labels_2_8','1704_labels_2_16','1704_labels_3_8','1704_labels_3_16','1704_labels_4_8','1704_labels_4_16','1704_augmented_labels_4_8','1704_augmented_labels_4_16']

directory='/disk1/rtproj17/results_gathering/seperate_test_numpy_results/' ##Directory of Numpy files containing predictions. 
txt_file_location='/disk1/rtproj17/test_data/txt_files' #Locations of oringal txt files 

for i in range(len(prediction_full_file)):
	
	
	prediction_full_name=prediction_full_file[i]
	label_full_name=labels_full_file[i]

	if i<6:
		model_name='Conv'+prediction_full_name[16:]
	else:
		model_name='Conv_aug'+prediction_full_name[26:]
	predictions_full=np.load(directory+prediction_full_name+'.npy')	
	labels_full=np.load(directory+label_full_name+'.npy')	
	frequency_test(labels_full,predictions_full,model_name)
	

for i in range(len(predictions_file_names)):

	prediction_name=predictions_file_names[i]
	labels_name=labels_file_name[i]
	
	predictions=np.load(directory+prediction_name+'.npy')
	labels=np.load(directory+labels_name+'.npy')
	
	files=os.listdir(txt_file_location)
	if i%2==0: #Utilised in as every second file was 8 category files. 
	#Trackers utilised to know where the test files change to another TC. 
		tracker=pickle.load(open("directory_of_tracker/tracker_for_difference_8.p","rb"))
	else:
		tracker=pickle.load(open("directory_of_track/tracker_for_difference_16.p","rb"))
	counts=[]

	for file in files:
		counts.append(tracker.count(file))

	truth_labels=binary_conversion(labels,counts,txt_file_location)
	prediction_labels=binary_conversion(predictions,counts,txt_file_location)

	hits,miss,false_alarm,correct_negative,total,true_counter,fail_counter=contingency_table_data(truth_labels,prediction_labels,25)
	print('------------------------')
	print(predictions_file_names[i])
	print('hits',hits)
	print('misses',miss)
	print('false negative',false_alarm)
	print('correc negative',correct_negative)
	print('total',total)
	print('true_counter',true_counter)
	print('fail_counter',fail_counter)
	if total ==0 or miss+false_alarm==0:
		continue
	
	'''
	skill score measures of each network. 
	'''
	accuracy=(hits+correct_negative)/total

	bias_score=(hits+false_alarm)/(hits+miss)

	hit_rate=hits/(hits+miss)

	false_alarm_ratio=false_alarm/(hits+false_alarm)

	false_alarm_rate=false_alarm/(correct_negative+false_alarm)

	success_ratio=hits/(hits+false_alarm)

	critical_success_index=hits/(hits+miss+false_alarm)

	gilbert_score=gilbert_skill_score(hits,miss,false_alarm,total)

	HKD=Hansen_Kupen_discriminant(hits,miss,false_alarm,correct_negative)

	Cohen_k=Heidke_skill_score(hits,miss,false_alarm,correct_negative)

	OR=odds_ratio(hits,miss,false_alarm,correct_negative)

	YQ=Yules_Q(hits,miss,false_alarm,correct_negative)
	
	'''
	NOTE that the reason why constant strength is to prevent a case of Finley (1884) Tornado Forecasts which resulted in inflated 
	accuracy scores because of no tonados. Did not want to get a case where the scores were inflated due to correct predictions in constant values. 
	'''
	
	print('------------------------')
	print('accuracy',accuracy)
	print('bias_score',bias_score)
	print('hit_rate',hit_rate)
	print('false alarm ratio',false_alarm_ratio)
	print('false alarm rate',false_alarm_rate)
	print('success ratio',success_ratio)
	print('critical success index',critical_success_index)
	print('gilbert skill score',gilbert_score)
	print('hansen kupen discriminant',HKD)
	print('Cohen k',Cohen_k)
	print('Odds ratio',OR)
	print('Yules q',YQ)
	print('------------------------')


