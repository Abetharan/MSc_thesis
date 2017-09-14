#Title:Pre-Process
#Creatd by:Abetharan Antony
#Institution:Imperial College London
#Msc Thesis: Improving Tropical Cyclone Forecasting Using Neural Networks
#
#Objective: To focus image onto tropical cyclone using data from CIMMS group. 
#Additionally, puts in truth dvorak values into array Dvorak_values. 
#Further, filters the values to ensure only correct dvorak_values are present. 
#Current Algorithm has specifcs based on naming convention of the Image_Scraper.py 
#script. Percularities in the data.txt files lead to very messy code and slight
#abuse of try and except. 
#=================================================================================

#Libraries used#

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
import os
import pandas as pd
import time as t
from collections import Counter
import pickle

##Dimension[680,1080] every 111 represents 20 degrees 

def grid_line_deleter(im,ocean):
	'''	
	Purpose: Removes white grid lines found in satellite image
	args.
		im:Image file PIL opened
		ocean: Ocean satellite image is from
		
	return.
		data: Returns back data with white lines removed 
		resolution: number of pixels for 1 degree
		zero_degree: location of 0 degrees (only works for indian ocean)
		number_of_lines: number of white lines present along x 

	'''

	data=np.asarray(im)
	plt.imshow(data,cmap='gray',interpolation='nearest')
	plt.show()
	x=np.concatenate(np.where(data[0,:]==255))

	if ocean=="EP":
		black_line_finder_y=np.concatenate(np.where(data[:,0]==1))
		y=np.concatenate(np.where(data[:,4]==255))

	else:															
		y=np.concatenate(np.where(data[:,0]==255))
	
	if 'black_line_finder_y' in locals():
		data=np.delete(data,black_line_finder_y,axis=0)
	number_of_lines=len(x)
	data=np.delete(data,y,axis=0)																			
	data=np.delete(data,x,axis=1)

	if ocean=="EP":
		data=data[y[0]:,x[0]:x[len(x)-1]]
	else:
		try:
			data=data[y[0]:y[len(y)-1],x[0]:x[len(x)-1]]
		except:
			try:
				y=np.concatenate(np.where(data[:,40]==255))
				data=data[y[0]:y[len(y)-1],x[0]:x[len(x)-1]]
			except:
				data=np.zeros((2,2))
				data[0,0]=-1
				pass
	try:																									#Try except structure above and below required for images that are not able to find anything.
		resolution=(x[2]-x[1])/10																			#Typical reasons involve broken images or images covered in black or other colors and thus cannot detect the 
		zero_degree=y[3]																					#the lines. To allow for continueing computation I utilised the try and except. 
	except:
		resolution=None
		zero_degree=None
	plt.imshow(data,cmap='gray',interpolation='nearest')
	plt.show()
	return data, resolution, zero_degree,number_of_lines

def eye_focus(data,longitude,latitude,resolution,ocean,zero_degree,number_of_lines):
	'''	
	Purpose: Centres image onto eye using lat/long data frmo .txt data files 
	args.
		data: Data with grid lines removed
		longitude: Longitude as string
		latitude: Latitude as string
		resolution: number of pixel in a degree
		ocean: ocean satellite image is from
		zero_degree: location of centre (only relevant for indian ocean)
		number_of_lines: no of grid lines along x
	return.
		zoomed_image: An image focused onto the centre of the eye of a tropical cyclone
		padding: Logic statement if picture is not specified size 

	'''



	start_point_longitude_dict={"SH_west_8":80,"SH_west_10": 70,"SH_east":120,"AL":100,"EP":170,"WP":100,"IO":40} 
	if zero_degree>np.shape(data)[0]:
		raise ValueError("Zero_degree is wrong check input")
	
	north_south=latitude[len(latitude)-1]
	east_west=longitude[len(longitude)-1]
	list=[2,3,4,5,6]
	
	for i in list:
		if len(latitude)==i:
			Latitude_number=latitude[:(i-1)]
			j=len(Latitude_number)
			Latitude_input=float(Latitude_number[:(j-2)]+"."+Latitude_number[(j-2):])
		else:
			continue

	for i in list:
		if len(longitude)==i:
			Longitude_number=longitude[:(i-1)]
			k=len(Longitude_number)
			Longitude_input=float(Longitude_number[:(k-2)]+"."+Longitude_number[(k-2):])
		else:
			continue
	print("longitude:", Longitude_input)
	print("latitude:", Latitude_input)
	#if Longitude_input>start_point_longitude_dict[str(ocean)]:
	#	raise ValueError("Longitude outside boundary")
	#if Latitude_input>50:
	#	raise ValueError("Latitude outside boundary")


	if ocean=="SH":																								#Require to differentiate between two satellite images which correspond to  
		if Longitude_input<=120:
			if number_of_lines<10:																				#east/west side of australia.
				ocean="SH_west_8"
			else:
				ocean="SH_west_10"
		else:
			ocean="SH_east"

	if ocean=="SH_east":																						#Australia east satellite images span from E to W and thus require this for logical
		if east_west=="W":																						#indexing in the algorithm I am using. 
			if Longitude_input>=160:
				adder=180-Longitude_input
				Longitude_input=180+adder

	if ocean=="IO":
		print("Indian ocean algorithm")
		if north_south=="N":																					#Depending on whether coordinates or North or south bound add or subtract from 
			index_row=zero_degree-int(round(Latitude_input*resolution))											#zero degree line found from the grid line delete function above. 30 degrees 
		else:																									#above and below in these pictures. 
			index_row=zero_degree+int(round(Latitude_input*resolution))
	else:
		if north_south!="N":
			index_row=int(round(Latitude_input*resolution))			
		else:
			if ocean=="EP":
				start_latitude=60
			else:
				start_latitude=50

			index_row=int(round(abs(start_latitude-Latitude_input)*resolution))


	index_column=int(round(abs(Longitude_input-start_point_longitude_dict[str(ocean)])*resolution))				#Zero index column represents the start_point degrees and thus, require to subtract.


	number_of_pixels=int(124) 																					#Number of pixels each side of the centre
	print("Centering Image")

	#while number_of_pixels>index_row or number_of_pixels>index_column:											#Loop checks if full size image is available otherwise reduces, number_pix
	#	number_of_pixels+=-2																										 
	

	zoomed_image=data[(-number_of_pixels+index_row):(number_of_pixels+index_row),								#Chooses the data corresponding to center
					  (-number_of_pixels+index_column):(number_of_pixels+index_column)]
	
	if number_of_pixels!=96 or np.shape(zoomed_image)[0]!=192 or np.shape(zoomed_image)[1]!=192 :				#As we only want same size pictures with no padding use this if statement to check
		padding=True
	else:
		padding=False																			

	plt.imshow(zoomed_image,cmap='gray',interpolation='nearest')
	plt.show()
	return zoomed_image,padding


def picture_analyser(txt_path,image_path):

	'''	
	Purpose: Structures extracts data to centre satellite images onto eye of tropical cyclone.
			 As well as, extracting dvorak values as labels. 
			 Everything, gets put into numpy array.
			 For use immediately in neural networks using Tensorflow. 
	args.
		txt_path: Location of .txt data files taken from CIMMS archives 
		image_path: Location of satellite image using naming convention from 
					image_download_script.py
		
	return.
		pixel_information: Numpy array with structure [no_data,flat_image]
		labels: Numpy array with structure [no_data]
		Put into 32-bit form for use in Tensorflow 
	'''
	
	bad_image_list=[]
	size_fail=[]
	pixel_information=[]
	labels=[]
	dvorak_fail_counter=[]
	pixel_information_not_200=[]
	labels_not_200=[]
	tracker_for_difference=[]
	###Txt file loader###

	for filename_txt in os.listdir(txt_path):
		if filename_txt.startswith("dvk_cp") or filename_txt.startswith("dvk_wp") or filename_txt.startswith("dvk_sh") or filename_txt.startswith("dvk_ep") or filename_txt.startswith("dvk_al"):
			continue		
		if filename_txt.startswith(".DS"):
			continue

		print("filename being read: ",filename_txt)													##Recall filename refers to the .txt file 

		filename_txt_original=filename_txt
		filename_txt=os.path.join(txt_path,filename_txt)											##Required to open the txt file
		data=pd.read_csv(filename_txt,sep=" ",header=None,usecols=[0,1,2,3,4,5,6,7])
		
		data.columns=["Ocean","Identity","Date and Time"," ","Latitude","Longitude"," ",
						  "Dvorak T/Current Intensity"]
		date_time=data[['Date and Time']]															#Extract the variables required for future use
		Dvorak_pandas=data[['Dvorak T/Current Intensity']]
		latitude=data[['Latitude']]
		longitude=data[['Longitude']]
		ocean=data[['Ocean']]
		
		if type(Dvorak_pandas.loc[0].values[0])==str:
			print("filename being skipped due to no dvorak values: ",filename_txt_original)
			continue



		##Image loader##
		
		searcher=filename_txt_original[6:12]
		if str(ocean.loc[[0]].values[0][0])=="SH":
			if 	int(searcher[0:2])==4 and int(searcher[2:])==2017 or int(searcher[0:2])==3 and int(searcher[2:])==2017 or int(searcher[0:2])==5 and int(searcher[2:])==2017 or int(searcher[0:2])==5 and int(searcher[2:])==2015 :
				searcher=searcher
			elif int(searcher[0:2])<=5 and int(searcher[2:])<=2017:
				new_year=str(int(searcher[1:])-1)
				searcher=searcher[0:1]+new_year														#Exctracts ID and year 
		
		if int(searcher[0:2])<10:
			image_directory=str(ocean.loc[[0]].values[0][0])+searcher 								##Finds the directory of the folders where the picture corresponding to the				
		else:
			image_directory=str(ocean.loc[[0]].values[0][0])+"0"+searcher 

		image_directory=os.path.join(image_path,image_directory)									
		j=0
		for filename_image in os.listdir(image_directory):											#Runs through all the images for same identity and year as txt file			
			filename_image_original=filename_image
		
			for i in range(len(date_time)):															#Loops finds the the time and date which correspond to image
																									#that is being loaded in and finds the corresponding data index
																									#required
									
				date=str(date_time.loc[[i]].values[0][0])[:8]										#Seperates the time and date
				time=str(date_time.loc[[i]].values[0][0])[8:]
				

				if time[0]!=str(0):																	#Due to the .txt files all being 30 minutes before the satellite images
																									#Require to round up the number to the nearest 1000 to be in line with satellite image
					time_path=str(int(math.ceil(int(time) / 100.0)) * 100)
					
					
					if time_path==str(2400):														##24th hour is 00 in satellite image 
						time_path="00"
						
						
					time_path=time_path[:2]															#Only the first 2 indices have been used to label the images 
				

				elif time[2]!=str(0):																#Required as all integers below 10 was giving odd rounding errors
					
					time_path=time[:2]
					time_path="0"+str(int(time_path)+1)
					

				else:																				#Some time paths did not require roundings. 
					time_path=time[:2]

				

				if time_path[1]==str(0):															#If it is the 24th hour day changes hence this statement
				
					local_filename=str(int(date)+1)+"00"											#Local file name defined as the time and date of a row in pandas structure
			 

				else:
					local_filename=date+time_path
				

				
				if local_filename==filename_image[:10]:												#Tries to match the local filename i.e a row of pandas to the filename
					j+=1																			#Leave the loop if an image and corresponding pandas data row 
					print("Match found", j, "out of ",len(date_time))								#Has been found. 		
					if j==len(date_time):
						print("all pictures have been matched")
						t.sleep(1)
					break

			if local_filename != filename_image[:10]:																						#If no image file correspond to any of the pandas dates continue to next image
				
				print("NO picture corresponding to any of the pandas",local_filename,image_directory)
			
				continue

			#Imports image and deletes the grid liens

			filename_image=os.path.join(image_directory,filename_image)																		
			print("Loading image",local_filename)

			try:
				im = Image.open(filename_image).convert("L")																				#Loads image and gray scale filters it. 
			except:
				continue

			Ocean_input=ocean.loc[[i]].values[0][0]	
			data,resolution,zero_degree,number_of_lines=grid_line_deleter(im,Ocean_input)													#Applies my grid-linear deleter function
			if data[0,0]==-1:
				bad_image_list.append((local_filename,filename_txt_original))
				continue													
			#Zooms in on the storm
			
			Latitude_input=latitude.loc[[i]].values[0][0]																					#Finds the latitude corresponding to the image
			Longitude_input=longitude.loc[[i]].values[0][0]																					#Finds the longitude corresponding to the image
			zoomed_data,padding=eye_focus(data,Longitude_input,Latitude_input,resolution,Ocean_input,zero_degree,number_of_lines)					#Centering on cyclone function 
			zoomed_data=(zoomed_data.ravel())
			
			if padding==True:
				size_fail.append((local_filename,filename_txt_original))
				pixel_information_not_200.append(zoomed_data)
			else:																							#Require the pixel data to be 1d array thus using ravel.
				pixel_information.append(zoomed_data)																							#Append the pixel information into an array for CNN
		
			Dvorak_value_full=str(Dvorak_pandas.loc[[i]].values[0][0])																		#Finds the Dvorak_T value corresponding to the image														

			dvorak_len=[2,3,4]
			print("Dvorak in full form:",Dvorak_value_full)	
			
			if len(Dvorak_value_full)>4:	
				Dvorak_value=int(Dvorak_value_full[2]+Dvorak_value_full[3])	
			else:
				for i in dvorak_len:
					if len(str(Dvorak_value_full))==i:
						z=i-1
						if "." in Dvorak_value_full:
							Dvorak_value=int(Dvorak_value_full[0]+Dvorak_value_full[1])
						else:
							try:
								Dvorak_value=int(Dvorak_value_full[z-1]+Dvorak_value_full[z])
							except:
								dvorak_fail_counter.append((local_filename,filename_txt_original))

			#Percularity added due to reduced data points in dvorak values less than 5 and above 7, gathered them into one to increase group size. Not-Ideal. 
				
				print("Current Intensity:",Dvorak_value)
			

			print('Rounded Dvorak value is: {}'.format(Dvorak_value))

			if padding==True:
				labels_not_200.append(Dvorak_value)
			else:
				labels.append(Dvorak_value)													    #Appends the label i.e Dvorak T Label from txt file to the image
				tracker_for_difference.append(filename_txt_original)
	
	print("number of lost pictures at dvorak:",np.shape(dvorak_fail_counter)[0])
	print("number of pictures lost at line clean up",np.shape(bad_image_list)[0])
	print("number of pictures with uneven shape",np.shape(size_fail)[0])
	print("labels_not_200",np.shape(labels_not_200))
	


	print('data spread {}'.format(Counter(labels)))

	
	labels_not_200=np.asarray(labels_not_200,dtype=np.int32)
	labels=np.reshape(labels,np.shape(labels))
	

	print('data shape {}'.format(np.shape(pixel_information)))
	print('label shape{}'.format(np.shape(labels)))
	
	##Filtering out all anamalous dvorak values i.e any value not ending in 0 or 5. 


		

	corrected_array=[]
	correct_labels=[]							
	counter=-1						

	for item in pixel_information:															#Filtering loop in case any images of the wrong pixel dimension is added. 
		counter+=1																			#All image should be 40000 long for 200x200 image. 
		if len(item)!=36864:																#Padding algorithm might be wrong and thus produce incorrect images. 
			print(np.shape(item))
		else:
			correct_labels.append(labels[counter])
			corrected_array.append(item)


	pixel_information=np.asarray(corrected_array,dtype=np.float32)										#Require float32/int32 and above input for tensorflow 
	pixel_information=np.reshape(pixel_information,(np.shape(pixel_information)[0],36864))
	
	labels=np.asarray(correct_labels,dtype=np.int32)
	labels=np.reshape(labels,np.shape(labels))
	pickle.dump(tracker_for_difference,open('tracker_for_difference_16.p','wb'))



	np.save("labels_192_16_label_same_ocean_WP",labels)															#saving for faster network training. 
	np.save("pixel_array_192_16_label_same_ocean_WP",pixel_information)
	
	
	return pixel_information,labels


pixel_information,labels=picture_analyser()

