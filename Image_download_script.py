#Title:Image Scraper
#Creatd by:Abetharan Antony
#Institution:Imperial College London
#Msc Thesis: Improving Tropical Cyclone Forecasting Using Neural Networks
#
#Objective: Download files from CIMMS archives based of .txt files containing dvorak
#data. 
#Current Algorithm has created a specfic naming convention which is utilized
#in other scripts.
#=================================================================================


import pandas as pd
import urllib #python 2
#import urllib.request #python 3
import os 
import numpy
import time as t
import math

def image_taker(path_files):
	'''	
		Purpose: Download Satellite images down from CIMMS archives:http://tropic.ssec.wisc.edu/archive/
		args.
			path_files: Path to where .txt files of corresponding images are located  
		return.
			Image files corresponding to .txt files. 
	
		'''
	URL_list={'EP': "http://tropic.ssec.wisc.edu/archive/data/NEPacific/date/IRImage/x_step.NEPacific.IRImage.png",
			'WP':"http://tropic.ssec.wisc.edu/archive/data/NWPacific/date/IRImage/x_step.NWPacific.IRImage.png",
			'AL':"http://tropic.ssec.wisc.edu/archive/data/NWAtlantic/date/IRImage/x_step.NWAtlantic.IRImage.png",
			'SH':"http://tropic.ssec.wisc.edu/archive/data/Australia/date/IRImageEast/x_step.Australia.IRImageEast.png",
			'australia_west':"http://tropic.ssec.wisc.edu/archive/data/Australia/date/IRImageWest/x_step.Australia.IRImageWest.png",
			'IO':"http://tropic.ssec.wisc.edu/archive/data/Indian/date/IRImage/x_step.Indian.IRImage.png"}

	Oceans=["EP","WP","AL","SH","IO"]
	for filename in os.listdir(path_files):

			if filename.startswith("dvk_cp") or filename.startswith("dvk_al") or filename.startswith("dvk_ep") or filename.startswith("dvk_io") or filename.startswith("dvk_sh")  :
				continue																	#Keeps track on which filename is being evaluated.  
			if filename.startswith("dvk"):
				print(filename)
				filename=os.path.join(path_files,filename)
				data=pd.read_csv(filename,sep=" ",header=None,usecols=[0,1,2,3,4,5,6,7])
				
				data.columns=["Ocean","Identity","Date and Time"," ","Latitude","Longitude"," ",
								  "Dvorak T/Current Intensity"]
				date_time=data[['Date and Time']]
				Longitude=data[['Longitude']]
				file_Ocean=str(data[['Ocean']].loc[0].values[0])


				
				for i in range(len(date_time)):															#Loop for finding all the pictures in a the pandas structure													
																										#created from the .txt file
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

					
									
			
					date=local_filename[:len(local_filename)-2]											#Finds the date
					time=local_filename[len(local_filename)-2:]
															#Finds the time
					datetime=date+"."+time
					
					longitude=str(Longitude.loc[[i]].values[0][0])

					list=[3,4,5,6]
					for j in list:
						if len(longitude)==j:
							Longitude_number=longitude[:(j-1)]
							k=len(Longitude_number)
							Longitude_input=float(Longitude_number[:(k-2)]+"."+Longitude_number[(k-2):])
						else:
							continue

					for seas in Oceans:
						if file_Ocean==seas:
							URL=URL_list[str(seas)]
							
		
					if file_Ocean=="SH":
						if Longitude_input<=120.0:
								URL=URL_list["australia_west"]

	
							

					
					URL_date=URL.replace("date",date)													#Adds the date to URL
					URL_complete=URL_date.replace("x_step",datetime)									#Add the time and completes the URL
					directory_name=str(file_Ocean)+"0"+str(data['Identity'].loc[0])+date[:4]			#Creates the folder name
						
					
					newpath = directory_name
					image_name=date+time+".png"
					

					if not os.path.exists(newpath):														#Creates new folder if folder does not exist 
					    os.makedirs(newpath)
					
				
					path_name=os.path.join(newpath,image_name)
					if os.path.exists(path_name):
						continue
					try:																				#Path_images directory where images are store, image_name=file_name
						urllib.urlretrieve(URL_complete,path_name)
						#urllib.request.urlretrieve(URL_complete,path_name)								#Python 3
					except:
						continue
						

#image_taker()
image_taker()

