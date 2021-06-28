# Code by Vincent Francis and Sethu K. Boopathy Jegathambal

import cv2
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
from skimage import morphology
from scipy.spatial import ConvexHull
import subprocess as sp
from skimage.morphology import convex_hull_image
import scipy.misc
import datetime
import scipy.io as sio

def euclidDist(center,pint):

	"""
	Function to calculate euclidean distance between two points.
	Input: Two Points
	Output: distance
	"""
	c1,c2=center
	p1,p2=pint
	distance=(float(c1-p1)**2.0+float(c2-p2)**2.0)**0.5
	return distance

def InvertMaskAndCluster(image):

	"""
	Invert the provided mask and cluster (connectivity based), in order to remove small isolated speckles in the image.
	"""

	#print("Type of input variable:",type(image))
	#print("Image pixel datatype:",image.dtype)
	# Inverting the mask.
	binary_inverse=np.invert(image.astype(np.uint8))
	# Clustering
	labels,nums=measure.label(binary_inverse,background=0,return_num=True)
	#print("labels,nums:",labels,nums)
	calc_area=[]
	# Iterated over available number of clusters
	for i in range(1,nums+1):
		calculated_area=np.sum(labels==i) # Counting the number of pixels with a particular label
		calc_area.append(calculated_area) # Appending to a list

	# Keep the max area cluster
	max_value = max(calc_area) # finding the biggest cluster i.e the background silhouette
	max_index = calc_area.index(max_value) # index of the item is recorded
	binary_inverse=np.multiply(binary_inverse,labels==(max_index+1)) # Remove all other labels (image1(multiple labels),image2(binary))
	#print("Filling holes, number of clusters in the this image: "+str(nums))
	if view_plots==True:
		plt.title("Result from Function InvertMaskAndCluster")
		plt.imshow(binary_inverse)
		plt.show()
	filled=np.invert(binary_inverse)
	if view_plots==True:
		plt.title("Result from Function InvertMaskAndCluster")
		plt.imshow(filled)
		plt.show()

	return filled

def clusterAndSave(image,image_clus_info):
	"""
	Function to find clusters of cells and save the lysosome distribution
	"""
	print("Cluster and save function")
	labels,nums=measure.label(image,background=0,return_num=True)
	plt.imshow(labels,cmap="CMRmap")
	plt.show()
	calcs=[]
	valid_labels=[]
	numClus=0
	print("Number of clusters: ", nums)
	for i in range(1,nums+1):
		calculated_area=np.sum(labels==i)
		calcs.append(calculated_area)
		if calculated_area>cell_size_thresh:
			plt.imsave('current_image.png',im1+im2)
			plt.subplot('121')
			plt.imshow(labels==i)
			plt.title("Current cluster: "+str(i))
			plt.subplot('122')
			plt.imshow(im1+im2)
			plt.title("Sum of two channels")
			plt.show()

			answ=raw_input("Append as valid cluster? (y/n)").lower()

			if answ[0]=='y':
				simplified_image_summary.append([file_name1])
				simplified_image_summary.append([numClus])
				#convex_hull_image
				numClus+=1
				print("current cluster and area ",i,calculated_area)
				curr_clus_info=[]
				valid_labels.append(i)
				# 0:First item is cluster number
				curr_clus_info.append(i)
				points=np.transpose(np.where(labels==i))
				label_canvas=np.zeros(image.shape)
				for pts in points:
					label_canvas[pts[0],pts[1]]=1
				label_canvas=InvertMaskAndCluster(label_canvas)

				if method=="contourApproximation":
					outerpoints=contourApproximation(label_canvas)
					hull=None
				else:
					outerpoints=[]
					hull = ConvexHull(points)
					for simplex in hull.vertices:
						a1=points[simplex,0].astype('float32')#.tolist();
						a2=points[simplex,1].astype('float32')#.tolist()
						new_pt=(a2,a1) # returns y,x
						outerpoints.append(new_pt)
				pts = np.array(outerpoints, np.int32)
				pts = pts.reshape((-1,1,2))
				if testing==True:
					print("pts:",pts);raw_input("checking points order")
				img_hull=cv2.polylines(0*imgray.copy(),[pts],True,(255,255,255),1,1)

				if view_plots==True:
					plt.imshow(img_hull.reshape(imgray.shape));
					plt.title("Contour that will be used")
					plt.show()
				#plt.imsave('temp_center_pick.png',imgray1+imgray2+2*img_hull.reshape(imgray.shape)) #Changed this line to save as two different images.
				if folder3!=None:
					overlay_image=np.zeros(im1.shape)
					overlay_image[:,:,0]=img_hull.reshape(imgray.shape)
					overlay_image[:,:,1]=img_hull.reshape(imgray.shape)
					overlay_image[:,:,2]=img_hull.reshape(imgray.shape)
					plt.imsave("temp.png",overlay_image)
					overlay_image = cv2.imread("temp.png")
					plt.imsave('temp_center_pick.png',im1+im2+im3+overlay_image)
				else:
					overlay_image=np.zeros(im1.shape)
					overlay_image[:,:,0]=img_hull.reshape(imgray.shape)
					overlay_image[:,:,1]=img_hull.reshape(imgray.shape)
					overlay_image[:,:,2]=img_hull.reshape(imgray.shape)
					plt.imsave("temp.png",overlay_image)
					overlay_image = cv2.imread("temp.png")
					plt.imsave('temp_center_pick.png',im1+im2+overlay_image)

				# 1: Image of the cluster:
				curr_clus_info.append(1*(labels==i))
				# 2:Maybe pick center and append that here.
				cluster_cell_center=getCenter(os.path.join(os.getcwd(),'temp_center_pick.png'))
				curr_clus_info.append(cluster_cell_center)
				# 3: Points in the hull
				curr_clus_info.append(outerpoints)
				# 4: Points in the hull reshaped
				curr_clus_info.append(pts)
				# 5: Image of the hull
				curr_clus_info.append(img_hull)
				# Attaching the summary of converging hull to the list
				if method=="contourApproximation":
					summaryOf_hull=convergingCA(cluster_cell_center,pts)
					"""
					Repeat one more time for getting the Contout approximation(CA)
					"""
					CAintensities_list=[]
					CorrespondingIntensitiesofCAlevels={}
					converging_contour_fig=np.zeros(im1.shape)
					for level in range(0,nfrag):
						CAlevel_mask,contr_image_returned=getFilledCA(summaryOf_hull,level)
						converging_contour_fig[:,:,0]=converging_contour_fig[:,:,0]+contr_image_returned
						CorrespondingIntensitiesofCAlevels[level]=CAlevel_mask
						CAintensities_list.append(CAlevel_mask)
					plt.imsave(os.path.join(output_path,tag+'_img_'+str(imgnum)+'_clus_'+str(i)+'_converging_CA.png'),converging_contour_fig)
					simplified_image_summary.append(CAintensities_list)
					image_clus_info.append(CorrespondingIntensitiesofCAlevels)
					# This is attached to the summary list of the image
					image_clus_info.append(curr_clus_info)
				else:
					summaryOf_hull=converging_hull(cluster_cell_center,pts)
					# Attaching the converging points for the cluster
					# Final Point being the point and the first point being the 10th of the distance
					image_clus_info.append(summaryOf_hull)
					intensities_list=[]
					CorrespondingIntensitiesofCONVlevels={}
					for level in range(0,nfrag):
						convlevel_mask=getfilledHull(summaryOf_hull,level)
						CorrespondingIntensitiesofCONVlevels[level]=convlevel_mask
						intensities_list.append(convlevel_mask)
					simplified_image_summary.append(intensities_list)
					image_clus_info.append(CorrespondingIntensitiesofCONVlevels)
					image_clus_info.append(curr_clus_info)
			else:
				hull=0
	simplified_summary.append(simplified_image_summary)
	image_clus_info.append(numClus) # Last element gives the number of usable cluster in the image
	props=measure.regionprops(labels, cache=True)
	return props,calcs,hull,image_clus_info

def convergingCA(center,CA_pts):
	contourPointsSummary={}
	print("In convergingCA function")
	CA_pts_restructured=[]
	for count,item in enumerate(CA_pts):
		pt_restruct=(item[0][0],item[0][1])
		distp2c=euclidDist(center,pt_restruct)
		#print("Distance from point to center",distp2c)
		CA_pts_restructured.append((item[0][0],item[0][1]))
		list_of_points=[]
		for i in range(1,nfrag+1):
			step_point=vectorMethod(center,pt_restruct,i*1.0/nfrag)
			#print("center: ",center,"point: ",pt_restruct,"fraction of distance: ",i*1.0/nfrag,"new step point: ",step_point)
			list_of_points.append(step_point)
		contourPointsSummary[str(count)]=list_of_points
	#print("Keys: ",contourPointsSummary.keys())
	#print(contourPointsSummary)
	return contourPointsSummary

def getFilledCA(summary,lvl): # Not complete
	pt_list=[]
	for key in sorted(map(int,summary.keys())):
		key=str(key)
		#print(sorted(map(int,summary.keys())))
		fraglist=summary[key]
		#print("key: ",key);print("fraglist");print(fraglist);print("[fraglist[lvl]]:",[fraglist[lvl]])
		pt_list.append([fraglist[lvl]])
	pt_list = np.array(pt_list, np.int32)
	if testing==True:
		print(pt_list.shape)
		raw_input("See struct")
	img_approximation=cv2.polylines(0*imgray.copy(),[pt_list],True,(255,255,255),1,1)
	CA_mask_image=InvertMaskAndCluster(img_approximation)
	if view_plots==True:
		plt.subplot('131')
		plt.title("Using cv2.polylines")
		plt.imshow(img_approximation)
		plt.subplot('132')
		plt.title("After filling with invert hull mask")
		plt.imshow(CA_mask_image)
		plt.subplot('133')
		plt.title("Multiplied with im2")
		plt.imshow((CA_mask_image>0)*im2[:,:,ch2])
		plt.show()
	print("Sum of intensity using contour approximation:", np.sum((CA_mask_image>0)*im2[:,:,ch2])) # Channel number can be problematic
	return np.sum((CA_mask_image>0)*im2[:,:,ch2]),img_approximation

def converging_hull(center,hull_points):
	# input the hull points onto an image and  call this function and check it the results are similar
	# points in the image are calculated by using the distance formula.
	# Create dictionary
	contourPointsSummary={}
	# iterate over the hull points
	for hull_point in hull_points:
		pint=np.ndarray.tolist(hull_point[0]) # point of interest
		distp2c=euclidDist(center,pint)
		#print("Distance from point to center",distp2c)
		list_of_points=[]
		for i in range(1,nfrag+1):
			step_point=vectorMethod(center,pint,i*1.0/nfrag)
			#print("center",center,"point",pint,"fraction of distance",i*1.0/nfrag,"new step point",step_point)
			list_of_points.append(step_point)
		contourPointsSummary[str(hull_point[0])]=list_of_points
	print("Keys: ",contourPointsSummary.keys())
	print(contourPointsSummary)
	if testing==True:
		raw_input("cnts")
	return contourPointsSummary

def getfilledHull(summary,lvl):
	pt_list=[]
	for key in summary.keys():
		pt_list.append(summary[key][lvl])
	pt_list=np.transpose(np.transpose(np.array(pt_list)))
	hul=ConvexHull(pt_list)
	outerpoints=[]
	for simplex in hul.vertices:
		a1=pt_list[simplex,0].astype('float32') #.tolist();
		a2=pt_list[simplex,1].astype('float32') #.tolist()
		new_pt=(a1,a2) # returns y,x
		outerpoints.append(new_pt)
	pts = np.array(outerpoints, np.int32)
	pts = pts.reshape((-1,1,2))
	img_hull=cv2.polylines(0*imgray.copy(),[pts],True,(255,255,255),thickness=1,lineType=1)
	if view_plots==True:
		plt.imshow(img_hull.reshape(imgray.shape))
		plt.show()
		plt.subplot('131')
		plt.title("Using cv2.polylines")
		plt.imshow(img_hull)
		plt.subplot('132')
		plt.title("After filling with invert hull mask")
		plt.imshow(convex_hull_image(img_hull))
		plt.subplot('133')
		plt.title("Multiplied with im2")
		plt.imshow(convex_hull_image(img_hull)*im2[:,:,ch2])
		plt.show()
	plt.imsave('temp_center_pick.png',imgray1+imgray2+img_hull.reshape(imgray.shape))
	print("Sum of intensiry using convex hull method:", np.sum(convex_hull_image(img_hull)*im2[:,:,ch2]))
	return np.sum(convex_hull_image(img_hull)*im2[:,:,ch2])

def getCenter(file_name):
	"""
	Function calls an external code execution: Runs click_and_crop.py with input argument as a image filename
	"""
	centerInput='invalid'
	# To be put inside the function.
	while centerInput=='invalid':
		centerread=sp.Popen(['python', '.\\click_and_crop.py','-i',file_name],stdout=sp.PIPE)
		center=centerread.stdout.read().decode('utf-8').strip("[()]'\n\r").split(",")
		print("In main program...",center)
		if len(center)==2:
			centerInput='valid'
			center=map(int,center)
			return center
		else:
			if testing==True:
				return (635,480)
			else:
				print("Problem with click_and_crop.py")
				exit(0)

# P = d(B - A) + A
def vectorMethod(center,pint,d):
	returned_point=[(float(d)*(pint[0]-center[0]))+center[0],(float(d)*(pint[1]-center[1]))+center[1]]
	return map(int,returned_point)

def contourApproximation(approx_input,step_size=0.005):
	"""
	Get an approximate the contour of the cell image.
	This allows us to reduce the number of points and increase the speed of execution.
	Change the step_size arguement to get a suitable approximation.
	"""

	img = approx_input
	ret,thresh = cv2.threshold(img,127,255,0)
	imc2,contours,hierarchy = cv2.findContours(thresh, 1, 2)
	curr_len=0
	for item in contours:
		# Pick the longest contour if there are many.
		if len(item)>curr_len:
			curr_len=len(item)
			cnt=item
	# Epsilon constrols the approximation accurarcy
	epsilon = step_size*cv2.arcLength(cnt,True)
	approx = cv2.approxPolyDP(cnt,epsilon,True)
	zeros_image=np.zeros(imc2.shape)
	useable_points=[]
	for num in range(len(approx)):
	    useable_points.append((approx[num][0][0],approx[num][0][1]))
	useable_points_initial=useable_points
	useable_points = np.array(useable_points, np.int32)
	useable_points = useable_points.reshape((-1,1,2))
	if testing==True:
		print("Reordered the points to reflect the pixel IDs of the points on the countour.")
		print(useable_points);
	# Method 1: Using cv2.polylines
	img_approximation1=cv2.polylines(0*zeros_image.copy(),[useable_points],True,(255,255,255),10,1)
	# Method 2: Using cv2.drawContours - works too
	#img_approximation2=cv2.drawContours(0*zeros_image.copy(),[approx],0,(255,255,255),10)
	if view_plots==True:
		plt.subplot('121')
		plt.title("Approximated contour")
		plt.imshow(img_approximation1)
		plt.show()

	return useable_points_initial

def SaveResults(npy_file,save_as,consol,consol_saveas):
	"""
	Saving the results.
	"""
	print("Results")
	dataPlot=np.load(npy_file,allow_pickle=True)
	print("Input data shape: ", dataPlot.shape)
	shp=dataPlot.shape;temp_matrix=[];temp_normalized_matrix=[]
	for item1 in range(shp[0]):
		for item2 in range(2,len(dataPlot[item1]),3):
			temp_matrix.append(dataPlot[item1][item2])
			temp_normalized_matrix.append(np.multiply(dataPlot[item1][item2],1.0/dataPlot[item1][item2][-1]))
	np.save(save_as,temp_matrix)
	np.save(save_as+"_normalized",temp_normalized_matrix)
	np.save(consol_saveas,consol)


method="contourApproximation" # contourApproximation, Hully
consolidated={}
#Folders of lysosome channel  and cellbody channels
#Lysosome folder

folder1="F:\\JP_KO_quantification_16-10-2020-Copy\\WT-hela-unstarved\\output\\ch1"
#Cellbody Folder
folder2="F:\\JP_KO_quantification_16-10-2020-Copy\\WT-hela-unstarved\\output\\ch2-inked-png"
# Optional folder 3 - if nucleus image is available - Assign None otherwise
folder3="F:\\JP_KO_quantification_16-10-2020-Copy\\WT-hela-unstarved\\output\\ch3" #None


ch1=2
ch2=1
cell_size_thresh=5000
output_path=".\\output\\"+datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
output_folder=os.makedirs(output_path)
tag=raw_input("Output file tag (specify the expt type): ")
testing=False # Used while testing the code. To check the vatiables during execution
view_plots=False # Produces plots of the intermediary results. Good way to understand the pipeline


nfrag=20 # number of steps from center to the periphery
simplified_summary=[]
simplified_intensity_summary=[]
print('Number of images in folder; ',len(os.listdir(folder1)))

# for image in folder1:
for imgcnt,image in enumerate(os.listdir(folder1)):
	simplified_image_summary=[]
	imgnum=imgcnt+1
	Center=[]
	curr_image=[]

	file_name2=os.path.join(folder1,image)
	file_name1=os.path.join(folder2,image)
	print("Filename1:",file_name1)
	print("Filename2:",file_name2)
	im1 = cv2.imread(file_name1)
	plt.imshow(im1[:,:,ch1])
	plt.show()
	imgray1 = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
	im2= cv2.imread(file_name2)
	if folder3!=None:
		file_name3=os.path.join(folder3,image)
		im3= cv2.imread(file_name3)

	plt.imshow(im2[:,:,ch2])
	plt.title("This image should contain the lysosome")
	plt.show()
	imgray2 = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)

	# Apply threshold to the lysosome image. Removes background noise - comment the following 2 lines if not needed.
	UI_threshold=float(raw_input("Threshold for the lysosome image (can be decided by looking at the images before running this code)."))
	im2=im2*(im2>=UI_threshold).astype(np.int)

	# dilating the intensity
	kernel=np.ones((6,6))
	kernel[2:3,2:3]=1 #0
	#dilation

	imgray2_dilated = cv2.dilate(imgray2, kernel, iterations=2)

	imgray1_smoothed=cv2.blur(imgray1,(8,8)) # Recently added

	imgray=imgray1_smoothed+imgray2

	# 0:7 items to be appended
	curr_image.append(file_name1)
	curr_image.append(file_name2)
	curr_image.append(im1)
	curr_image.append(im2)
	curr_image.append(imgray1)
	curr_image.append(imgray2)
	curr_image.append(imgray)

	# Erode the given image
	selem_create=np.ones((5,5))
	selem_create[2,2]=1 #0
	skimage_erosion1=morphology.erosion(imgray, selem=selem_create, out=None, shift_x=False, shift_y=False)

	#8
	curr_image.append(skimage_erosion1)

	# Otsu Thresholding and Binarization
	ret3,th3 = cv2.threshold(skimage_erosion1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	#9
	curr_image.append(th3)

	# ClusterAndFill
	clus_info=[]
	props,calcs,hull,clus_info=clusterAndSave(th3,clus_info)

	#10
	curr_image.append(clus_info)
	#print(curr_image)
	consolidated[str(image)]=curr_image

print("Saving simplified_summary")
np.save(os.path.join(output_path,tag+'_simplified_summary.npy'),np.array(simplified_summary))
print("Saved numpy array")
npy_file=os.path.join(output_path,tag+'_simplified_summary.npy')
# Plotting the distribution of the intensities:
SaveResults(npy_file,os.path.join(output_path,tag+'_simplified_matrix_summary'),consolidated,os.path.join(output_path,tag+'_consolidated_summary'))
