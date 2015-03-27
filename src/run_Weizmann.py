import cv2,numpy as np;
import os,os.path
from utils import getDirectories,mkdirs
from utils.stats import Stats,comparator,getNewConfusion,updateConfusion
from saliency import SaliencyMethods,SaliencyProps
from saliency import get_instance as sal_instance


def getMask(img):
	mask = (np.where(img[:,:,0]==255,True,False)) |(np.where(img[:,:,1]==255,True,False))
	mask = mask | (np.where(img[:,:,2]==255,True,False))
	return np.uint8(mask)
		
def getSalienyMap(sal,image):
	saliency = sal.process(image);
	mask = sal.saliency_cut(image,saliency);
	return mask;
	
def compareWithGroundtruth(sal,datasetPath,imageName):
	print "Processing "+imageName +" ....";
	confusion = getNewConfusion();
	srcImgPath = os.path.join(datasetPath,imageName,"src_color",imageName+".png")
	image = cv2.imread(srcImgPath)
	mask = getSalienyMap(sal,image)
	
	#write results
	mask_image = image*mask[:,:,None];
	mkdirs(os.path.join(datasetPath,imageName,"results"));
	file_name = os.path.join(datasetPath,imageName,"results",imageName+"_"+sal.method+".png")
	cv2.imwrite(file_name,mask_image)
	
	humanSegPath = os.path.join(datasetPath,imageName,"human_seg");
	for bgImg in os.listdir(humanSegPath):
		if bgImg.endswith(".png"):
			bgMask = getMask(cv2.imread(os.path.join(humanSegPath,bgImg)))
			updateConfusion(confusion,comparator(bgMask,mask))
	print "Confusion : ",confusion;
	return confusion;	

			
def processFolder(datasetPath):
	stats = Stats(datasetPath) 
	props = SaliencyProps()
	sal = sal_instance(SaliencyMethods.COLOR_FREQUENCY,props);
	for category in getDirectories(datasetPath):
		stats.addCategories(category)
		categoryPath = os.path.join(datasetPath,category)
		for image in getDirectories(categoryPath):
			stats.update(category,image,compareWithGroundtruth(sal,categoryPath,image));
		stats.writeCategoryResult(category)
	stats.writeOverallResults()

if __name__ == "__main__":
	import sys;	
	if sys.argv.__len__()<=1:
		print "dataset path not provided........."
	else :
		processFolder(sys.argv[1]);
