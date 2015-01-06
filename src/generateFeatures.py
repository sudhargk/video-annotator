import cv2,errno,os
import numpy as np
from os import listdir,makedirs
from os.path import isfile,join,isdir
import segmentation.bgSubstaction as bgsub
import features.interest_points as ip
from sklearn.cluster import MeanShift,DBSCAN, estimate_bandwidth
def createHeader(filepath,num_features):
	_file = open(filepath,'a+b');
	_file.write(str(num_features));
	_file.write(os.linesep)
	
def write_to_file (frames,filepath,perms,min_frames=10):
	_file = open(filepath,'a+b');o
	num_frames = frames.__len__();
	indices = np.arange(num_frames);
	for index in np.arange(perms):
		perm = np.random.permutation(indices)
		perm_frames=[];
		for p in np.sort(perm[:min_frames]):
			perm_frames.extend(frames[p].flatten());
		for ele in perm_frames:
			_file.write('%f '%ele);
		_file.write(os.linesep)

def write(frames,out_paths,split_ratio = [9,3,3],min_frames=10):
	#training
	write_to_file(frames,out_paths[0],split_ratio[0],min_frames);
	#testing
	write_to_file(frames,out_paths[1],split_ratio[1],min_frames);
	#validation
	write_to_file(frames,out_paths[2],split_ratio[2],min_frames);
	
def extractFromVideo(inPath,out_paths=['train.txt','test.txt','val.txt'],MIN_FRAMES = 10,DO_RESIZE=True,new_sz = (20,40)):
	#Default values
	MAX_FRAMES = 100
	ALPHA = 0.25
	
	print "Starting background subtraction and object tracking........... %s"%inPath
	
	cap = cv2.VideoCapture(inPath)
	ret, prev_frame = cap.read()	

	w = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
	w_crop = 80; h_crop = 160
	
	def cap_read():
		if cap.isOpened():
			ret, frame = cap.read()
			if ret:
				return frame
		return None
		
	bgsubImpl = bgsub.get_instance(bgsub.BGMethod.EIGEN_SUBSTRACTION,cap_read)
	bgsubImpl.setShape((h,w))
	
	N = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT));
	
	prv_mean = None; prev_img=None; i=0;
	frames =[];
	while(True):
		mask_frame =  bgsubImpl.process()
		mask_frame = cv2.medianBlur(mask_frame,5)
		mask_frame = cv2.medianBlur(mask_frame,3)
		print 'Proceesing ... {0}%\r'.format((i*100/N)),
		if bgsubImpl.isFinish():
			break	
					
		# labelling
		#img = np.zeros((h,w,3), np.uint8)	# Create a black image
		img = bgsubImpl.cur_frame
		
		points =  np.column_stack(np.where(mask_frame==1))
		if points.shape[0] > 0:
			mean = points.mean(axis=0)
			if not prv_mean is None:
				mean = ALPHA*mean + (1-ALPHA)*prv_mean
			(y,x)=np.int32(mean);
			if x-w_crop/2>0 and x+w_crop/2<w and y-h_crop/2>0 and y+h_crop/2<h :
				img = img[y-h_crop/2:y+h_crop/2,x-w_crop/2:x+w_crop/2];
				if DO_RESIZE:
					i=i+1;
					img = cv2.resize(img,new_sz);
					img  = np.asarray(img,dtype='float64')/256;
					frames.append(img.flatten());
			prv_mean = np.int32(mean);
			
		if i % MAX_FRAMES == 0:
			write(frames,out_paths,min_frames=MIN_FRAMES)
	cap.release();
	
	if frames.__len__() > MIN_FRAMES*2:
		write(frames,out_paths,min_frames=MIN_FRAMES)
	
	
def extract(dirPath,export_path="out/",paths=['train_%s.txt','test_%s.txt','val_%s.txt']):
	try:		
		makedirs(export_path)
		print "Created folder.... %s"%export_path
	except OSError as exc:
		if exc.errno == errno.EEXIST and isdir(export_path):
			print "Folder already exists."
		else:
			print "Error creating folder.."
	
	for _dir in listdir(dirPath):
		print "Initializing %s......."%_dir
		out_paths = [export_path+(p)%_dir for p in paths];
		for _file in out_paths:
			createHeader(_file,10*20*40*3);
		print "Processing %s......."%_dir
		for _file in listdir(join(dirPath,_dir)):
			_path = join(dirPath,_dir,_file)
			if isfile(_path):
				extractFromVideo(_path,out_paths);
	
if __name__ == "__main__":
	import sys;
	if sys.argv.__len__()==2:
		extract(sys.argv[1]);
	else:
		print "Insufficient arguments......"
	
