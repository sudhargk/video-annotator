import cv2
import numpy as np
import segmentation.bgSubstaction as bgsub
from sklearn.cluster import MeanShift,DBSCAN, estimate_bandwidth
	
def tracker(inPath,outPath='out.avi'):
		
	#Default values
	MIN_FRAMES_COUNT = 1000
	MIN_POINTS_TO_CLUSTER = 10
	MAX_CLUSTERS = 100
	SKIP_FRAMES = 0
	ALPHA = 0.35
	DO_CLUSTERING = True
	DO_CROPPING = False
	DO_LABELLING = True
	DO_BOUNDING_BOX = False

	random_colors = np.random.randint(256, size=(MAX_CLUSTERS, 3))
	
	#Clustering model
	#model = MeanShift(bandwidth=None, bin_seeding=True)
	model = DBSCAN(eps=5, min_samples=35)

	print "INPATH : ",inPath
	print "OUTPATH : ",outPath
	print "Starting background subtraction for object tracking..........."
	
	
	cap = cv2.VideoCapture(inPath)
	ret, prev_frame = cap.read()
	skip_i = 0;	
	while(cap.isOpened()):
		skip_i=skip_i+1;
		if skip_i > SKIP_FRAMES:
			break;
		ret, prev_frame = cap.read()
		if not ret:
			break;
	
	w = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
	w_crop = 80; h_crop = 160
	fourcc = cv2.cv.CV_FOURCC(*'XVID')
	if DO_CROPPING:
		vidout = cv2.VideoWriter(outPath,fourcc,20,(w_crop,h_crop))
	else:
		vidout = cv2.VideoWriter(outPath,fourcc,20,(w,h))	
	
	def cap_read():
		if cap.isOpened():
			ret, frame = cap.read()
			if ret:
				return frame
		return None
	bgsubImpl = bgsub.get_instance(bgsub.BGMethod.EIGEN_SUBSTRACTION,cap_read)
	bgsubImpl.setShape((h,w))
	
	N = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)); i=0;
	N = min(N-SKIP_FRAMES,MIN_FRAMES_COUNT)
	prv_mean = None
	bgdModel = np.zeros((1,65),np.float64)
	fgdModel = np.zeros((1,65),np.float64)
	while(True):
		i=i+1;
		mask_frame =  bgsubImpl.process()
		mask_frame = cv2.medianBlur(mask_frame,5)
		mask_frame = cv2.medianBlur(mask_frame,3)
		print 'Proceesing ... {0}%\r'.format((i*100/N)),
		if bgsubImpl.isFinish() or i>MIN_FRAMES_COUNT:
			break	

		#clustering
		if DO_CLUSTERING :
			points = np.where(mask_frame==1)
			points = np.column_stack(points)
			points_len = points.shape[0]
			if points_len > MIN_POINTS_TO_CLUSTER:
				model.fit(points);
				for idx,lbl in zip(range(points_len),model.labels_):
					mask_frame[points[idx][0]][points[idx][1]]=lbl+1;
					
		# labelling
		#img = np.zeros((h,w,3), np.uint8)	# Create a black image
		img = bgsubImpl.cur_frame
		#_mask_frame = mask_frame
		#cv2.grabCut(img,_mask_frame,None,bgdModel,fgdModel,2,cv2.GC_INIT_WITH_MASK)
		#bgmask = np.where((_mask_frame==2)|(_mask_frame==0),0,1).astype('uint8')
		#img = img*bgmask[:,:,np.newaxis]
		if DO_LABELLING:
			for lbl,val in enumerate(np.unique(mask_frame)):
				if lbl == 0:
					continue;
				if lbl >= MAX_CLUSTERS:
					break;
				color = tuple(random_colors[lbl])
				for point_x,point_y in np.column_stack(np.where(mask_frame==val)):
					cv2.circle(img,(point_y,point_x), 2,color, 1)
		
		if DO_BOUNDING_BOX:
			points =  np.column_stack(np.where(mask_frame==1))
			if points.shape[0] > 0:
				mean = points.mean(axis=0)
				if not prv_mean is None:
					mean = ALPHA*mean + (1-ALPHA)*prv_mean
				(y,x)=np.int32(mean);
				if x-w_crop/2>0 and x+w_crop/2<w and y-h_crop/2>0 and y+h_crop/2<h :
					if DO_CROPPING:
						img = img[y-h_crop/2:y+h_crop/2,x-w_crop/2:x+w_crop/2];
					else:
						cv2.rectangle(img,(x-w_crop/2,y-h_crop/2),(x+w_crop/2,y+h_crop/2),(255,0,0),2);
						cv2.putText(img,"waving",(x+w_crop/2,y+h_crop/2),cv2.FONT_HERSHEY_SIMPLEX, 1, 190);
					vidout.write(img);
				prv_mean = np.int32(mean);
		else:
			vidout.write(img);
		
		
	cap.release();
	vidout.release()
	cv2.destroyAllWindows()
	
if __name__ == "__main__":
	import sys;
	if sys.argv.__len__()==2:
		tracker(sys.argv[1]);
	elif sys.argv.__len__()==3:
		tracker(sys.argv[1],sys.argv[2]);
	
