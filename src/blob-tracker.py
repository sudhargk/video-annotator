import cv2
import numpy as np
import segmentation.bgSubstaction as bgsub

#A box r inside the other box q
def inside(r, q):
	rx, ry, rw, rh = r
	qx, qy, qw, qh = q
	return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

#Draw rectangle and store on img
def draw_detections(img, rects, thickness = 1,shrinkage = 0.05):
	for x, y, w, h in rects:
		# the HOG detector returns slightly larger rectangles than the real objects.
		# so we slightly shrink the rectangles to get a nicer output.
		pad_w, pad_h = int(shrinkage*w), int(shrinkage*h)
		cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)

	
def tracker(path):
	#initialization for default value
	if path=='0':
		path=0;	
		
	print "Background substraction for object tracking............"
	cap = cv2.VideoCapture(path)
	ret, prev_frame = cap.read()
	
	w = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
	fourcc = cv2.cv.CV_FOURCC(*'XVID')
	vidout = cv2.VideoWriter('out.avi',fourcc,20,(w,h))
	
	
	def cap_read():
		if cap.isOpened():
			ret, frame = cap.read()
			if ret:
				return frame
		return None
	bgsubImpl = bgsub.get_instance(bgsub.BGMethod.EIGEN_SUBSTRACTION,cap_read)
	bgsubImpl.setShape((h,w))
	
	N = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)); i=0;
	
	bgdModel = np.zeros((1,65),np.float64)
	fgdModel = np.zeros((1,65),np.float64)
	while(True):
		i=i+1;
		mask_frame =  bgsubImpl.process() 
		print 'Proceesing ... {0}%\r'.format((i*100/N))
		if bgsubImpl.isFinish():
			break
		#cv2.grabCut(bgsubImpl._cur_frame,mask_frame,None,bgdModel,fgdModel,2,cv2.GC_INIT_WITH_MASK)
		#bgmask = np.where((mask_frame==2)|(mask_frame==0),0,1).astype('uint8')
		#frame = bgsubImpl._cur_frame*bgmask[:,:,np.newaxis]
		vidout.write(mask_frame);
		
	cap.release();
	vidout.release()
	cv2.destroyAllWindows()
	
if __name__ == "__main__":
	import sys;
	tracker(sys.argv[1]);
