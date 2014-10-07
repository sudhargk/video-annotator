import cv2
import numpy as np


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
	
	hog = cv2.HOGDescriptor()
	hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector())
	
	
	cap = cv2.VideoCapture(path)
	w = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
	fourcc = cv2.cv.CV_FOURCC(*'XVID')
	vidout = cv2.VideoWriter('out.avi',fourcc,20,(w,h))
	
	print "Perform human tracking"
	while(cap.isOpened()):
		ret, cur_frame = cap.read()
		if not ret:
			break;
		gray_cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
		found, w = hog.detectMultiScale(gray_cur_frame, winStride=(8,8), padding=(32,32), scale=1.05)
		found_filtered = []
		#discarding the bounding box on within other
		for ri, r in enumerate(found):
			for qi, q in enumerate(found):
				if ri != qi and inside(r, q):
					break
				else:
					found_filtered.append(r)
		
		draw_detections(cur_frame, found_filtered, 1)
		vidout.write(cur_frame);
		
	
	cap.release();
	vidout.release()
	cv2.destroyAllWindows()
	
if __name__ == "__main__":
	import sys;
	tracker(sys.argv[1]);
