import cv2
import numpy as np
from matplotlib import pyplot as plt

def segment(path):
	if path=='0':
		path=0;
	cap = cv2.VideoCapture(path)

	while(cap.isOpened()):
		_, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
		# noise removal
		kernel = np.ones((3,3),np.uint8)
		opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 4)
		
		# sure background area
		sure_bg = cv2.dilate(opening,None,iterations=3)
		ret,sure_bg = cv2.threshold(sure_bg,1,128,1)
		
		# Finding sure foreground area
		sure_fg = cv2.erode(opening,None,iterations=3)
		
		# Finding unknown region
		markers = cv2.add(sure_bg,sure_fg)
		markers = np.int32(markers)
		
		#watershed algorithm
		cv2.watershed(frame,markers)
		
		frame[markers==-1] = [0,0,255]
		
		cv2.imshow('morph',frame);
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	import sys;
	##default
	segment(sys.argv[1]);
