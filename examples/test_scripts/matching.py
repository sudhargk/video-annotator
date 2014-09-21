import cv2
import numpy as np
from matplotlib import pyplot as plt

def matching(path):
	if path=='0':
		path=0;
	cap = cv2.VideoCapture(path)
	# setup initial location of window
	methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
            
	template = cv2.imread('images/template.png');
	template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY);
	w, h = template_gray.shape[::-1]
	
	hsv_roi =  cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
	roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
	cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
	
	# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
	term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )	
	while(cap.isOpened()):
		_, frame = cap.read()
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);
		res = cv2.matchTemplate(frame_gray,template_gray,cv2.TM_SQDIFF_NORMED)
		
		_,_, min_loc, max_loc = cv2.minMaxLoc(res)
		track_window = (min_loc[0],min_loc[1],w,h);
		
		#hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		#dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
		
		
		# apply meanshift to get the new location
		#ret, track_window = cv2.CamShift(dst, track_window, term_crit)
		
		x,y,w,h = track_window
		cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)

		cv2.imshow('morph',frame);
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	import sys;
	##default
	matching(sys.argv[1]);
