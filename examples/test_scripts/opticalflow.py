import cv2
import numpy as np
from matplotlib import pyplot as plt

def optflow(path):
	if path=='0':
		path=0;
		
	# Create some random colors
	color = np.random.randint(0,255,(100,3))
		
	# params
	feature_params = dict(maxCorners = 20 ,qualityLevel = 0.5, minDistance = 10,
								blockSize = 7 )
	lk_params = dict( winSize  = (3,3), maxLevel = 2,
						criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
						
	cap = cv2.VideoCapture(path)
	
	_, prevframe = cap.read()
	prev_gray = cv2.cvtColor(prevframe, cv2.COLOR_BGR2GRAY)
	kp1 = cv2.goodFeaturesToTrack(prev_gray, **feature_params)

	mask = np.zeros_like(prevframe)
	while(cap.isOpened()):
		_, curframe = cap.read()
		curr_gray = cv2.cvtColor(curframe, cv2.COLOR_BGR2GRAY)
		kp2, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, kp1, None, **lk_params)
		
		# Select good points
		if st.__len__ > 0 :
			good_old = kp1[st==1]
			good_new = kp2[st==1]
		
		 # draw the tracks
		for i,(new,old) in enumerate(zip(good_new,good_old)):
			a,b = new.ravel()
			c,d = old.ravel()
			cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
			cv2.circle(curframe,(a,b),5,color[i].tolist(),-1)
		
		curframe = cv2.add(curframe,mask)
		
		cv2.imshow('morph',curframe);
		prev_gray = curr_gray
		kp1 = good_new.reshape(-1,1,2)
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	import sys;
	##default
	optflow(sys.argv[1]);
