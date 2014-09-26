import cv2
import numpy as np
import features.interest_points as ip


def corner(path):
	if path=='0':
		path=0;
	cap = cv2.VideoCapture(path)

	while(cap.isOpened()):
		_, frame = cap.read()
		kp = ip.extract(ip.IPMethod.SIFT,frame);
		frame = cv2.drawKeypoints(frame, kp, color=(255,0,0))
		
		cv2.imshow('morph',frame);
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()
if __name__ == "__main__":
	import sys;
	##default
	corner(sys.argv[1]);
