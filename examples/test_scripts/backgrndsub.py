import cv2
import numpy as np
from matplotlib import pyplot as plt

def optflow(path):
	if path=='0':
		path=0;
						
	cap = cv2.VideoCapture(path)
	fgbg = cv2.BackgroundSubtractorMOG()
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret:
			fgmask = fgbg.apply(frame)
			opening = cv2.morphologyEx(fgmask,cv2.MORPH_OPEN,kernel, iterations = 4)
			# Finding sure foreground area
			sure_fg = cv2.dilate(opening,None,iterations=3)
			cv2.imshow('frame',opening)
			
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		else:
			break;
	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	import sys;
	##default
	optflow(sys.argv[1]);
