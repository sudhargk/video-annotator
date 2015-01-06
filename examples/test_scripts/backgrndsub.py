import cv2
import numpy as np
from matplotlib import pyplot as plt

def bgSub(path):
	if path=='0':
		path=0;
						
	cap = cv2.VideoCapture(path)
	fgbg = cv2.BackgroundSubtractorMOG()
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	N = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)); i=0;
	while(cap.isOpened()):
		i=i+1;
		ret, frame = cap.read()
		if ret:
			fgmask = fgbg.apply(frame)
			opening = cv2.morphologyEx(fgmask,cv2.MORPH_OPEN,kernel, iterations = 2)
			
			cv2.imshow('frame',fgmask)
			
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		else:
			break;
	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	import sys;
	##default
	bgSub(sys.argv[1]);
