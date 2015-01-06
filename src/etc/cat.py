import cv2
import numpy as np


def cat(inPaths,outPath='out.avi'):
	i=0;
	cap = cv2.VideoCapture(inPaths[i])
	fourcc = cv2.cv.CV_FOURCC(*'XVID')
	w = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
	vidout = cv2.VideoWriter(outPath,fourcc,20,(w,h));
	while(True):
		while(cap.isOpened()):
			ret, cur_frame = cap.read()
			if not ret:
				break;
			vidout.write(cur_frame);
		cap.release();
		i = i+1;
		if(i>=inPaths.__len__()):
			break;
		cap = cv2.VideoCapture(inPaths[i])
	vidout.release();	
			
if __name__ == "__main__":
	import sys;
	cat(sys.argv[1:]);
	
	
