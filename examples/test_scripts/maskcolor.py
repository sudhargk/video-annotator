import cv2
import numpy as np
from matplotlib import pyplot as plt

def maskcolor(path,_hsv_mask):
	if path=='0':
		path=0;
	cap = cv2.VideoCapture(path)

	while(cap.isOpened()):
		_, frame = cap.read()
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

		# Threshold the HSV image to get only blue colors
		mask = cv2.inRange(hsv,_hsv_mask[0],_hsv_mask[1]);

		# Bitwise-AND mask and original image
		_res = cv2.bitwise_and(frame,frame, mask= mask)
		_gray = cv2.cvtColor(_res,cv2.COLOR_RGB2GRAY)
		_edge = cv2.adaptiveThreshold(_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
		cnt,hchy = cv2.findContours(_edge,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		print hchy
		cv2.drawContours(frame, cnt, -1, (0,255,0), 3)
		#x,y,w,h = cv2.boundingRect(np.vstack(cnt))
		#cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)		
		#rect = cv2.minAreaRect(np.vstack(cnt))
		#box = cv2.cv.BoxPoints(rect)
		#box = np.int0(box)
		#cv2.drawContours(frame,[box],0,(0,0,255),2)
		cv2.imshow('a',frame)
		#cv2.imshow('b',_edge)
		#cv2.imshow('c',res)
		#plt.show()

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()

def hsv_mask(color):
	rgb_map = {
			"red" : [[0,100,100],[10,255,255]],
			"green" : [[50,100,100],[70,255,255]],
			"blue" : [[110,100,100],[130,255,255]],
			"yellow" : [[80,100,100],[100,255,255]]
		}
	return np.asarray(rgb_map[color]);

if __name__ == "__main__":
	import sys;
	##default
	color = "red";
	if(sys.argv.__len__>3):
		color = sys.argv[2];

	_hsv_mask = hsv_mask(color);
	maskcolor(sys.argv[1],_hsv_mask);
