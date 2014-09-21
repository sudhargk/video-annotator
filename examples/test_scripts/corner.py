import cv2
import numpy as np
from matplotlib import pyplot as plt

def corner(path):
	if path=='0':
		path=0;
	cap = cv2.VideoCapture(path)

	while(cap.isOpened()):
		_, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = np.float32(gray)
		
		# Harris corner detector scale variant
		#dst = cv2.cornerHarris(gray,3,3,0.03)
		#dst = cv2.dilate(dst,None)
		#frame[dst>0.01*dst.max()]=[0,0,255]
		
		# Some other guy scale variant
		#corners = cv2.goodFeaturesToTrack(gray,25,0.01,10);
		#for i in corners:
		#	x,y = i.ravel()
		#	cv2.circle(frame,(x,y),3,255,-1)
		
		#Non maximal suppression with fast detector
		#fast = cv2.FastFeatureDetector()
		#kp = fast.detect(frame,None)
		#frame = cv2.drawKeypoints(frame, kp, color=(255,0,0))
		
		# CeNsuRe detector with Brief
		#star = cv2.FeatureDetector_create("STAR")
		#brief = cv2.DescriptorExtractor_create("BRIEF")
		#kp = star.detect(frame,None)
		#kp, des = brief.compute(frame, kp)
		#frame = cv2.drawKeypoints(frame, kp, color=(255,0,0))
		
		#ORB FAST PLUSE BRIEF
		orb = cv2.ORB();
		kp = orb.detect(frame,None)
		kp, des = orb.compute(frame, kp)
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
