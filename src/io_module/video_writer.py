import cv2;
import numpy as np
class VideoWriter(object):
	def __init__(self,filename,width,height,fourcc = cv2.cv.CV_FOURCC(*'XVID'),fps = 20):
		self.filename = filename;
		self.shape = (int(width),int(height))
		self.fourcc = fourcc
		self.fps = fps
		
	def build(self):
		self.vidout = cv2.VideoWriter(self.filename,self.fourcc,self.fps,self.shape)
		
	def write(self,frame):
		self.vidout.write(frame);
		
	def close(self):
		self.vidout.release();
		
