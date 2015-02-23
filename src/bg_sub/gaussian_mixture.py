import cv2
import numpy as np
from bg_sub import BGSubtractionImpl

"""
	A background subtraction technique using gausian mixture technique
	Args :
		threshold (int) : values [0-256] integer threshold on difference between foreground 
					and background, default = 50
"""
class BackgroundSubtractorGMGImpl(BGSubtractionImpl):
	def __init__(self,_nextFrame,threshold=50):
		super(BackgroundSubtractorGMGImpl,self).__init__(_nextFrame)
		self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
		self.fgbg = cv2.BackgroundSubtractorMOG2()
		self.cur_frame = None
		self.prev_frame = None
		self.threshold = threshold
	
	def process(self):
		if self.prev_frame is None:
			self.prev_frame = self._nextFrame()
			fgmask = self.fgbg.apply(self.prev_frame)
			self.prev_frame = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, self.kernel)
			self.prev_frame = cv2.morphologyEx(self.prev_frame, cv2.MORPH_OPEN, self.kernel)
			self.prev_frame = cv2.cvtColor(self.prev_frame,cv2.COLOR_GRAY2RGB)
		
		self.cur_frame = self._nextFrame()
		if self.cur_frame is None:
			self.finish = True
		else:
			fgmask = self.fgbg.apply(self.cur_frame)
			cur_diff = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)
			cur_diff = cv2.cvtColor(cur_diff,cv2.COLOR_GRAY2RGB)
			diff = self.frame_differencing(self.prev_frame,cur_diff,self.threshold);
			self.prev_frame=cur_diff
			return diff;
