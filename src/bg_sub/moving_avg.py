import cv2
import numpy as np
from bg_sub import BGSubtractionImpl

"""
	A background subtraction by  moving average
	Args:
		alpha (float) : values [0-1] specifies the importance to current frame, default 0.3
		threshold (int) : values [0-256] integer threshold on difference between foreground 
					and background, default = 10
	
"""
class MovingAvgImpl(BGSubtractionImpl):
	def __init__(self,_nextFrame,alpha=0.6,threshold=10):
		super(MovingAvgImpl,self).__init__(_nextFrame)
		self.prev_frame = None
		self.cur_frame = None
		self.alpha = alpha
		self.threshold = threshold
		
	def process(self):
		if self.prev_frame is None:
			self.prev_frame = np.float32(self._nextFrame())
		self.cur_frame = self._nextFrame()
		if self.cur_frame is None:
			self.finish = True
		else:
			self.prev_frame = np.float32(self.prev_frame)
			cv2.accumulateWeighted(np.float32(self.cur_frame),self.prev_frame,self.alpha,None)
			self.prev_frame = cv2.convertScaleAbs(self.prev_frame)
			diff = self.frame_differencing(self.prev_frame,self.cur_frame,self.threshold)
			return diff;
