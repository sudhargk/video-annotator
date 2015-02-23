import cv2
import numpy as np
from bg_sub import BGSubtractionImpl
"""
	A  background subtraction based on eigen subtraction
	Args:
		N (int) : Number of previous frames, default = 4
		K (int) : Number of eigen direction neccesarily less than N, default = 2
		threshold (int) : values [0-256] integer threshold on difference between foreground 
					and background, default = 10
"""

class EigenBGSubImpl(BGSubtractionImpl):
	def __init__(self,_nextFrame,N=4,K=2,threshold=10):
		super(EigenBGSubImpl,self).__init__(_nextFrame)
		self.prev_frames = []
		self.shape = None
		self.cur_frame = None
		self.N = N
		self.K = K
		self.threshold = threshold
	
	def frame_differencing(self,prev_frame,cur_frame,threshold=10):
		diff = cv2.absdiff(cur_frame,prev_frame);
		_,diff = cv2.threshold(diff,threshold,1,cv2.THRESH_BINARY);
		return diff	
	
	def process(self):
		while self.prev_frames.__len__() < self.N:					# load N prev_frames
			frame  = self._nextFrame()
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)			# cvtColor RGB2GRAY all prev_frames
			self.prev_frames.append(frame.flatten())
		
		self.cur_frame = self._nextFrame()
		if self.cur_frame is None:
			self.finish = True
			return;
		else:
			self._cur_frame = cv2.cvtColor(self.cur_frame, cv2.COLOR_BGR2GRAY)	# cvtColor RGB2GRAY all cur_frame
			mean = np.mean(self.prev_frames,axis=0)
			mean_subtracted = [frame - mean for frame in self.prev_frames];
			mean_subtracted = np.asarray(mean_subtracted)
			eigv,eigt = np.linalg.eig(np.cov(mean_subtracted));
			eigt = np.dot(mean_subtracted.T,eigt); 
			eigt = eigt / np.linalg.norm(eigt,axis=0)
			idx = np.argsort(eigv)[::-1]
			eigt = eigt[:,idx]; eigv = eigv[idx]
			
			assert(self.K<=self.N and self.shape!=None);
			score = np.dot(self._cur_frame.flatten()-mean,eigt[:,:self.K])		
			recon = np.dot(eigt[:,:self.K],score)+mean
			recon = np.uint8(recon.reshape(self.shape))
			
			diff = self.frame_differencing(recon,self._cur_frame,self.threshold)
			self.prev_frames = self.prev_frames[1:]
			self.prev_frames.append(self._cur_frame.flatten())
			return diff;

