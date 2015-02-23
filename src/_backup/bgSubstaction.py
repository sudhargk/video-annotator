import cv2
import numpy as np
		
class BGMethod(object):
	FRAME_DIFFERENCING =1;
	MOVING_AVERAGE = 2;
	EIGEN_SUBSTRACTION=3;
	GMG_SUBSTRACTION = 4;


def get_instance(method,_nextFrame):
	if method == BGMethod.FRAME_DIFFERENCING:
		return FrameDifferencingImpl(_nextFrame);
	elif method == BGMethod.MOVING_AVERAGE:	
		return MovingAvgImpl(_nextFrame);
	elif method == BGMethod.EIGEN_SUBSTRACTION:	
		return EigenBGSubImpl(_nextFrame);
	elif method == BGMethod.GMG_SUBSTRACTION:
		return BackgroundSubtractorGMGImpl(_nextFrame);

class BGSubtractionImpl(object):
	def __init__(self,_nextFrame):
		self.finish = False
		self._nextFrame = _nextFrame
	
	def setShape(self,shape):
		self.shape = shape	
	
	def isFinish(self):
		return self.finish	
	
	def process(self):
		raise NotImplementedError
		
	def frame_differencing(self,prev_frame,cur_frame,threshold_value=100):
		diff = cv2.subtract(cur_frame,prev_frame);
		diff = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
		_,diff = cv2.threshold(diff,threshold_value,1,cv2.THRESH_BINARY)
		return diff	


class FrameDifferencingImpl(BGSubtractionImpl):
	def __init__(self,_nextFrame):
		super(FrameDifferencingImpl,self).__init__(_nextFrame)
		self.prev_frame = None
		self.cur_frame = None
	
	
	def process(self):
		if self.prev_frame is None:
			self.prev_frame = self._nextFrame()
		self.cur_frame = self._nextFrame()
		if self.cur_frame is None:
			self.finish = True
		else:
			diff = self.frame_differencing(self.prev_frame,self.cur_frame)
			self.prev_frame = self.cur_frame
			#diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
			return diff;


class BackgroundSubtractorGMGImpl(BGSubtractionImpl):
	def __init__(self,_nextFrame):
		super(BackgroundSubtractorGMGImpl,self).__init__(_nextFrame)
		self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
		self.fgbg = cv2.BackgroundSubtractorMOG2()
		self.cur_frame = None
		self.prev_frame = None
	
	def process(self):
		if self.prev_frame is None:
			self.prev_frame = self._nextFrame()
			fgmask = self.fgbg.apply(self.prev_frame)
			self.prev_frame = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)
			self.prev_frame = cv2.cvtColor(self.prev_frame,cv2.COLOR_GRAY2BGR)
		
		self.cur_frame = self._nextFrame()
		if self.cur_frame is None:
			self.finish = True
		else:
			fgmask = self.fgbg.apply(self.cur_frame)
			cur_diff = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)
			cur_diff = cv2.cvtColor(cur_diff,cv2.COLOR_GRAY2BGR)
			diff = self.frame_differencing(self.prev_frame,cur_diff);
			#diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
			self.prev_frame=cur_diff		
			return diff;
		
class MovingAvgImpl(BGSubtractionImpl):
	def __init__(self,_nextFrame,alpha=0.01):
		super(MovingAvgImpl,self).__init__(_nextFrame)
		self.prev_frame = None
		self.cur_frame = None
		self.alpha = alpha
		
	def process(self):
		if self.prev_frame is None:
			self.prev_frame = np.float32(self._nextFrame())
		self.cur_frame = self._nextFrame()
		if self.cur_frame is None:
			self.finish = True
		else:
			cv2.accumulateWeighted(np.float32(self.cur_frame),np.float32(self.prev_frame),self.alpha,None)
			self.prev_frame = cv2.convertScaleAbs(self.prev_frame)
			diff = self.frame_differencing(self.prev_frame,self.cur_frame)
			return diff;

class EigenBGSubImpl(BGSubtractionImpl):
	def __init__(self,_nextFrame,N=4,K=2):
		super(EigenBGSubImpl,self).__init__(_nextFrame)
		self.prev_frames = []
		self.shape = None
		self.cur_frame = None
		self.N = N
		self.K = K
	
	def frame_differencing(self,prev_frame,cur_frame,threshold_value=10):
		diff = cv2.absdiff(cur_frame,prev_frame);
		#diff = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
		_,diff = cv2.threshold(diff,threshold_value,1,cv2.THRESH_BINARY);
		return diff	

		
	def process(self):
		while self.prev_frames.__len__() < self.N:
			frame  = self._nextFrame()
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			self.prev_frames.append(frame.flatten())
		
		self.cur_frame = self._nextFrame()
		if self.cur_frame is None:
			self.finish = True
			return;
		else:
			self._cur_frame = cv2.cvtColor(self.cur_frame, cv2.COLOR_BGR2GRAY)
			mean = np.mean(self.prev_frames,axis=0)
			mean_subtracted = [frame - mean for frame in self.prev_frames];
			mean_subtracted = np.asarray(mean_subtracted)
			eigv,eigt = np.linalg.eig(np.cov(mean_subtracted));
			eigt = np.dot(mean_subtracted.T,eigt); 
			eigt = eigt / np.linalg.norm(eigt,axis=0)
			idx = np.argsort(eigv)[::-1]
			eigt = eigt[:,idx]; eigv = eigv[idx]
			
			
			assert(self.K<=self.N and self.shape!=None);
			score = np.dot(self._cur_frame.flatten()-mean,eigt[:,range(self.K)])		
			recon = np.dot(eigt[:,range(self.K)],score)+mean
			recon = np.uint8(recon.reshape(self.shape))
			
			diff = self.frame_differencing(recon,self._cur_frame)
			
			self.prev_frames = self.prev_frames[1:]
			self.prev_frames.append(self._cur_frame.flatten())
			return diff;

