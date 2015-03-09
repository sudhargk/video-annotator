import cv2,types
from utils import normalize
import numpy as np
"""
	Factory method for generating instance of background subtraction technique
	Args:
		method (int) : Supports 4 different methods
			if method == 1 then frame differencing,
			else if method == 2 then moving average,
			else if method = 3 then eigen subtraction
			else if method == 4 then mixture of gaussian
		nextFrame (function)  : A function returns the next frame of video sequence if not end of video
			otherwise return None.
	Returns: 
		instance of BGSubtractionImpl
	Raises :
		Not Implemented error
"""
def get_instance(method,_nextFrame):
	assert(isinstance(_nextFrame, types.FunctionType) or isinstance(_nextFrame, types.MethodType))
	if method == BGMethods.FRAME_DIFFERENCING:
		from bg_sub.frame_difference import FrameDifferencingImpl
		return FrameDifferencingImpl(_nextFrame);
	elif method == BGMethods.MOVING_AVERAGE:	
		from bg_sub.moving_avg import MovingAvgImpl
		return MovingAvgImpl(_nextFrame);
	elif method == BGMethods.EIGEN_SUBSTRACTION:	
		from bg_sub.eigen_sub import EigenBGSubImpl
		return EigenBGSubImpl(_nextFrame);
	elif method == BGMethods.GMG_SUBSTRACTION:
		from bg_sub.gaussian_mixture import BackgroundSubtractorGMGImpl
		return BackgroundSubtractorGMGImpl(_nextFrame);
	else:
		raise NotImplementedError("method=" +method +" Not implemented");
"""
	
"""
class BGMethods(object):
	FRAME_DIFFERENCING =1;
	MOVING_AVERAGE = 2;
	EIGEN_SUBSTRACTION=3;
	GMG_SUBSTRACTION = 4;

"""
	Base class for background subtraction
"""
class BGSubtractionImpl(object):
	def __init__(self,_nextFrame):
		self.finish = False
		self._nextFrame = _nextFrame
		
	def setShape(self,shape):
		self.shape = shape	
	"""
		Check if the end of video sequence 
		Returns :
			A bool value  True if reached end of sequence otherwise false.
	"""
	def isFinish(self):
		return self.finish	
	
	def process(self):
		raise NotImplementedError
	"""
		A simple frame differencing  method with binary threshold
		Args:
			prev_frame (np.array): Previous frame having all three channels
			next_frame (np.array): Next frame having all three channels
			threshold (float) : threshold value default 100 
		Returns:
			Binary Mask after thresholding.
	"""
	def frame_differencing(self,prev_frame,cur_frame,threshold=100):
		diff = cv2.absdiff(cur_frame,prev_frame);
		if diff.ndim==3:	#color input
			diff = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
		self.variation = normalize(np.float32(diff));
		_,diff = cv2.threshold(diff,threshold,1,cv2.THRESH_BINARY)
		return diff	
