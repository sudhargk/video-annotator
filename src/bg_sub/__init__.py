from cv2 import absdiff,threshold,THRESH_BINARY
from features.color import GRAY
from utils import normalize
from numpy import float32

class BGMethods(object):
	FRAME_DIFFERENCING =1;
	EIGEN_SUBSTRACTION=2;
	MOG_SUBSTRACTION = 3;

"""
	Factory method for generating instance of background subtraction technique
	Args:
		method (int) : Supports 4 different methods
			if method == 1 then frame differencing,
			else if method = 2 then eigen subtraction
			else if method == 3 then mixture of gaussian
	Returns: 
		instance of BGSubtractionImpl
	Raises :
		Not Implemented error
"""
def get_instance(method):
	if method == BGMethods.FRAME_DIFFERENCING:
		from bg_sub.frame_difference import FrameDifferencingImpl
		return FrameDifferencingImpl();
	elif method == BGMethods.EIGEN_SUBSTRACTION:	
		from bg_sub.eigen_sub import EigenBGSubImpl
		return EigenBGSubImpl();
	elif method == BGMethods.MOG_SUBSTRACTION:
		from bg_sub.gaussian_mixture import BackgroundSubtractorMOGImpl
		return BackgroundSubtractorMOGImpl();
	else:
		raise NotImplementedError("method=" +method +" Not implemented");


"""
	Base class for background subtraction
"""
class BGSubtractionImpl(object):
	def __init__(self,threshold = 0.1):
		self.threshold = threshold;
	
	def process(self):
		raise NotImplementedError

	"""
		A simple frame differencing  method with binary threshold
		Args:
			prev_frame (np.array): Previous frame having all three channels
			next_frame (np.array): Next frame having all three channels
		Returns:
			Absolute frame difference
	"""
	def __frame_differencing__(self,prev_frame,cur_frame):
		diff = absdiff(cur_frame,prev_frame);
		if diff.ndim==3:	#color input
			diff = GRAY(diff)
		variation = normalize(float32(diff));
		return variation;
	
	"""
		Computes mask after hresholding
		Returns:
			Mask after thresholding
	"""
	def threshold_mask(self,diff,_threshold=None):
		if _threshold is None:
			_threshold = self.threshold;
		_,mask = threshold(diff,_threshold,1,THRESH_BINARY)
		return mask;
