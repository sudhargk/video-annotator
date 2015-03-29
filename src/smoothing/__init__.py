import numpy as np
from cv2 import resize
from utils import normalize


class SmoothMethods(object):
	EIGEN_BASED = 1
	GMM_BASED = 2
	SSL_BASED = 3
	

"""
	Perfrom tempral smoothing of mask obtained on each frame
	Args:
		_feats  (function) : feature extraction function
		numVectors (int) : number of eigen direction, must be less than number of features dimension
		method (int) : method = 1, then eigen based method
					   method = 2 then gmm based method , 
					   method = 3 then ssl based method , 
					   ellse Raise exception
"""
def get_instance(_feats,method):
	if method == SmoothMethods.EIGEN_BASED:
		from smoothing.eigen_based import EigenBased as Smoothner
		return Smoothner(_feats);
	elif method == SmoothMethods.GMM_BASED:
		from smoothing.gmm_based import GMMBased as Smoothner
		return Smoothner(_feats);
	elif method == SmoothMethods.SSL_BASED:
		from smoothing.ssl_based import SSLBased as Smoothner
		return Smoothner(_feats);
	else:
		raise NotImplementedError;
		
class Smoothing(object):	
	def __init__(self,_feats):
		self.feats = _feats
	"""
		Performs smoothing on list of frames,
		Args:
			blocks (list) : list of frames
			mask (list) : list of frame masks
			smoothFrames (list) : list of frame indexes that needs to be smoothened
		Returns:
			new mask for all smoothFrames.
	"""	
	def process(self,blocks,masks,smoothFrames=None):
		raise NotImplementedError;
