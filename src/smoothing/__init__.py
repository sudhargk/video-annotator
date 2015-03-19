import numpy as np
from cv2 import resize
from utils import normalize
SMOOTHING_EIGEN_BASED = 0
SMOOTHING_GMM_BASED = 1
SMOOTHING_SSL_BASED = 2
"""
	Perfrom tempral smoothing of mask obtained on each frame
	Args:
		_feats  (function) : feature extraction function
		numVectors (int) : number of eigen direction, must be less than number of features dimension
		method (int) : method = 0, then eigen based method
					   method = 1 then gmm based method , 
					   ellse Raise exception
"""
def get_instance(_feats,method):
	if method == SMOOTHING_EIGEN_BASED:
		from smoothing.eigen_based import EigenBased as Smoothner
		return Smoothner(_feats);
	elif method == SMOOTHING_GMM_BASED:
		from smoothing.gmm_based import GMMBased as Smoothner
		return Smoothner(_feats);
	elif method == SMOOTHING_SSL_BASED:
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
