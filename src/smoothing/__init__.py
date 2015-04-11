import cv2,numpy as np
from utils import normalize
from skimage.measure import label
from skimage.morphology import remove_small_objects
from scipy.ndimage.morphology import binary_fill_holes

KERNEL = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3));

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
		
	def __post_process__(self,oldMasks,newMasks):
		newMasks = self.__morphologicalOps__(newMasks);
		newMasks = self.__remove_spurious__(oldMasks,newMasks);
		return newMasks;
		
	def __morphologicalOps__(self,masks):
		new_masks = [];
		for mask in masks:
			_mask = cv2.medianBlur(np.uint8(mask),3)
			_mask = cv2.morphologyEx(_mask, cv2.MORPH_OPEN, KERNEL)
			_mask = cv2.medianBlur(_mask,3)
			_mask = cv2.morphologyEx(_mask, cv2.MORPH_CLOSE,KERNEL)
			_mask = binary_fill_holes(_mask)
			_mask = remove_small_objects(_mask,min_size=128,connectivity=2)
			new_masks.extend([_mask])
		return np.uint8(new_masks);
		
	
	def __remove_spurious__(self,oldMasks,newMasks,delta=0):
		_newMasks= [];
		for oldMask,newMask in zip(oldMasks,newMasks):
			_newMask = np.zeros(newMask.shape);
			(lbls,num) = label(newMask,connectivity=2,neighbors=4,return_num=True,background=0)
			for lbl in range(np.max(lbls)+1):
				pixels = np.where(lbls==lbl); _max = np.max(pixels,1); 
				_min = np.min(pixels,1); area = np.prod(_max - _min);
				if  (4 * np.sum(newMask[_min[0]:_max[0],_min[1]:_max[1]]) > area) and \
					(np.sum(oldMask[_min[0]:_max[0],_min[1]:_max[1]]) > delta):
					_newMask[_min[0]:_max[0],_min[1]:_max[1]]=newMask[_min[0]:_max[0],_min[1]:_max[1]];
			_newMasks.extend([_newMask]);
		return _newMasks;
	
