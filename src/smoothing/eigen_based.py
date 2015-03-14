from  smoothing import Smoothing
import numpy as np
from utils import normalize

class EigenBased(Smoothing):	
	def __init__(self,_feats, numVectors = 1,threshold=0.8):
		super(EigenBased,self,).__init__(_feats)
		self.numVectors =  numVectors
		self.threshold = threshold
	
	"""g
		Computes eigen vetors based on the mask frams
	"""	
	def __computeEigenVec__(self,mask_feats):
		mean = np.mean(mask_feats,axis=0)
		mask_feats = mask_feats-mean;	#mean subtraction
		eigv,eigt = np.linalg.eig(np.cov(mask_feats,rowvar=0));
		eigt = eigt[:,np.argsort(eigv)[::-1]]
		return (eigt[:self.numVectors,:],mean)
	
	"""
		Computes the new mask for give frame, based on threshold
	"""
	def __computeNewMask__(self,frame_feat,shape,eigt,mean):
		print frame_feat.shape
		assert(shape[0]*shape[1] == frame_feat.__len__())
		frames_confidence = normalize(np.sum(np.dot(frame_feat-mean,eigt.transpose()),1))
		frames_mask =  (frames_confidence>self.threshold).reshape((shape[0],shape[1]))
		return frames_mask;
	
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
		numBlocks = blocks.__len__();
		assert(numBlocks>0)
		shape = blocks[0].shape;	frameSize = np.prod(shape[:2])
		blockFeats = np.vstack([self.feats(block) for block in blocks])
		blockMask = np.hstack([mask.flatten() for mask in masks])
		print blockMask.shape,np.sum(blockMask==1)
		(eigt,mean) = self.__computeEigenVec__(blockFeats[blockMask==1,:])
		if smoothFrames is None:
			smoothFrames = range(numBlocks);
		else:
			smoothFrames = [idx for idx in smoothFrames if idx < numBlocks];
		newMasks = [self.__computeNewMask__(blockFeats[frameIdx*frameSize:(frameIdx+1)*frameSize],shape,eigt,mean) 
							for frameIdx in smoothFrames]
		return newMasks
		
		

	
