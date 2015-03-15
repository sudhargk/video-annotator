from  smoothing import Smoothing
import numpy as np
from utils import normalize

class EigenBased(Smoothing):	
	def __init__(self,_feats, numVectors = 4,threshold=0.8):
		super(EigenBased,self,).__init__(_feats)
		self.numVectors =  numVectors
		self.threshold = threshold
	
	"""
		Computes eigen vetors based on the mask frams
	"""	
	def __computeEigenVec__(self,fg_mask_feats,bg_mask_feats):
		#foreground model
		fg_mean = np.mean(fg_mask_feats,axis=0)
		fg_mask_feats = fg_mask_feats-fg_mean;	#mean subtraction
		eigv,eigt = np.linalg.eig(np.cov(fg_mask_feats,rowvar=0));
		fg_eigt = eigt[:,np.argsort(eigv)[::-1]]
		fg_model = (fg_eigt[:self.numVectors,:],fg_mean)
		
		#background model
		bg_mean = np.mean(bg_mask_feats,axis=0)
		bg_mask_feats = bg_mask_feats-bg_mean;	#mean subtraction
		eigv,eigt = np.linalg.eig(np.cov(bg_mask_feats,rowvar=0));
		bg_eigt = eigt[:,np.argsort(eigv)[::-1]]
		bg_model = (bg_eigt[:self.numVectors,:],bg_mean)
		return (fg_model,bg_model)
	
	"""
		Computes the new mask for give frame, based on threshold
	"""
	def __computeNewMask__(self,frame_feat,shape,models):
		assert(shape[0]*shape[1] == frame_feat.__len__())
		(fg_eigt,fg_mean) = models[0]
		fg_score = normalize(np.sum(np.dot(frame_feat-fg_mean,fg_eigt.transpose()),1))
		(bg_eigt,bg_mean) = models[1]
		bg_score = normalize(np.sum(np.dot(frame_feat-bg_mean,bg_eigt.transpose()),1))
		frames_mask =  (fg_score>bg_score+0.2).reshape((shape[0],shape[1]))
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
	def process(self,blocks,fgmasks,bgmasks,smoothFrames=None):
		numBlocks = blocks.__len__();
		assert(numBlocks>0)
		shape = blocks[0].shape;	frameSize = np.prod(shape[:2])
		blockFeats = np.vstack([self.feats(block) for block in blocks])
		fgBlockMask = np.hstack([mask.flatten() for mask in fgmasks])
		bgBlockMask = np.hstack([mask.flatten() for mask in bgmasks])
		model = self.__computeEigenVec__(blockFeats[fgBlockMask==1,:],blockFeats[bgBlockMask==1,:])
		if smoothFrames is None:
			smoothFrames = range(numBlocks);
		else:
			smoothFrames = [idx for idx in smoothFrames if idx < numBlocks];
		newMasks = [self.__computeNewMask__(blockFeats[frameIdx*frameSize:(frameIdx+1)*frameSize],shape,model) 
							for frameIdx in smoothFrames]
		return newMasks
		
		

	
