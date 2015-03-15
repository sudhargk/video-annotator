from  smoothing import Smoothing
import numpy as np
from utils import normalize
from sklearn.mixture import GMM as GMM

class GMMBased(Smoothing):
	def __init__(self,_feats, numMixtures = 10,threshold=0.6):
		super(GMMBased,self,).__init__(_feats)
		self.numMixtures =  numMixtures
		self.threshold = threshold
		
	def __build_model__(self,fg_mask_feats,bg_mask_feats):
		fg_gmm = GMM(n_components=self.numMixtures,covariance_type='diag',random_state=1);
		fg_gmm.fit(fg_mask_feats);
		bg_gmm = GMM(n_components=self.numMixtures,covariance_type='diag',random_state=1);
		bg_gmm.fit(bg_mask_feats);
		return (fg_gmm,bg_gmm);
		
	def __get_score__ (self,frame_feats,shape,gmm):
		assert(shape[0]*shape[1] == frame_feats.__len__())
		fg_score = gmm[0].score(frame_feats)
		bg_score = gmm[1].score(frame_feats)
		print fg_score[0],bg_score[0]
		frames_mask =  (fg_score>bg_score).reshape((shape[0],shape[1]))
		return frames_mask;
		
	def process(self,blocks,fgMasks,bgMasks,smoothFrames=None):
		numBlocks = blocks.__len__();
		assert(numBlocks>0)
		shape = blocks[0].shape;	frameSize = np.prod(shape[:2])
		blockFeats = np.vstack([self.feats(block) for block in blocks])
		blockFgMask = np.hstack([mask.flatten() for mask in fgMasks])
		blockBgMask = np.hstack([mask.flatten() for mask in bgMasks])
		gmm = self.__build_model__(blockFeats[blockFgMask==1,:],blockFeats[blockBgMask==1,:]);
		if smoothFrames is None:
			smoothFrames = range(numBlocks);
		else:
			smoothFrames = [idx for idx in smoothFrames if idx < numBlocks];			
		newMasks = [self.__get_score__(blockFeats[frameIdx*frameSize:(frameIdx+1)*frameSize],shape,gmm) 
							for frameIdx in smoothFrames]
		return newMasks
	
