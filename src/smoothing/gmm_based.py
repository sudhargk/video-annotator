import time
from  smoothing import Smoothing
import numpy as np,cv2
from utils import normalize
from sklearn.mixture import GMM as GMM

class GMMBased(Smoothing):
	def __init__(self,_feats, numMixtures = 2):
		super(GMMBased,self,).__init__(_feats)
		self.numMixtures =  numMixtures
		
	def __build_model__(self,fg_mask_feats,bg_mask_feats):
		fg_gmm = GMM(n_components=self.numMixtures,covariance_type='diag',n_iter=4,random_state=1);
		fg_gmm.fit(fg_mask_feats);
		bg_gmm = GMM(n_components=self.numMixtures,covariance_type='diag',n_iter=4,random_state=1);
		bg_gmm.fit(bg_mask_feats);
		return (fg_gmm,bg_gmm);
		
	def __get_score__ (self,block,frame_feats,shape,gmm):
		assert(shape[0]*shape[1] == frame_feats.__len__())
		fg_score = gmm[0].score(frame_feats)
		bg_score = gmm[1].score(frame_feats)
		frames_mask =  (fg_score>bg_score).reshape((shape[0],shape[1]))
		return np.uint8(frames_mask);
		"""
		bgdModel = np.zeros((1,65),np.float64); fgdModel = np.zeros((1,65),np.float64)
		fg_score = gmm[0].score(frame_feats)
		bg_score = gmm[1].score(frame_feats)
		fg_prob =  (normalize((fg_score - bg_score))).reshape((shape[0],shape[1]))
		_,mask = cv2.threshold(np.float32(fg_prob),0.6,1,cv2.THRESH_BINARY)
		pixels = np.where(mask==1); _max = np.max(pixels,1); _min = np.min(pixels,1)
		rect = np.array([_min[1],_min[0],_max[1]-_min[1],_max[0]-_min[0]],dtype=np.float32);
		rect = tuple(rect)
		mask_cut = np.zeros(block.shape[:2],np.uint8)
		mask_cut[fg_prob >= 0.3] = 2
		mask_cut[fg_prob >= 0.5] = 3
		mask_cut[fg_prob >= 0.7] = 1
		cv2.grabCut(block,mask_cut,None,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_MASK)
		frames_mask = np.float32(np.where((mask_cut==2)|(mask_cut==0),0,1))
		return np.uint8(frames_mask);
		"""
	
	def eigenSmoothing(self,blockFgMasks,numVectors = 4):
		numBlocks = blockFgMasks.__len__();shape = blockFgMasks[0].shape
		flattenFgMask = np.vstack([mask.flatten() for mask in blockFgMasks])
		mean = np.mean(flattenFgMask,axis=0)
		mean_subtracted = flattenFgMask-mean
		eigv,eigt = np.linalg.eig(np.cov(mean_subtracted));
		eigt = np.dot(mean_subtracted.T,eigt); 
		eigt = eigt / np.linalg.norm(eigt,axis=0)
		idx = np.argsort(eigv)[::-1]
		eigt = eigt[:,idx]; eigv = eigv[idx]
		score = np.dot(mean_subtracted,eigt[:,:numVectors])	
		recon = np.dot(eigt[:,:numVectors],score.T)
		recon = recon + mean[:,None]
		return np.uint8(recon>0.8).reshape((numBlocks,shape[0],shape[1]))
		
	
	def process(self,blocks,fgMasks,bgMasks,smoothFrames=None):
		numBlocks = blocks.__len__();
		assert(numBlocks>0)
		shape = blocks[0].shape;	frameSize = np.prod(shape[:2])
		blockFeats = np.vstack([self.feats(block) for block in blocks])
		blockFgMask = np.hstack([mask.flatten() for mask in fgMasks])
		blockBgMask = np.hstack([mask.flatten() for mask in bgMasks])
		start_time = time.time();
		gmm = self.__build_model__(blockFeats[blockFgMask==1,:],blockFeats[blockBgMask==1,:]);
		print "building model : ",time.time()-start_time	
		if smoothFrames is None:
			smoothFrames = range(numBlocks);
		else:
			smoothFrames = [idx for idx in smoothFrames if idx < numBlocks];			
		newMasks = [self.__get_score__(blocks[frameIdx],
							blockFeats[frameIdx*frameSize:(frameIdx+1)*frameSize],shape,gmm) 
							for frameIdx in smoothFrames]
		#newMasks = self.eigenSmoothing(newMasks)
		return newMasks
