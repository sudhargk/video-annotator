import time,numpy as np,cv2
from sklearn.mixture import GMM as GMM
from sklearn.cluster import KMeans

from  smoothing import Smoothing
from utils import normalize


RANDOM = 12

class GMMBased(Smoothing):
	def __init__(self,_feats, numMixtures = 2):
		super(GMMBased,self,).__init__(_feats)
		self.numMixtures =  numMixtures
		self.threshold = 0.7
		self.fg_gmm = None
		self.bg_gmm = None
						
	
	def __compute_mean__(self,mask_feats):
		cluster = KMeans(init='k-means++', n_clusters=self.numMixtures,max_iter = 10,
						random_state=RANDOM)
		if len(mask_feats)==0:
			return None;
		cluster.fit(mask_feats)	
		return cluster.cluster_centers_;
	
	def __fit_model__(self,model,mask_feats):
		if model is None:
			cl_means = self.__compute_mean__(mask_feats);
			if not cl_means is None:
				model= GMM(n_components=self.numMixtures,covariance_type='diag',n_iter=4,
						init_params='wc');
				model.means_ = cl_means;
			"""
			else:
				_covars = model.covars_; _weights = model.weights_; _means = model.means_
				model = GMM(n_components=self.numMixtures,covariance_type='diag',n_iter=4,
								init_params='');
				model.means_ = _means; model.covars_ = _covars;	model.weights_ = _weights;
			"""
		if len(mask_feats)>0:
			model.fit(mask_feats);
		return model;
		
	def __build_model__(self,fg_mask_feats,bg_mask_feats):
		self.fg_gmm = self.__fit_model__(self.fg_gmm,fg_mask_feats);
		self.bg_gmm = self.__fit_model__(self.bg_gmm,bg_mask_feats);
		return (self.fg_gmm,self.bg_gmm);
		
	def __get_score__ (self,block,frame_feats,shape,gmm):
		assert(shape[0]*shape[1] == frame_feats.__len__())
		if gmm[0] is not None and gmm[1] is not None:
			fg_score = gmm[0].score(frame_feats)
			bg_score = gmm[1].score(frame_feats)		
			frames_mask =  (fg_score> bg_score).reshape((shape[0],shape[1]))
		else:
			frames_mask =  np.zeros((shape[0],shape[1]),dtype=np.uint8);
		return np.uint8(frames_mask);
		
	
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
		
	
	def process(self,blocks,fgMasks,bgMasks, smoothFrames=None):
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
		newMasks = [self.__get_score__(blocks[frameIdx],
						blockFeats[frameIdx*frameSize:(frameIdx+1)*frameSize],shape,gmm) 
						for frameIdx in smoothFrames]
		
		#newMasks = self.eigenSmoothing(newMasks)
		newMasks= self.__post_process__(fgMasks,newMasks);
		return newMasks
