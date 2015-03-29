import time
from  smoothing import Smoothing
import numpy as np,cv2
from utils import normalize
from sklearn.mixture import GMM as GMM
from sklearn.cluster import KMeans

RANDOM = 12

class GMMBased(Smoothing):
	def __init__(self,_feats, numMixtures = 2):
		super(GMMBased,self,).__init__(_feats)
		self.numMixtures =  numMixtures
		self.threshold = 0.7
		self.fg_gmm = None
		self.bg_gmm = None
						
	
	def __compute_mean__(self,mask_feats):
		#start_time = time.time();
		cluster = KMeans(init='k-means++', n_clusters=self.numMixtures,max_iter = 10,
						random_state=RANDOM)
		cluster.fit(mask_feats)
		#print "building kmeans : ",time.time()-start_time	
		return cluster.cluster_centers_;
			
	def __build_model__(self,fg_mask_feats,bg_mask_feats):
		if self.fg_gmm is None and self.bg_gmm is None:
			fg_cl_means = self.__compute_mean__(fg_mask_feats);
			bg_cl_means = self.__compute_mean__(bg_mask_feats);
			self.fg_gmm = GMM(n_components=self.numMixtures,covariance_type='diag',n_iter=4,
						init_params='wc');
			self.bg_gmm = GMM(n_components=self.numMixtures,covariance_type='diag',n_iter=4,
						init_params='wc',random_state=RANDOM);
			self.fg_gmm.means_ = fg_cl_means;	self.bg_gmm.means_ = bg_cl_means;
		else:
			fg_covars = self.fg_gmm.covars_; bg_covars = self.bg_gmm.covars_;
			fg_weights = self.fg_gmm.weights_; bg_weights = self.bg_gmm.weights_;
			fg_means = self.fg_gmm.means_; bg_means = self.bg_gmm.means_;
			self.fg_gmm = GMM(n_components=self.numMixtures,covariance_type='diag',n_iter=4,
						init_params='');
			self.bg_gmm = GMM(n_components=self.numMixtures,covariance_type='diag',n_iter=4,
						init_params='');
			self.fg_gmm.means_ = fg_means;		self.bg_gmm.means_ = bg_means;
			self.fg_gmm.covars_ = fg_covars;	self.bg_gmm.covars_ = bg_covars;
			self.fg_gmm.weights_ = fg_weights;	self.bg_gmm.weights_ = bg_weights;
			
		self.fg_gmm.fit(fg_mask_feats);
		self.bg_gmm.fit(bg_mask_feats);
		return (self.fg_gmm,self.bg_gmm);
		
	def __get_score__ (self,block,frame_feats,shape,gmm):
		assert(shape[0]*shape[1] == frame_feats.__len__())
		fg_score = gmm[0].score(frame_feats)
		bg_score = gmm[1].score(frame_feats)
		frames_mask =  ( fg_score> bg_score).reshape((shape[0],shape[1]))
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
		return newMasks
