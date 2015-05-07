#STD PACKAGES
import cv2,numpy as np
from sklearn.mixture import GMM as GMM
from sklearn.cluster import KMeans

#MY PACKAGES
from utils import normalize,pdist2
from tracker import Tracker

#CONSTANTS
HSV_LOW = 0; HSV_MAX = 180; RANDOM=1
class MixtureBased(Tracker):
	def __init__(self):
		self.numMixtures = 1;
		self.n_bins = 360
		self.n_iter = 5
		self.hist_gmm = None
		self.shape_kmeans = None
		
	def __computeFeats__(self,hsv_frame,window,mask):
		shape = hsv_frame.shape
		hsv_roi = hsv_frame[window[1]:window[3],window[0]:window[2],:];
		_mask = mask[window[1]:window[3],window[0]:window[2]]		
		shape_val = np.array([window[0],window[1],
					window[2]-window[0],window[3]-window[1]]);
		hist_val = cv2.calcHist([hsv_roi],[0],np.uint8(_mask),[self.n_bins],[HSV_LOW,HSV_MAX]).flatten()
		hist_val = cv2.calcHist([hsv_roi],[0],None,[self.n_bins],[HSV_LOW,HSV_MAX]).flatten()
		return np.hstack((normalize(hist_val),shape_val));
	
	def __computeFrameFeats__(self,hsv_frame,mask):
		windows = self.__detect_object__(mask);
		activeWindowFeats = [self.__computeFeats__(hsv_frame,window,mask) for window in windows]
		return activeWindowFeats;
	
	def __compute_mean__(self,mask_feats):
		cluster = KMeans(init='k-means++', n_clusters=self.numMixtures,max_iter = 10,
						random_state=RANDOM)
		cluster.fit(mask_feats)
		return cluster.cluster_centers_;
	
	def __build_model__(self,hist_feats,shape_feats):
		if self.hist_gmm is None and self.shape_kmeans is None:
			hist_means = self.__compute_mean__(hist_feats);
			self.shape_means = self.__compute_mean__(shape_feats);
			self.hist_gmm = GMM(n_components=self.numMixtures,covariance_type='diag',
									n_iter=self.n_iter,init_params='wc',random_state=RANDOM);
			self.shape_kmeans = KMeans(init='k-means++', n_clusters=self.numMixtures,max_iter = 10)
		else:
			shape_means = self.shape_kmeans.cluster_centers_
			self.shape_kmeans = KMeans(init='k-means++', n_clusters=self.numMixtures,max_iter = 10)
			hist_means = self.hist_gmm.means_; 
			hist_covars = self.hist_gmm.covars_; 
			hist_weights = self.hist_gmm.weights_;
			self.hist_gmm = GMM(n_components=self.numMixtures,covariance_type='diag',
									n_iter=self.n_iter,init_params='');
			self.hist_gmm.means_ = hist_means;	
			self.hist_gmm.covars_ = hist_covars;
			self.hist_gmm.weights_ = hist_weights;
		self.hist_gmm.fit(hist_feats);
		self.shape_kmeans.fit(shape_feats);
		
	def track_object(self,frames,masks):
		hsv_frames = [];
		for frame in frames:
			hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
			hsv_frames.extend([hsv_frame]);
		activeWindowFeats = [self.__computeFrameFeats__(hsv_frame,mask) for (hsv_frame,mask) in zip(hsv_frames,masks)]
		activeWindowFeats = [np.vstack(activeWindowFeat) for activeWindowFeat in activeWindowFeats if len(activeWindowFeat) > 0];
		if len(activeWindowFeats) > 0:
			activeWindowFeats = np.vstack(activeWindowFeats);
			self.__build_model__(activeWindowFeats[:,:self.n_bins],activeWindowFeats[:,self.n_bins:]);
		term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 2, 1 );
		frame_track_windows = []
		for hsv_frame in hsv_frames:
			track_windows=[];
			for idx in range(self.numMixtures):
				if not self.hist_gmm is None :
					back_proj = cv2.calcBackProject([hsv_frame],[0],self.hist_gmm.means_[idx,:],[0,self.n_bins],1);
				else:
					back_proj = None;
					
				if not self.shape_kmeans is None:
					window = np.array(self.shape_kmeans.cluster_centers_[idx,:],dtype =int)
				else:
					shape = hsv_frame.shape[:2]
					window = np.array([shape[1]/4,shape[0]/4,shape[1]/2,shape[0]/2])
					
				window = tuple(window.clip(0))
				if(back_proj != None and window[2]> 10 and window[3]> 10):
					ret,window = cv2.meanShift(back_proj, window, term_crit)
				window = (window[0],window[1],window[0]+window[2],window[1]+window[3]);
				track_windows.extend([(window,idx)]);
			frame_track_windows.extend([track_windows])
		return frame_track_windows;
