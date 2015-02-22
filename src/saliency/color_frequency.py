import time,cv2
import numpy as np 
from utils import normalize
from saliency import Saliency
from scipy.spatial.distance import pdist,squareform

class ColorFrequency(Saliency):
	def __init__(self,properties):
		self.weights = [0.5]
		super(ColorFrequency, self).__init__(properties);
		self.method = "cf"
		
	def performSaliency(self):
		start_time = time.time();
		_mean = self.mean[:self.num_regions,];
		_color = self.data[:self.num_regions,].copy();
		_norm = np.sqrt(self.shape[0]*self.shape[0] + self.shape[1]*self.shape[1]);
		_allp_dist = squareform(pdist(_mean))
		prev_saliency = np.ones(self.num_regions,np.float)
		for dist_weight in self.weights:
			allp_dist = np.exp(-_allp_dist/(_norm*dist_weight));
			norm_dist=1/np.sum(allp_dist,1)
			avg_color = np.dot(allp_dist*norm_dist[:,None],_color)
			saliency = np.linalg.norm(_color - avg_color,axis=1)#*np.sqrt(prev_saliency)
			saliency = saliency/(np.max(saliency))
			indices = saliency<0.1; _color[indices,:] = np.zeros(self.data.shape[1]);
			indices = saliency>0.9; _color[indices,:] = np.zeros(self.data.shape[1]);
			prev_saliency=saliency; prev_saliency[saliency<0.1]=0
		self.saliency  = sum([np.where(self.regions==region,255*saliency[region],0)
								for region in range(self.num_regions)],0)
		self.saliency = np.uint8(self.saliency);
		if self.props.doProfile:
			cv2.imwrite(self.PROFILE_PATH+self.method+'_p.png',self.saliency);
			print "Freq Tuning (preprocess) : ",time.time()-start_time	
