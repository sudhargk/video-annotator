import time,cv2
import numpy as np 
from utils import normalize
from saliency import Saliency
from scipy.spatial.distance import pdist,squareform

class ContextAware(Saliency):
	def __init__(self, properties):
		self.weights = [0.03,0.05,0.1]
		super(ContextAware, self).__init__(properties);
		self.method = "ca"
		
	def performSaliency(self):
		start_time = time.time();
		_mean = self.mean[:self.num_regions,];
		_color = self.data[:self.num_regions,].copy();
		_norm = np.sqrt(self.shape[0]*self.shape[0] + self.shape[1]*self.shape[1]);
		_allp_dist = squareform(pdist(_mean))
		saliency = np.zeros(self.num_regions,dtype=np.float)
		allp_col_dist =  squareform(pdist(self.color_data))
		for dist_weight in self.weights:
			allp_dist = np.exp(-_allp_dist/(_norm*dist_weight));
			norm_dist=1/np.sum(allp_dist,1)
			_saliency = np.sum(np.dot(allp_dist*norm_dist[:,None],allp_col_dist),1)
			_saliency=normalize(_saliency); saliency += _saliency/np.sqrt(dist_weight);
		saliency = normalize(saliency)
		self.saliency = 1 - np.exp(-saliency)
		self.saliency  = sum([np.where(self.regions==region,255*saliency[region],0)
								for region in range(self.num_regions)],0)
		self.saliency = np.uint8(self.saliency);
		if self.props.doProfile:
			cv2.imwrite(self.PROFILE_PATH+self.method+'_p.png',self.saliency);
			print "Spatial Distribution (preprocess) : ",time.time()-start_time
