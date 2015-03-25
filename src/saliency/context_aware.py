import time,cv2
import numpy as np 
from utils import normalize
from saliency import Saliency
from scipy.spatial.distance import pdist,squareform

class ContextAware(Saliency):
	def __init__(self, properties):
		self.weights = [0.03,0.05]
		super(ContextAware, self).__init__(properties);
		self.method = "ca"
		
	def __performSaliency__(self,region_desc):
		start_time = time.time();
		(num_regions,regions,region_props,data) = region_desc
		frame_shape = regions.shape;
		_mean = region_props[0][:num_regions,];
		_color = data[:num_regions,].copy();
		_norm = np.sqrt(frame_shape[0]*frame_shape[0] + frame_shape[1]*frame_shape[1]);
		_allp_dist = squareform(pdist(_mean))
		saliency = np.zeros(num_regions,dtype=np.float)
		allp_col_dist =  squareform(pdist(data))
		for dist_weight in self.weights:
			allp_dist = np.exp(-_allp_dist/(_norm*dist_weight));
			norm_dist=1/np.sum(allp_dist,1)
			_saliency = np.sum(np.dot(allp_dist*norm_dist[:,None],allp_col_dist),1)
			_saliency=normalize(_saliency); saliency += _saliency*np.sqrt(dist_weight);
		saliency = 1 - np.exp(-normalize(saliency))
		saliency  = sum([np.where(regions==region,saliency[region],0)
								for region in range(num_regions)],0)
		if self.props.doProfile:
			cv2.imwrite(self.PROFILE_PATH+self.method+'_p.png',np.uint8(saliency*255));
			print "Spatial Distribution (preprocess) : ",time.time()-start_time
		
		return saliency
