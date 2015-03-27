import time,cv2
import numpy as np 
from utils import normalize
from saliency import Saliency
from scipy.spatial.distance import pdist,squareform

class ColorFrequency(Saliency):
	def __init__(self,properties):
		self.weights = [0.3,0.1]
		super(ColorFrequency, self).__init__(properties);
		self.method = "cf"
		
	def __performSaliency__(self,region_desc):
		start_time = time.time();
		(num_regions,regions,region_props,data) = region_desc
		frame_shape = regions.shape;
		_mean = region_props[0][:num_regions,];
		_color = data[:num_regions,].copy();
		_norm = np.sqrt(frame_shape[0]*frame_shape[0] + frame_shape[1]*frame_shape[1]);
		_allp_dist = squareform(pdist(_mean))
		prev_saliency = np.ones(num_regions,np.float)
		for dist_weight in self.weights:
			allp_dist = np.exp(-_allp_dist/(_norm*dist_weight));
			norm_dist=1/np.sum(allp_dist,1)
			avg_color = np.dot(allp_dist*norm_dist[:,None],_color)
			saliency = np.linalg.norm(_color - avg_color,axis=1)#*np.sqrt(prev_saliency)
			saliency = saliency/(np.max(saliency))
			indices = saliency<0.1; _color[indices,:] = np.zeros(data.shape[1]);
			indices = saliency>0.9; _color[indices,:] = np.zeros(data.shape[1]);
			prev_saliency=saliency; prev_saliency[saliency<0.1]=0
		saliency  = sum([np.where(regions==region,saliency[region],0)
								for region in range(num_regions)],0)
		if self.props.doProfile:
			cv2.imwrite(self.PROFILE_PATH+self.method+'_p.png',np.uint8(saliency*255));
			print "Freq Tuning (preprocess) : ",time.time()-start_time	
			
		return saliency;
