import time,cv2
import numpy as np
from utils import normalize,pdist2
from saliency import Saliency
from scipy.spatial.distance import pdist,squareform

class RegionContrast(Saliency):
	def __init__(self, properties):
		self.color_weight = 0.05;
		self.dist_weight = 0.2
		super(RegionContrast, self).__init__(properties);
		self.method = "rc"
		
	def __performSaliency__(self,region_desc,useDistribution=False):
		start_time = time.time();
		(num_regions,regions,region_props,data) = region_desc
		frame_shape = regions.shape;
		
		_norm = np.sqrt(frame_shape[0]*frame_shape[0] + frame_shape[1]*frame_shape[1]);
		allp_exp_dist=np.exp(-squareform(pdist(region_props[0]))/(_norm*self.dist_weight));
		norm_dist=1/np.sum(allp_exp_dist,0)
		allp_col_dist =  squareform(pdist(data))
		allp_exp_dist = allp_exp_dist*norm_dist[:,None]
		uniqueness =normalize(np.sum(allp_col_dist*allp_exp_dist,0));
		
		if useDistribution:		
			allp_exp_col_dist = np.exp(allp_col_dist/(np.max(allp_col_dist)*self.color_weight));
			norm_col_dist=1/np.sum(allp_exp_col_dist,0)
			allp_exp_col_dist = allp_exp_col_dist*norm_col_dist[:,None];
			weighted_mean = np.dot(allp_exp_col_dist,region_props[0])
			allp_mean_var = pdist2(region_props[0],weighted_mean)
			distribution = normalize(np.sum(allp_mean_var*allp_exp_col_dist,0))
			saliency = normalize(uniqueness*np.exp(-1*distribution))
		else:
			saliency = uniqueness
		
		saliency  = sum([np.where(regions==region,saliency[region],0)
								for region in range(num_regions)],0)
		if self.props.doProfile:
			u_frame = sum([np.where(regions==region,255*uniqueness[region],0) 
							for region in range(num_regions)],0)
			cv2.imwrite(self.PROFILE_PATH+self.method+'_u.png',u_frame);
			if useDistribution:
				d_frame = sum([np.where(regions==region,255*distribution[region],0)
						for region in range(num_regions)],0)
				cv2.imwrite(self.PROFILE_PATH+self.method+'_d.png',d_frame);
			cv2.imwrite(self.PROFILE_PATH+self.method+'_p.png',np.uint8(saliency*255));
			print "Region contrast : ",time.time()-start_time
	
		return saliency;
