import time,cv2
import numpy as np
from utils import normalize,pdist2
from saliency import Saliency
from scipy.spatial.distance import pdist,squareform

class RegionContrast(Saliency):
	def __init__(self, properties):
		self.color_weight = 0.05;
		self.dist_weight = 0.45
		super(RegionContrast, self).__init__(properties);
		self.method = "rc"
		
	def performSaliency(self):
		start_time = time.time();
		_norm = np.sqrt(self.shape[0]*self.shape[0] + self.shape[1]*self.shape[1]);
		allp_exp_dist=np.exp(-squareform(pdist(self.mean))/(_norm*self.dist_weight));
		norm_dist=1/np.sum(allp_exp_dist,0)
		allp_col_dist =  squareform(pdist(self.data))
		allp_exp_dist = allp_exp_dist*norm_dist[:,None]
		uniqueness =normalize(np.sum(allp_col_dist*allp_exp_dist,0));
				
		allp_exp_col_dist = np.exp(allp_col_dist/(np.max(allp_col_dist)*self.color_weight));
		norm_col_dist=1/np.sum(allp_exp_col_dist,0)
		allp_exp_col_dist = allp_exp_col_dist*norm_col_dist[:,None];
		weighted_mean = np.dot(allp_exp_col_dist,self.mean)
		allp_mean_var = pdist2(self.mean,weighted_mean)
		distribution = normalize(np.sum(allp_mean_var*allp_exp_col_dist,0))
		saliency = normalize(uniqueness*np.exp(-2 *distribution))
		self.saliency  = sum([np.where(self.regions==region,255*saliency[region],0)
								for region in range(self.num_regions)],0)
		self.saliency = np.uint8(self.saliency);
		if self.props.doProfile:
			u_frame = sum([np.where(self.regions==region,256*uniqueness[region],0) 
							for region in range(self.num_regions)],0)
			d_frame = sum([np.where(self.regions==region,256*distribution[region],0)
						for region in range(self.num_regions)],0)
			cv2.imwrite(self.PROFILE_PATH+self.method+'_p.png',self.saliency);
			cv2.imwrite(self.PROFILE_PATH+self.method+'_d.png',d_frame);
			cv2.imwrite(self.PROFILE_PATH+self.method+'_u.png',u_frame);
			print "Region contrast : ",time.time()-start_time
