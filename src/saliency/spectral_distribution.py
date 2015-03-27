import time,cv2
import numpy as np
from utils import normalize
from saliency import Saliency
from sklearn.mixture import GMM as GMM

class SpectralDistribution(Saliency):
	def __init__(self,properties):
		self.components = 10;
		super(SpectralDistribution, self).__init__(properties);
		self.method = "sd"
		
	def __performSaliency__(self,region_desc):
		start_time = time.time();
		(num_regions,regions,region_props,data) = region_desc
		frame_shape = regions.shape;
		
		gmm = GMM(n_components=self.components,covariance_type='diag',random_state=1);
		gmm.fit(data); 	prob = np.transpose(gmm.predict_proba(data))
		prob_norm = 1/np.sum(prob+0.000000001,1);
		prob_mean = np.dot(prob,region_props[0])*prob_norm[:,None]
		prob_var = np.zeros([self.components,2],dtype=np.float32);
		for c_idx in range(self.components):
			prob_var[c_idx,:] = np.dot(prob[c_idx,:],np.power(region_props[0]-prob_mean[c_idx,:],2))*prob_norm[c_idx];
		spatial_var = np.sum(prob_var,1);
		inv_spatial_var  = 1- normalize(spatial_var)
		saliency =  normalize(np.dot(inv_spatial_var,prob));
		saliency  = sum([np.where(regions==region,saliency[region],0)
								for region in range(num_regions)],0)
		if self.props.doProfile:
			cv2.imwrite(self.PROFILE_PATH+self.method+'_p.png',np.uint8(saliency*255));
			print "Spatial Distribution (preprocess) : ",time.time()-start_time

		return saliency
