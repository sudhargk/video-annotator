import time,cv2
import numpy as np
from utils import normalize
from saliency import Saliency
from sklearn.mixture import GMM as GMM

class SpectralDistribution(Saliency):
	def __init__(self,properties):
		self.components = 50;
		self.gmm = GMM(n_components=self.components,covariance_type='full',random_state=1);
		super(SpectralDistribution, self).__init__(properties);
		self.method = "sd"
		
	def performSaliency(self):
		start_time = time.time();
		self.gmm.fit(self.data);
		prob = np.transpose(self.gmm.predict_proba(self.data))
		prob_norm = 1/np.sum(prob+0.000000001,1);
		prob_mean = np.dot(prob,self.mean)*prob_norm[:,None]
		prob_var = np.zeros([self.components,2],dtype=np.float32);
		for c_idx in range(self.components):
			prob_var[c_idx,:] = np.dot(prob[c_idx,:],np.power(self.mean-prob_mean[c_idx,:],2))*prob_norm[c_idx];
		spatial_var = np.sum(prob_var,1);
		_max = np.max(spatial_var); _min = np.min(spatial_var);
		inv_spatial_var  = (_max-spatial_var)/(_max-_min);
		saliency =  normalize(np.dot(inv_spatial_var,prob));
		self.saliency  = sum([np.where(self.regions==region,255*saliency[region],0)
								for region in range(self.num_regions)],0)
		self.saliency = np.uint8(self.saliency);
		if self.props.doProfile:
			cv2.imwrite(self.PROFILE_PATH+self.method+'_p.png',self.saliency);
			print "Spatial Distribution (preprocess) : ",time.time()-start_time
