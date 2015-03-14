import cv2,numpy as np
from utils import normalize
"""
	Extracts EigenBased Texture Features
"""

def eigenBasedFeats(_input):
	assert(_input.ndim==1|_input.ndim==3)
	if(_input.ndim == 3):
		_input = cv2.cvtColor(_input,cv2.COLOR_RGB2GRAY)
	eigen = cv2.cornerEigenValsAndVecs(_input,15,3);
	eigen = eigen.reshape(_input.shape[0], _input.shape[1], 3, 2)
	texture_mag = normalize(np.sqrt(eigen[:,:,0,0]**2 +  eigen[:,:,0,1]**2))
	texture_dir1 = normalize(np.arctan2(eigen[:,:,1,1],eigen[:,:,1,0]))
	texture_dir2 = normalize(np.arctan2(eigen[:,:,2,1],eigen[:,:,2,0]))
	texture_prop  = np.dstack((texture_mag,texture_dir1,texture_dir2));
	return texture_prop;

"""
def glcmFeats(_input):
	assert(_input.ndims==1|_input.ndims==3|)
	if(_input.ndims == 3):
		_input = cv2.cvtColor(_input,cv2.COLOR_RGB2GRAY)
	def texture_prop(pos,patch_size = 2):
		_mean_min = pos-patch_size;
		_mean_max = +patch_size;
		glcm = greycomatrix(_input[_mean_min[0]:_mean_max[0],_mean_min[1]:_mean_max[1]],
					[3], [0], 256, symmetric=True, normed=True)
		_dis = greycoprops(glcm, 'dissimilarity')[0, 0];
		_cor = greycoprops(glcm, 'correlation')[0, 0];
		return (_dis,_cor);
"""
