import cv2,time,os
import numpy as np
from skimage.measure import regionprops
from skimage import  segmentation, color
from skimage.feature import greycomatrix, greycoprops
from sklearn.feature_extraction import image
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.measure import label
from scipy.spatial.distance import pdist,squareform
from features.color import LAB,GRAY
from utils import mkdirs
from scipy import ndimage
from utils import normalize

"""
	Factory method for different saliency methdds.
	Args :
		method (int) : Repressents different saliency method, 
					method = 1, Color frequency based
					method = 2, Context aware based
					method = 3, Region Contrast based
					method = 4, Spectral distribution base
		props (SaliencyProps):  Common properties on saliency method
	Returns :
		Saliency Object
	Raises:
		NotImplementedError : if method value is not valid
"""
def get_instance(method,props):
	if method == SaliencyMethods.COLOR_FREQUENCY:
		from saliency.color_frequency import ColorFrequency
		return ColorFrequency(props);
	elif method == SaliencyMethods.CONTEXT_AWARE:
		from saliency.context_aware import ContextAware
		return ContextAware(props);
	elif method == SaliencyMethods.REGION_CONTRAST:
		from saliency.region_contrast import RegionContrast
		return RegionContrast(props);
	elif method==SaliencyMethods.SPECTRAL_DISTRIBUTION:
		from saliency.spectral_distribution import SpectralDistribution
		return SpectralDistribution(props);
	else :
		raise NotImplementedError;
	
class SaliencyMethods(object):
	COLOR_FREQUENCY = 1;
	CONTEXT_AWARE = 2;
	REGION_CONTRAST = 3; 
	SPECTRAL_DISTRIBUTION =4

"""
	Saliency Properties (__init___)
	Args :
		num_superpxiels (int): number of slic pixels, default 400,
		compactness (int): slic compactness, default 40
		threshold (float): threshold used on saliency, default 0.7,
		doProfile (bool): allow profiling default False,
		useLAB (bool): use LAB features, default True,
		useColor (bool): use color features, default True,
		useTexture (bool) : use texture features, default False,
		doCUT (bool) : perform grab cut operations, default False,
"""			
class SaliencyProps(object):
	def __init__(self, num_superpixels = 400,compactness = 40, threshold = 0.8, doProfile = False,
						useLAB=True,useColor=True,useTexture=False,doCUT=True):
		self.num_superpixels = num_superpixels
		self.compactness = compactness
		self.doProfile = doProfile;
		self.useLAB = useLAB;
		self.useColor = useColor;
		self.useTexture = useTexture;
		self.threshold = threshold;
		self.doCUT = doCUT;

class Saliency(object):	
	def __init__(self,props):
		self.method = None
		self.props  = props
		if self.props.doProfile:
			self.PROFILE_PATH = os.environ.get("PROFILE_PATH") + os.sep;
			mkdirs(self.PROFILE_PATH);
			
			
	""" 
		Quantization of all colors using histogram equalization
		Args :
			do_equalize (bool) : perform color equalization
	"""
	def __quantize__ (self,cur_frame,do_equalize=True):
		start_time = time.time();
		q_frame = cur_frame.copy();
		if do_equalize:
			q_frame[:,:,0] = cv2.equalizeHist(cur_frame[:,:,0])
			q_frame[:,:,1] = cv2.equalizeHist(cur_frame[:,:,1])
			q_frame[:,:,2] = cv2.equalizeHist(cur_frame[:,:,2])
		if self.props.doProfile:
			cv2.imwrite(self.PROFILE_PATH + self.method+'_q.png',self.q_frame);
			print "Quantization time : ",time.time()-start_time
		return q_frame
			
	"""
		Extract Color properties either LAB color or RGB color
		Returns :
			Color properties
	"""
	def __extract_color__(self,frame,regions,region_props):
		num_regions = len(np.unique(regions));
		if self.props.useLAB:
			lab_frame = LAB(frame);
			color_data = np.array([np.sum(lab_frame[np.where(regions==region)],0)
											for region in range(num_regions)])
		else:
			color_data = np.array([np.sum(frame[np.where(regions==region)],0)
											for region in range(num_regions)])
		_inv_freq = 1/(region_props[1]+0.0000001); color_data = color_data*_inv_freq[:,None]
		return color_data;
		
	"""
		Extract GLCM based texture property
		Returns :
			Texture properties
	"""
	def __extract_texture__(self,frame,regions,region_props,useGLCM=False,useEigen=True):
		num_regions = len(np.unique(regions));
		gray = GRAY(frame)
		texture_data_glcm = None; texture_data_eig = None
		if useGLCM:
			def texture_prop(region,patch_size = 2):
				_mean_min = region_props[0][region]-patch_size;
				_mean_max = region_props[0][region]+patch_size;
				glcm = greycomatrix(gray_frame[_mean_min[0]:_mean_max[0],_mean_min[1]:_mean_max[1]],
							[3], [0], 256, symmetric=True, normed=True)
				_dis = greycoprops(glcm, 'dissimilarity')[0, 0];
				_cor = greycoprops(glcm, 'correlation')[0, 0];
				return (_dis,_cor);
			texture_data_glcm = np.array([texture_prop(region) for region in range(num_regions)])
		
		if useEigen:			
			eigen = cv2.cornerEigenValsAndVecs(gray,15,3);
			eigen = eigen.reshape(gray.shape[0], gray.shape[1], 3, 2)
			texture_mag = normalize(np.sqrt(eigen[:,:,0,0]**2 +  eigen[:,:,0,1]**2))
			texture_dir1 = normalize(np.arctan2(eigen[:,:,1,1],eigen[:,:,1,0]))
			texture_dir2 = normalize(np.arctan2(eigen[:,:,2,1],eigen[:,:,2,0]))
			texture_prop  = np.dstack((texture_mag,texture_dir1,texture_dir1));
			texture_data_eig = np.array([np.sum(texture_prop[np.where(regions==region)],0)
												for region in range(num_regions)])
			_inv_freq = 1/(region_props[1]+0.0000001); 	
			texture_data_eig = texture_data_eig * _inv_freq[:,None]
			
		if useGLCM and useEigen:
			texture_data = np.hstack((texture_data_glcm,texture_data_eig));
		elif useGLCM:
			texture_data = texture_data_glcm
		elif useEigen:
			texture_data = texture_data_eig
		else:
			raise ArgumentError("argument useGLCM and useEigen both cannot be false");
		return texture_data
		
	"""
		Initial segmentation using oversegmentation using slic
	"""
	def __build_region__(self,q_frame):
		start_time = time.time();
		regions = segmentation.slic(q_frame,self.props.num_superpixels, self.props.compactness,
				convert2lab=self.props.useLAB,multichannel=True)
		num_regions = len(np.unique(regions));
		s_frame = color.label2rgb(regions,q_frame, kind='avg')
		mean = np.array([region['centroid'] for region in regionprops(regions+1)])
		freq = np.array([np.sum(regions==region) for region in range(num_regions)])
		region_props = (mean,freq);
		
		if self.props.useColor:
			color_data = self.__extract_color__(q_frame,regions,region_props);		
		if self.props.useTexture:
			texture_data = self.__extract_texture__(q_frame,regions,region_props);
			
		if self.props.useTexture and self.props.useColor:
			data = np.hstack((color_data,texture_data))
		elif self.props.useTexture:
			data = texture_data
		else :
			data = color_data
				
		if self.props.doProfile:
			cv2.imwrite(self.PROFILE_PATH+self.method+'_s.png',s_frame);					
			print "Build region (preprocess) : ",time.time()-start_time
	
		return (num_regions,regions,region_props,data);
	"""
		Performs saliency cut using the grab cut algorithm, if doCUT set false then it just applies threshold
	"""
	
	def saliency_cut(self,frame,saliency,max_iters = 1):
		_,mask = cv2.threshold(saliency,self.props.threshold,1,cv2.THRESH_BINARY);
		return mask
			
	
	def __performSaliency__(self):
		raise NotImplementedError;
		
	"""
		Process the given input frame
	"""
	def process(self,cur_frame):
		shape = cur_frame.shape;
		q_frame = self.__quantize__(cur_frame)						# returns q_frame
		region_desc = self.__build_region__(q_frame)				# returns color, mean, freq
		saliency = self.__performSaliency__(region_desc);		# returns saliency
		return np.float32(saliency);
	
