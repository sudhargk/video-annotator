import cv2,time,os
import numpy as np
from skimage.measure import regionprops
from skimage import  segmentation, color
from skimage.feature import greycomatrix, greycoprops
from scipy.spatial.distance import pdist,squareform
from utils import mkdirs

"""
	Factory method for different saliency methdds.
	Args :
		method : Repressents different saliency method, 
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
def buildSaliency(method,props):
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
			
class SaliencyProps(object):
	def __init__(self, num_superpixels = 400,compactness = 40, threshold =0.7, doProfile = True,
						useLAB=True,useColor=True,useTexture=True,doCUT=True):
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
	"""
	def quantize (self,cur_frame,do_equalize=True):
		start_time = time.time();
		self.q_frame = cur_frame.copy();
		if do_equalize:
			self.q_frame[:,:,0] = cv2.equalizeHist(cur_frame[:,:,0])
			self.q_frame[:,:,1] = cv2.equalizeHist(cur_frame[:,:,1])
			self.q_frame[:,:,2] = cv2.equalizeHist(cur_frame[:,:,2])
		if self.props.doProfile:
			cv2.imwrite(self.PROFILE_PATH + self.method+'_q.png',self.q_frame);
			print "Quantization time : ",time.time()-start_time
			
	"""
		Extract Color properties either LAB color or RGB color
	"""
	def extract_color(self):
		if self.props.useLAB:
			lab_frame = cv2.cvtColor(self.q_frame,cv2.COLOR_RGB2LAB);
			color_data = np.array([np.sum(lab_frame[np.where(self.regions==region)],0)
											for region in range(self.num_regions)])
		else:
			color_data = np.array([np.sum(self.q_frame[np.where(self.regions==region)],0)
											for region in range(self.num_regions)])
		_inv_freq = 1/(self.freq+0.0000001); color_data = color_data*_inv_freq[:,None]
		return color_data;
		
	"""
		Extract GLCM based texture property
	"""
	def extract_texture(self):
		gray_frame = cv2.cvtColor(self.q_frame,cv2.COLOR_RGB2GRAY)
		def texture_prop(region,patch_size = 2):
			_mean_min = self.mean[region]-patch_size;
			_mean_max = self.mean[region]+patch_size;
			glcm = greycomatrix(gray_frame[_mean_min[0]:_mean_max[0],_mean_min[1]:_mean_max[1]],
						[3], [0], 256, symmetric=True, normed=True)
			_dis = greycoprops(glcm, 'dissimilarity')[0, 0];
			_cor = greycoprops(glcm, 'correlation')[0, 0];
			return (_dis,_cor);
		texture_data = np.array([texture_prop(region) for region in range(self.num_regions)])
		return texture_data
		
	"""
		Initial segmentation using oversegmentation using slic
	"""
	def build_region(self):
		start_time = time.time();
		self.regions = segmentation.slic(self.q_frame,self.props.num_superpixels, self.props.compactness,
				convert2lab=self.props.useLAB,multichannel=True)
		self.num_regions = np.max(self.regions) + 1;
		self.s_frame = color.label2rgb(self.regions,self.q_frame, kind='avg')
		self.mean = np.array([region['centroid'] for region in regionprops(self.regions+1)])
		self.freq = np.array([np.sum(self.regions==region) for region in range(self.num_regions)])
		if self.props.useColor:
			self.color_data = self.extract_color();			
		if self.props.useTexture:
			self.texture_data = self.extract_texture();
			
		if self.props.useTexture and self.props.useColor:
			self.data = np.hstack((self.color_data,self.texture_data))
		elif self.props.useTexture:
			self.data = self.texture_data
		else :
			self.data = self.color_data
						
		if self.props.doProfile:
			cv2.imwrite(self.PROFILE_PATH+self.method+'_s.png',self.s_frame);					
			print "Build region (preprocess) : ",time.time()-start_time
	
	"""
		Performs saliency cut using the grab cut algorithm, if doCUT set false then it just applies threshold
	"""
	def saliency_cut(self,
			scale_l = np.array([[0,0],[0.25,0],[0,0.25],[0.25,0.25],[0.125,0.125]]),
			scale_u = np.array([[0.75,0.75],[1,0.75],[0.75,1],[1,1],[0.875,0.875]])):
		start_time = time.time();
		_,self.mask = cv2.threshold(self.saliency,self.props.threshold*255,1,cv2.THRESH_BINARY)
		if self.props.doCUT:
			scale_l = np.array(scale_l * self.shape[0],np.uint);
			scale_u = np.array(scale_u * self.shape[1],np.uint);
			rect = np.hstack((scale_l,scale_u));
			bgdModel = np.zeros((1,65),np.float64); fgdModel = np.zeros((1,65),np.float64)
			out_mask = np.zeros(self.shape[:2],np.bool);
			for idx in range(rect.__len__()):
				in_mask = np.uint8(self.mask.copy());
				cv2.grabCut(self.q_frame,in_mask,tuple(rect[idx,:]),bgdModel,fgdModel,2,cv2.GC_INIT_WITH_MASK)
				out_mask = out_mask|np.where((in_mask==0)|(in_mask==2),False,True)
				self.mask = out_mask
		self.mask = np.uint8(self.mask*255);	
		if self.props.doProfile:
			print "Saliency cut : ",time.time()-start_time
			
	def performSaliency(self):
		raise NotImplementedError;
		
	"""
		Process the given input frame
	"""
	def process(self,cur_frame):
		self.shape = cur_frame.shape;
		self.quantize(cur_frame)		# sets self.q_frame
		self.build_region()				# sets self.color,self.mean,self,freq
		self.performSaliency();			# sets self.saliency
		self.saliency_cut()				# sets self.mask
		zero = np.zeros(self.mask.shape,np.uint8);
		return np.dstack((zero,zero,self.mask))
	
