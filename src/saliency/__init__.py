import cv2,time,os
import numpy as np
from skimage.measure import regionprops
from skimage import  segmentation, color
from skimage.future import graph
from skimage.feature import greycomatrix, greycoprops
from sklearn.feature_extraction import image
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.measure import label
from scipy.spatial.distance import pdist,squareform
from utils import mkdirs
from scipy import ndimage

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
	def __init__(self, num_superpixels = 400,compactness = 40, threshold = 0.5, doProfile = False,
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
		Returns :
			Color properties
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
		Returns :
			Texture properties
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
		self.num_regions = np.unique(self.regions).__len__();
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
	
	def saliency_cut(self,max_iters = 1):
		start_time = time.time();
		self.saliency = np.float32(self.saliency);
		_,self.mask = cv2.threshold(self.saliency,self.props.threshold,1,cv2.THRESH_BINARY)
		"""
		labels = (self.regions+1)*self.mask
		_input = cv2.GaussianBlur(self.s_frame*self.mask[:,:,None],(5,5),2)
		for iter in range(max_iters):
			_graph = graph.rag_mean_color(_input,labels, mode='similarity')
			labels = graph.cut_normalized(labels,_graph)
			_input = color.label2rgb(labels,_input, kind='avg')
		
		#Eliminating cluster relative smaller regions size
		unique_labels = np.unique(labels)[1:]
		count_labels = np.array([np.sum(labels==lbl) for lbl in unique_labels])
		sel_labels = unique_labels[count_labels >= np.average(count_labels)]
		self.mask = sum(labels==lbl for lbl in sel_labels);
		e_labels = np.zeros(labels.shape,dtype=np.int32)
		for idx,lbl in enumerate(sel_labels):
			e_labels += (labels==lbl)*(idx+1)
		
		cv2.watershed(_input,e_labels)
		e_labels[e_labels==-1]=0;
		labels = np.unique(e_labels)[1:]
		print labels
		self.mask = sum([np.where(e_labels==lbl,1,0) for lbl in labels]);
		self.ncut = color.label2rgb(e_labels,_input, kind='avg')		
		"""
		#out_mask = np.zeros(self.shape[:2],np.bool);
		#for lbl in sel_labels:
		#	in_mask = np.uint8(labels==lbl);
		#	print np.sum(in_mask)
		#	bgdModel = np.zeros((1,65),np.float64); fgdModel = np.zeros((1,65),np.float64)
		#	cv2.grabCut(self.q_frame,in_mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
		#	new_mask = np.where((in_mask==0)|(in_mask==2),False,True)
		#	print np.sum(new_mask)
		#	out_mask = out_mask|new_mask
		#self.mask = np.uint8(out_mask)
		#gray_img  = cv2.cvtColor(self.s_frame,cv2.COLOR_RGB2GRAY)
		#np.uint8(self.s_frame)
		#markers = np.uint8(self.mask*labels)
		#print markers.shape
		#self.mask = cv2.watershed(np.uint8(self.s_frame),markers);
		
		#self.mask[self.mask==-1]=0;
		#self.cut_labels = unique_labels
		#self.ncut = self.s_frame * self.mask[:,:,None] 
		if self.props.doProfile:
			cv2.imwrite(self.PROFILE_PATH+self.method+'_cut.png',self.ncut);					
			print "Saliency cut : ",time.time()-start_time
			
	"""
	def saliency_cut(self,
			scale_l = np.array([[0,0],[0.25,0],[0,0.25],[0.25,0.25],[0.125,0.125]]),
			scale_u = np.array([[0.75,0.75],[1,0.75],[0.75,1],[1,1],[0.875,0.875]]),
			max_iter = 2):
		start_time = time.time();
		_,self.mask = cv2.threshold(self.saliency,self.props.threshold*255,1,cv2.THRESH_BINARY)
		if self.props.doCUT:
			scale_l = np.array(scale_l * self.shape[0],np.uint);
			scale_u = np.array(scale_u * self.shape[1],np.uint);
			rect = np.hstack((scale_l,scale_u));
			kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
			bgdModel = np.zeros((1,65),np.float64); fgdModel = np.zeros((1,65),np.float64)
			out_mask = np.zeros(self.shape[:2],np.bool);
			for iter in range(max_iter):
				in_mask = np.uint8(self.mask.copy());
				for idx in range(rect.__len__()):
					cv2.grabCut(self.q_frame,in_mask,tuple(rect[idx,:]),bgdModel,fgdModel,1,cv2.GC_INIT_WITH_MASK)
					out_mask = out_mask|np.where((in_mask==0)|(in_mask==2),False,True)
					self.mask = np.uint8(out_mask)
				#self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel)
				#self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, kernel)
				
		self.mask = np.uint8(self.mask);
		if self.props.doProfile:
			print "Saliency cut : ",time.time()-start_time
	"""		
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
		return self.mask;
	
