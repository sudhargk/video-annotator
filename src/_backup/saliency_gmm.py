import cv2,time
import numpy as np
from etc.slic_python import slic as slic
#from sklearn.mixture import DPGMM as GMM
from sklearn.mixture import GMM as GMM
from skimage.feature import greycomatrix, greycoprops
from skimage.future import graph
from skimage.measure import regionprops
from skimage import  segmentation, color
		
class Saliency(object):	
	def __init__(self, shape, num_superpixels = 200,compactness = 30,components = 20):
		self.PIXEL_MARGIN = 50;
		self.shape = shape
		self.num_superpixels = num_superpixels
		num_superpixels +=self.PIXEL_MARGIN
		self.compactness = compactness
		self.components = components;
		self.gmm = GMM(n_components=components,covariance_type='full',random_state=1);
		self.p_frame = np.zeros([shape[0],shape[1],shape[2]],dtype=np.uint8);
	
	# quantization of all colors
	def quantize (self,cur_frame,do_equalize=True):
		#Quantization
		start_time = time.time();
		self.q_frame = cur_frame.copy();
		if do_equalize:
			self.q_frame[:,:,0] = cv2.equalizeHist(cur_frame[:,:,0])
			self.q_frame[:,:,1] = cv2.equalizeHist(cur_frame[:,:,1])
			self.q_frame[:,:,2] = cv2.equalizeHist(cur_frame[:,:,2])
			cv2.imwrite('outq.png',self.q_frame);
		print "Quantization time : ",time.time()-start_time
	
	#initial segmentation using oversegmentation
	def build_region(self):
		start_time = time.time();
		labels = segmentation.slic(self.q_frame,self.num_superpixels, self.compactness,convert2lab=True,multichannel=True)
		_num_superpixels = np.max(labels) + 1;
		self.s_frame = color.label2rgb(labels,self.q_frame, kind='avg')
		self.freq = np.array([np.sum(labels==label) for label in range(_num_superpixels)])
		self.mean = np.array([region['centroid'] for region in regionprops(labels+1)],dtype=np.int16);	
		
		self.color_data = np.array([np.sum(self.q_frame[np.where(labels==label)],0) for label in range(_num_superpixels)])		
		_inv_freq = 1/(self.freq+0.0000001);  self.color_data = self.color_data*_inv_freq[:,None]
		gray_frame = cv2.cvtColor(self.q_frame,cv2.COLOR_RGB2GRAY)
		def texture_prop(label,patch_size = 5):
			_mean_min = self.mean[label]-patch_size;
			_mean_max = self.mean[label]+patch_size;
			glcm = greycomatrix(gray_frame[_mean_min[0]:_mean_max[0],_mean_min[1]:_mean_max[1]],
						[3], [0], 256, symmetric=True, normed=True)
			_dis = greycoprops(glcm, 'dissimilarity')[0, 0];
			_cor = greycoprops(glcm, 'correlation')[0, 0];
			return (_dis,_cor);
		self.texture_data = np.array([texture_prop(label) for label in range(_num_superpixels)])
		self.data = np.hstack((self.color_data,self.texture_data))
		
		cv2.imwrite('outs.png',self.s_frame);				
		print "Build region (preprocess) : ",time.time()-start_time
		return (labels,_num_superpixels);
	

	def Spatial_Distribution(self,num_superpixels):
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
		self.saliency =  np.dot(inv_spatial_var,prob);
		_min = np.min(self.saliency); _max = np.max(self.saliency);
		self.saliency =(self.saliency-_min)/(_max-_min)
		print "Spatial Distribution (preprocess) : ",time.time()-start_time
		
	#process input
	def process(self,cur_frame):
		self.quantize(cur_frame)
		(reg_labels,num_superpixels_found) = self.build_region()
		self.Spatial_Distribution(num_superpixels_found)
		#_,saliency = cv2.threshold(saliency,0.1,1,cv2.THRESH_BINARY)
		for row in range(self.shape[0]):
			for col in range(self.shape[1]):
				self.p_frame[row,col,2]= np.uint8(self.saliency[reg_labels[row,col]]* 255)
		return self.p_frame
		
class SaliencyImpl(object):
	def __init__(self,_nextFrame, shape):		
		self.finish = False
		self._nextFrame = _nextFrame
		self.cur_frame = None
		self.saliencyMod = Saliency(shape);
		
	def isFinish(self):
		return self.finish	
				
	def process(self):
		self.cur_frame = self._nextFrame()
		if self.cur_frame is None:
			self.finish = True
		else:
			p_frame = self.saliencyMod.process(self.cur_frame);
			return cv2.addWeighted(self.cur_frame,0.4,p_frame,0.6,0.0);
			

def over_segmentation(inPath,outPath='out.avi'):
	print "INPATH : ",inPath
	print "OUTPATH : ",outPath
	print "Starting saliency............"
	
	cap = cv2.VideoCapture(inPath)
	MIN_FRAMES_COUNT = 1000
	SKIP_FRAMES = 0
	ALPHA = 0.35
	
	##Skipping frames
	ret, prev_frame = cap.read()
	skip_i = 0;	
	while(cap.isOpened()):
		skip_i=skip_i+1;
		if skip_i > SKIP_FRAMES:
			break;
		ret, prev_frame = cap.read()
		if not ret:
			break;
	
	w = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
	fourcc = cv2.cv.CV_FOURCC(*'XVID')
	vidout = cv2.VideoWriter(outPath,fourcc,20,(w,h))	
	N = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT));
	N = min(N-SKIP_FRAMES,MIN_FRAMES_COUNT)
	
	#read frames
	def cap_read():
		if cap.isOpened():
			ret, frame = cap.read()
			if ret:
				return frame
		return None
	sv_impl = SaliencyImpl(cap_read,[h,w,3]);
	i=0;
	while(True):
		i=i+1;
		out_frame =  sv_impl.process()
		print 'Proceesing ... {0}%'.format((i*100/N)),
		if sv_impl.isFinish() or i>MIN_FRAMES_COUNT:
			break	
		#cv2.imwrite('frame'+str(i)+'.png',out_frame);	
		vidout.write(out_frame);
		
	cap.release();
	vidout.release()
	cv2.destroyAllWindows()
	
if __name__ == "__main__":
	import sys;
	if sys.argv.__len__()==2:
		inp = cv2.imread(sys.argv[1]);
		sal = Saliency(inp.shape);
		p_frame=sal.process(inp);
		inp=cv2.addWeighted(inp,0.4,p_frame,0.6,0.0);
		cv2.imwrite('outp.png',p_frame);
		cv2.imwrite('out.png',inp);
	else:
		print "Not enough or wrong parameter"		
	
	#if sys.argv.__len__()==2:
	#	over_segmentation(sys.argv[1]);
	#elif sys.argv.__len__()==3:
	#	over_segmentation(sys.argv[1],sys.argv[2]);

