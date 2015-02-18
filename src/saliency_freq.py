import cv2,time
import numpy as np
from etc.slic_python import slic as slic
from skimage.measure import regionprops
from skimage import  segmentation, color
from scipy.spatial.distance import pdist,squareform
	
class Saliency(object):	
	def __init__(self, shape, num_superpixels = 400,compactness = 40,components = 20):
		self.PIXEL_MARGIN = 50;
		self.shape = shape
		self.num_superpixels = num_superpixels
		num_superpixels +=self.PIXEL_MARGIN
		self.compactness = compactness
		self.components = components;
		self.p_frame = np.zeros([shape[0],shape[1],shape[2]],dtype=np.uint8);
	
	# quantization of all colors
	def quantize (self,cur_frame,do_equalize=True):
		#Quantization
		start_time = time.time();
		self.q_cur_frame = cur_frame.copy();
		if do_equalize:
			self.q_cur_frame[:,:,0] = cv2.equalizeHist(cur_frame[:,:,0])
			self.q_cur_frame[:,:,1] = cv2.equalizeHist(cur_frame[:,:,1])
			self.q_cur_frame[:,:,2] = cv2.equalizeHist(cur_frame[:,:,2])
			cv2.imwrite('outq.png',self.q_cur_frame);
		print "Quantization time : ",time.time()-start_time
	
	#initial segmentation using oversegmentation
	def build_region(self):
		start_time = time.time();
		labels = segmentation.slic(self.q_cur_frame,self.num_superpixels, self.compactness,convert2lab=True,multichannel=True)
		_num_superpixels = np.max(labels) + 1;
		self.s_frame = color.label2rgb(labels,self.q_cur_frame, kind='avg')
		self.mean = np.array([region['centroid'] for region in regionprops(labels+1)])
		self.color_data = np.array([np.sum(self.q_cur_frame[np.where(labels==label)],0) for label in range(_num_superpixels)])
		self.freq = np.array([np.sum(labels==label) for label in range(_num_superpixels)])
		_inv_freq = 1/(self.freq+0.0000001); self.color_data = self.color_data*_inv_freq[:,None]
		cv2.imwrite('outs.png',self.s_frame);				
		print "Build region (preprocess) : ",time.time()-start_time
		return (labels,_num_superpixels);
	
	def freq_tuning(self,num_superpixels,weights=[0.5,0.4,0.3,0.2,0.1,0.05,0.03,0.01]):
		start_time = time.time();
		_mean = self.mean[:num_superpixels,];
		_color = self.color_data[:num_superpixels,].copy();
		_norm = np.sqrt(self.shape[0]*self.shape[0] + self.shape[1]*self.shape[1]);
		_allp_dist = squareform(pdist(_mean))
		prev_saliency = np.ones(num_superpixels,dtype=np.float)
		for dist_weight in weights[:4]:
			allp_dist = np.exp(-_allp_dist/(_norm*dist_weight));
			norm_dist=1/np.sum(allp_dist,1)
			avg_color = np.dot(allp_dist*norm_dist[:,None],_color)
			saliency = np.linalg.norm(_color - avg_color,axis=1)*np.sqrt(prev_saliency)
			saliency = saliency/(np.max(saliency))
			indices = saliency<0.1; _color[indices,:] = np.zeros(3);
			prev_saliency=saliency; prev_saliency[saliency<0.1]=0
		"""
		weights.reverse();
		for dist_weight in weights:
			allp_dist = np.exp(-_allp_dist/(_norm*dist_weight));
			norm_dist=1/np.sum(allp_dist,1)
			avg_color = np.dot(allp_dist*norm_dist[:,None],_color)
			saliency = np.linalg.norm(_color - avg_color,axis=1)
			saliency = saliency/np.max(saliency)
			indices = saliency>0.8; _color[indices,:] = np.zeros(3);
		"""
		self.weighted_mean = avg_color		
		self.saliency = saliency
		print "Freq Tuning (preprocess) : ",time.time()-start_time
		
	#process input
	def process(self,cur_frame):
		self.quantize(cur_frame)
		(reg_labels,num_superpixels_found) = self.build_region()
		self.freq_tuning(num_superpixels_found)
		s_frame = np.zeros([self.shape[0],self.shape[1],self.shape[2]],dtype=np.uint8);
		av_frame = np.zeros([self.shape[0],self.shape[1],self.shape[2]],dtype=np.uint8);
		#_,saliency = cv2.threshold(saliency,0.1,1,cv2.THRESH_BINARY)
		for row in range(self.shape[0]):
			for col in range(self.shape[1]):
				s_frame[row,col,:]= np.uint8(self.color_data[reg_labels[row,col],:])
				av_frame[row,col,:]= np.uint8(self.weighted_mean[reg_labels[row,col],:])
				self.p_frame[row,col,2]= np.uint8(self.saliency[reg_labels[row,col]]* 255)
		#cv2.imwrite('outs.png',s_frame);
		#cv2.imwrite('outav.png',av_frame);
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

