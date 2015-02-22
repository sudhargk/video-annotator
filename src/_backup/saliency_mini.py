import cv2,time
import numpy as np
from etc.slic_python import slic as slic
from skimage import  segmentation, color
from skimage.measure import regionprops
from scipy.spatial.distance import pdist,squareform

def normalize(_input):
	_max = np.max(_input); _min = np.min(_input)
	return np.float32((_input-_min)/(_max-_min));

"Computes the pairwise euclidean distance across each pair of vectors"
def pdist2(A,B):
	A_2 = np.outer(np.sum(np.square(A),1),np.ones(A.shape[0]))
	B_2 = np.outer(np.ones(B.shape[0]),np.sum(np.square(B),1))
	AoutB = np.dot(A,B.transpose())
	return np.sqrt(A_2 + B_2 - 2*AoutB)
		
class Saliency(object):	
	def __init__(self, shape, num_superpixels = 400,compactness = 40):
		self.PIXEL_MARGIN = 50;
		self.shape = shape
		self.num_superpixels = num_superpixels
		num_superpixels +=self.PIXEL_MARGIN
		self.compactness = compactness
	
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
	def build_region(self,useLAB = True):
		start_time = time.time();
		self.regions = segmentation.slic(self.q_frame,self.num_superpixels, self.compactness,convert2lab=useLAB,multichannel=True)
		self.num_regions = np.max(self.regions) + 1;
		self.s_frame = color.label2rgb(self.regions,self.q_frame, kind='avg')
		self.mean = np.array([region['centroid'] for region in regionprops(self.regions+1)])
		if useLAB:
			lab_frame = cv2.cvtColor(self.q_frame,cv2.COLOR_RGB2LAB);
			self.color_data = np.array([np.sum(lab_frame[np.where(self.regions==label)],0)
											for label in range(self.num_regions)])
		else:
			self.color_data = np.array([np.sum(self.q_frame[np.where(self.regions==label)],0)
											for label in range(self.num_regions)])
		self.freq = np.array([np.sum(self.num_regions==label) for label in range(self.num_regions)])
		_inv_freq = 1/(self.freq+0.0000001); self.color_data = self.color_data*_inv_freq[:,None]
		cv2.imwrite('outs.png',self.s_frame);				
		print "Build region (preprocess) : ",time.time()-start_time
		
	#compute region contrast	
	def region_contrast(self,_num_superpixels,dist_weight = 0.25,color_weight=0.2):
		start_time = time.time();
		#uniqueness and distribution
		_norm = np.sqrt(self.shape[0]*self.shape[0] + self.shape[1]*self.shape[1]);
		allp_exp_dist=np.exp(-squareform(pdist(self.mean))/(_norm*dist_weight));
		norm_dist=1/np.sum(allp_exp_dist,0)
		allp_col_dist =  squareform(pdist(self.color_data))
		allp_exp_dist = allp_exp_dist*norm_dist[:,None]
		self.uniqueness =normalize(np.sum(allp_col_dist*allp_exp_dist,0));
		#u_frame = sum([np.where(self.regions==region,256*self.uniqueness[region],0) 
		#					for region in range(self.num_regions)],0)
		#cv2.imwrite('outu.png',u_frame);
		
		allp_exp_col_dist = np.exp(allp_col_dist/(np.max(allp_col_dist)*color_weight));
		norm_col_dist=1/np.sum(allp_exp_col_dist,0)
		allp_exp_col_dist = allp_exp_col_dist*norm_col_dist[:,None];
		weighted_mean = np.dot(allp_exp_col_dist,self.mean)
		allp_mean_var = pdist2(self.mean,weighted_mean)
		self.distribution = normalize(np.sum(allp_mean_var*allp_exp_col_dist,0))
		#d_frame = sum([np.where(self.regions==region,256*self.distribution[region],0)
		#				for region in range(self.num_regions)],0)
		#cv2.imwrite('outd.png',d_frame);
		print "Region contrast : ",time.time()-start_time
	
	def saliency_cut(self,thresh=0.7,
			scale_l = np.array([[[0,0],[0.25,0],[0,0.25],[0.25,0.25],0.125,0.125]]),
			scale_u = np.array([[0.75,0.75],[1,0.75],[0.75,1],[1,1],[0.875,0.875]])):
		_,_mask = cv2.threshold(self.mask,thresh*255,1,cv2.THRESH_BINARY)
		scale_l = np.array(scale_l * self.shape[0],np.uint);
		scale_u = np.array(scale_u * self.shape[1],np.uint);
		rect = np.hstack((scale_l,scale_u));
		bgdModel = np.zeros((1,65),np.float64); fgdModel = np.zeros((1,65),np.float64)
		out_mask = np.zeros(self.shape[:2],np.bool);
		for idx in range(rect.__len__()):
			in_mask = _mask.copy();
			cv2.grabCut(self.q_frame,in_mask,tuple(rect[idx,:]),bgdModel,fgdModel,2,cv2.GC_INIT_WITH_MASK)
			out_mask = out_mask|np.where((in_mask==0)|(in_mask==2),False,True)
		self.mask = np.uint8(out_mask*255);	
		
	#process input
	def process(self,cur_frame):
		self.quantize(cur_frame)
		num_labels = self.build_region()
		self.region_contrast(num_labels)
		saliency = normalize(self.uniqueness*np.exp(-2 *self.distribution))
		self.mask = sum([np.where(self.regions==region,255*saliency[region],0)
								for region in range(self.num_regions)],0)
		self.mask = np.uint8(self.mask);	
		self.saliency_cut()
		zero = np.zeros(self.mask.shape,np.uint8);
		return np.dstack((zero,zero,self.mask))
		
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
			mask = self.saliencyMod.process(self.cur_frame);
			return cv2.addWeighted(self.cur_frame,0.4,mask,0.6,0.0);
			

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
		mask=sal.process(inp);
		inp=cv2.addWeighted(inp,0.4,mask,0.6,0.0);
		cv2.imwrite('outp.png',mask);
		cv2.imwrite('out.png',inp);
	else:
		print "Not enough or wrong parameter"		
	
	#if sys.argv.__len__()==2:
	#	over_segmentation(sys.argv[1]);
	#elif sys.argv.__len__()==3:
	#	over_segmentation(sys.argv[1],sys.argv[2]);

