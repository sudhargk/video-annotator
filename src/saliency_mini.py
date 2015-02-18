import cv2,time
import numpy as np
from etc.slic_python import slic as slic

def rgb2int(color,quantize=32,MAX_COLORS=256):
		factor = MAX_COLORS/quantize;	
		return int(((color[0]/quantize)*factor + (color[1]/quantize))*factor + (color[2]/quantize))
			
def int2rgb(value,quantize=32,MAX_COLORS=256):
		factor = MAX_COLORS/quantize;	
		b = (value % factor)*quantize + quantize/2 ; value /= factor;
		g = (value % factor)*quantize + quantize/2 ; value /= factor;
		r = (value % factor)*quantize + quantize/2 
		return np.array([r,g,b],dtype=np.uint8);

def _quantize(color,quantize=16,MAX_COLORS=256):
	r = (color[0]/quantize)*quantize + quantize/2;
	g = (color[1]/quantize)*quantize + quantize/2;
	b = (color[2]/quantize)*quantize + quantize/2;
	return np.array([r,g,b],dtype=np.uint8)
	
class Saliency(object):	
	def __init__(self, shape, num_superpixels = 400,compactness = 40):
		self.PIXEL_MARGIN = 50;
		self.shape = shape
		self.num_superpixels = num_superpixels
		num_superpixels +=self.PIXEL_MARGIN
		self.compactness = compactness
		self.build_lab_dist_colors();
		self.color = np.zeros([num_superpixels],dtype=np.uint32)
		self.freq = np.zeros(num_superpixels,dtype=np.uint32)
		self.mean = np.zeros([num_superpixels,2],dtype=np.float32)
		self.p_frame = np.zeros([shape[0],shape[1],shape[2]],dtype=np.uint8);
		self.uniqueness = np.zeros(num_superpixels,dtype=np.float32)
		self.distribution = np.zeros(num_superpixels,dtype=np.float32)
	
	def build_lab_dist_colors(self,quantize=32,MAX_COLORS=256):
		factor = MAX_COLORS/quantize +1;
		rgb_colors = np.zeros([factor,factor,factor,3],dtype=np.uint8);
		lab_colors = np.zeros([factor,factor,factor,3],dtype=np.int16);
		for r_idx in range(factor):
			R = r_idx*quantize + quantize/2;
			for g_idx in range(factor):
				G = g_idx*quantize + quantize/2;
				for b_idx in range(factor):
					B = b_idx*quantize + quantize/2;
					rgb_colors[r_idx,g_idx,b_idx,:] = np.array([R,G,B]);
			lab_colors[r_idx,:,:,:] = cv2.cvtColor(rgb_colors[r_idx,:,:,:],cv2.COLOR_RGB2LAB);
		total_colors = factor*factor*factor
		
		self.lab_dist = np.zeros([total_colors,total_colors],dtype=np.float32);
		for x_idx in range(total_colors):
			for y_idx in range(x_idx+1,total_colors):
				idx_x = x_idx; idx_y = y_idx
				
				x_b = idx_x%factor;	idx_x = idx_x/factor
				x_g = idx_x%factor;	idx_x = idx_x/factor
				x_r = idx_x%factor
				
				y_b = idx_y%factor;	idx_y = idx_y/factor
				y_g = idx_y%factor;	idx_y = idx_y/factor
				y_r = idx_y%factor	
				
				diff_color = np.array(lab_colors[x_r,x_g,x_b,:]-lab_colors[y_r,y_g,y_b,:],dtype=np.float32)
				dist = np.linalg.norm(diff_color)
				self.lab_dist[x_idx,y_idx] = self.lab_dist[x_idx,y_idx]	= dist
		_min = np.min(self.lab_dist); _max=np.max(self.lab_dist)
		self.lab_dist = (self.lab_dist - _min)/(_max-_min)
	
	
	# quantization of all colors
	def quantize (self,cur_frame):
		#Quantization
		start_time = time.time();
		self.q_cur_frame  = cur_frame.reshape([self.shape[0]*self.shape[1],self.shape[2]])
		self.q_cur_frame = np.apply_along_axis(_quantize,1,self.q_cur_frame);
		self.q_cur_frame = self.q_cur_frame.reshape([self.shape[0],self.shape[1],self.shape[2]]);
		cv2.imwrite('outq.png',self.q_cur_frame);
		print "Quantization time : ",time.time()-start_time
	
	#initial segmentation using oversegmentation
	def build_region(self):
		start_time = time.time();
		lab_frame =  cv2.cvtColor(self.q_cur_frame,cv2.COLOR_RGB2LAB);
		labels = slic.slic_n(np.array(lab_frame),self.num_superpixels, self.compactness)
		print "Slic time : ",time.time()-start_time
		_num_superpixels = np.max(labels) + 1
		
		for lbl in range(_num_superpixels):
			self.mean[lbl][0] = 0; self.mean[lbl][1] = 0; self.freq[lbl] = 0
		
		_color = np.zeros([_num_superpixels,3],dtype=np.float32)
		for row in range(self.shape[0]):
			for col in range(self.shape[1]):
				self.freq[labels[row,col]] += 1
				_freq = self.freq[labels[row,col]]
				_color[labels[row,col],:] += (self.q_cur_frame[row,col,:]-_color[labels[row,col],:])/_freq;
				self.mean[labels[row,col]][0] += (row - self.mean[labels[row,col]][0])/_freq
				self.mean[labels[row,col]][1] += (col - self.mean[labels[row,col]][1])/_freq	
		
		for lbl in range(_num_superpixels):
			self.color[lbl] = rgb2int(np.array(_color[lbl,:],dtype=np.uint32));
				
		print "Build region (preprocess) : ",time.time()-start_time
		return (labels,_num_superpixels);
	
	
	#compute region contrast	
	def region_contrast(self,_num_superpixels,dist_weight = 0.125,color_weight=0.01):
		_norm = np.sqrt(self.shape[0]*self.shape[0] + self.shape[1]*self.shape[1]);
		start_time = time.time();
		#uniqueness and distribution
		"""
		allp_dist=np.exp(-pdist(self.mean)/(_norm*dist_weight));
		norm_dist=1/np.sum(allp_dist,1)
		allp_col_dist =  pdist(self.color_data)
		allp_exp_col_dist = np.exp(allp_col_dist/(np.max(allp_col_dist)*color_weight));
		norm_col_dist=1/np.sum(allp_exp_col_dist,1)
		allp_exp_col_dist = allp_exp_col_dist*norm_col_dist[:,None];
		weighted_mean = np.multiply(allp_exp_col_dist,self.mean)
		self.uniqueness =allp_col_dist*allp_dist*norm_dist[:,None];
		"""
		for x_idx in range(_num_superpixels):
			_weighted_mean = np.zeros(2,dtype=np.float32);
			_uniqueness = 0; _distribution = 0; _col_dist_sum=0; 
			_dist_sum=0; _col_dist_sum =0
			for y_idx in range(_num_superpixels):
				if (x_idx != y_idx):
					_col_dist =  self.lab_dist[self.color[x_idx],self.color[y_idx]];
					_col_exp_dist = np.exp(-_col_dist/color_weight)
					_weighted_mean += self.mean[y_idx]*_col_exp_dist
					_col_dist_sum += _col_exp_dist
			_weighted_mean = _weighted_mean/_col_dist_sum
			for y_idx in range(_num_superpixels):
				if (x_idx != y_idx):
					_col_neigh_dist=0; idx=0;
					_col_dist =  self.lab_dist[self.color[x_idx],self.color[y_idx]];
					_pos_dist = np.linalg.norm(self.mean[y_idx]-_weighted_mean)
					_pos_dist = _pos_dist*_pos_dist
					_col_exp_dist = np.exp(-_col_dist/color_weight)
					_pos_exp_dist = np.exp(-np.linalg.norm(self.mean[x_idx]-self.mean[y_idx])/(_norm * dist_weight))
			
					_dist_sum  += _pos_exp_dist;
					_uniqueness += _col_dist* _pos_exp_dist #* self.freq[y_idx]
					_distribution += _pos_dist * _col_exp_dist
			self.uniqueness[x_idx] = _uniqueness/_dist_sum
			self.distribution[x_idx] = np.sqrt(_distribution /_col_dist_sum)
		_min = np.min(self.uniqueness); _max = np.max(self.uniqueness);
		self.uniqueness = (self.uniqueness-_min)/(_max-_min)
		_min = np.min(self.distribution); _max = np.max(self.distribution);
		self.distribution =(self.distribution-_min)/(_max-_min)
		print "Compute Saliency : ",time.time()-start_time

		
	#process input
	def process(self,cur_frame):
		self.quantize(cur_frame)
		(reg_labels,num_superpixels_found) = self.build_region()
		self.region_contrast(num_superpixels_found)
		
		s_frame = np.zeros([self.shape[0],self.shape[1],self.shape[2]],dtype=np.uint8);
		u_frame = np.zeros([self.shape[0],self.shape[1]],dtype=np.uint8);
		d_frame = np.zeros([self.shape[0],self.shape[1]],dtype=np.uint8);
		self.uniqueness = self.uniqueness*self.uniqueness
		self.distribution = np.exp(-self.distribution)
		saliency = self.uniqueness*self.distribution
		_min = np.min(saliency); _max = np.max(saliency);
		saliency =(saliency-_min)/(_max-_min)
		#_,saliency = cv2.threshold(saliency,0.1,1,cv2.THRESH_BINARY)
		for row in range(self.shape[0]):
			for col in range(self.shape[1]):
				s_frame[row,col,:]= np.uint8(int2rgb(self.color[reg_labels[row,col]]))
				u_frame[row,col]= np.uint8(self.uniqueness[reg_labels[row,col]]*255)
				d_frame[row,col]= np.uint8(self.distribution[reg_labels[row,col]]*255)
				self.p_frame[row,col,2]= np.uint8(saliency[reg_labels[row,col]]* 255)
		cv2.imwrite('outs.png',s_frame);
		cv2.imwrite('outu.png',u_frame);
		cv2.imwrite('outb.png',d_frame);
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

