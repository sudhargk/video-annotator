import cv2,time
import numpy as np
from etc.slic_python import slic as slic

def rgb2int(color,quantize=32,MAX_COLORS=256):
		factor = MAX_COLORS/quantize;	
		return ((color[0]/quantize)*factor + (color[1]/quantize))*factor + (color[2]/quantize)
			
def int2rgb(value,quantize=32,MAX_COLORS=256):
		factor = MAX_COLORS/quantize;	
		b = (value % factor)*quantize + quantize/2 ; value /= factor;
		g = (value % factor)*quantize + quantize/2 ; value /= factor;
		r = (value % factor)*quantize + quantize/2 
		return np.array([r,g,b],dtype=np.uint8);

		
class SaliencyImpl(object):
		
	def __init__(self,_nextFrame, shape, num_superpixels = 80,compactness = 10):
		
		self.finish = False
		self._nextFrame = _nextFrame
		self.cur_frame = None
		self.shape = shape
		self.num_superpixels = num_superpixels
		self.compactness = compactness
		self.pallete = {};
		self.rgb_idx = np.zeros([shape[0],shape[1]],dtype=np.uint32)
		
		self.build_lab_dist_colors();
		self.color_map = [dict() for x in range(num_superpixels)]
		self.mean = np.zeros([num_superpixels,2],dtype=np.float32)
		self.freq = np.zeros(num_superpixels,dtype=np.uint32)
		self.col_dist = np.zeros([num_superpixels,num_superpixels],dtype=np.float32)
		self.q_cur_frame = np.zeros([shape[0],shape[1],shape[2]],dtype=np.uint8);
		self.p_frame = np.zeros([shape[0],shape[1],shape[2]],dtype=np.uint8);
		self.saliency = np.zeros(num_superpixels,dtype=np.float32)
		
	def build_lab_dist_colors(self,quantize=32,MAX_COLORS=256):
		factor = MAX_COLORS/quantize;
		rgb_colors = np.zeros([factor,factor,factor,3],dtype=np.uint8);
		lab_colors = np.zeros([factor,factor,factor,3],dtype=np.int8);
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
				
				diff_color = np.array(rgb_colors[x_r,x_g,x_b,:]-rgb_colors[y_r,y_g,y_b,:],dtype=np.float32)
				dist = np.linalg.norm(diff_color)
				self.lab_dist[x_idx,y_idx] = self.lab_dist[x_idx,y_idx]	= dist
					
	def isFinish(self):
		return self.finish	
			
	def build_color_pallete(self):
		for row in range(self.shape[0]):
			for col in range(self.shape[1]):
				self.rgb_idx[row,col] = rgb2int(self.cur_frame[row,col])
				if not self.pallete.has_key(self.rgb_idx[row,col]):
					self.pallete.__setitem__(self.rgb_idx[row,col],1)
				else:
					self.pallete[self.rgb_idx[row,col]] += 1
	
	def quantize (self,drop_threshold=0.05,MIN_COLORS=10):
		start_time = time.time();
		#build color pallete
		self.build_color_pallete();
		order  = sorted(self.pallete,key=self.pallete.get,reverse=False)
		
		total_pixels = sum(self.pallete.values());
		max_thresh = int(total_pixels*(1-drop_threshold))
		num_colors = len(order);  
		
		#picking thresholded colors
		thresh_idx = num_colors; freq = total_pixels;
		while(thresh_idx-1 >= MIN_COLORS and 
			freq - self.pallete[order[thresh_idx-1]] >=  max_thresh):
			freq -= self.pallete[order[thresh_idx-1]]; thresh_idx-=1
		self.pallete.clear()
	
		#build new colors
		_cmap = {};
		colors3d = np.zeros([thresh_idx,3],dtype=np.uint8);
		for idx in range(thresh_idx):
			_color = order[idx]
			colors3d[idx,:] = int2rgb(_color)	
			_cmap.__setitem__(_color,idx);
		for u_idx in range(thresh_idx,num_colors):
			_color = order[u_idx]
			_color3d[u_idx,:] = int2rgb(_color)[:]
			sim_idx = np.argmin([np.linalg.norm(_color3d-colors3d[n_idx,:]) 
				for n_idx in range(0,thresh_idx)])
			_cmap.__setitem__(_color,sim_idx);
				
		#recoloring
		shape = self.cur_frame.shape;
		for row in range(shape[0]):
			for col in range(shape[1]):
				self.q_cur_frame[row,col,:] = colors3d[_cmap[self.rgb_idx[row,col]]];
		print "Build time : ",time.time()-start_time
		
	#initial segmentation using oversegmentation
	def build_region(self):
		start_time = time.time();
		labels = slic.slic_n(np.array(self.q_cur_frame),self.num_superpixels-10, self.compactness)
		print "Slic time : ",time.time()-start_time
		_num_superpixels = np.max(labels) + 1
		for lbl in range(_num_superpixels):
			self.color_map[lbl].clear();
			self.mean[lbl][0] = 0; self.mean[lbl][1] = 0
			self.freq[lbl] = 0
			
		for row in range(self.shape[0]):
			for col in range(self.shape[1]):
				if not self.color_map[labels[row,col]].has_key(self.rgb_idx[row,col]):
					self.color_map[labels[row,col]].__setitem__(self.rgb_idx[row,col],1);
				else:
					self.color_map[labels[row,col]][self.rgb_idx[row,col]] += 1
				self.freq[labels[row,col]] += 1	
				self.mean[labels[row,col]][0] += (row - self.mean[labels[row,col]][0])/self.freq[labels[row,col]]
				self.mean[labels[row,col]][1] += (col - self.mean[labels[row,col]][1])/self.freq[labels[row,col]]
		print "Build region (preprocess) : ",time.time()-start_time
		return (labels,_num_superpixels);
	
	#compute region contrast	
	def region_contrast(self,_num_superpixels,dist_weight = 1):
		start_time = time.time();
		for x_idx in range(_num_superpixels):
			x_colors = self.color_map[x_idx]; x_freq = self.freq[x_idx];
			for y_idx in range(x_idx+1,_num_superpixels):
				y_colors = self.color_map[y_idx]; y_freq = self.freq[y_idx];
				_col_dist = 0;  
				for x_color in x_colors.keys():
					f_x = float(x_colors[x_color])/x_freq;
					for y_color in y_colors.keys():
						f_y =  float(y_colors[y_color])/y_freq
						_col_dist += f_y * self.lab_dist[x_color,y_color];
					_col_dist *= f_x
				self.col_dist[x_idx,y_idx]=self.col_dist[y_idx,x_idx]=_col_dist
		_norm = np.sqrt(self.shape[0]*self.shape[0] + self.shape[1]*self.shape[1]);
		
		for x_idx in range(_num_superpixels):
			_saliency = 0
			for y_idx in range(_num_superpixels):		
				if x_idx != y_idx:
					_dist = np.exp(-np.linalg.norm(self.mean[x_idx]-self.mean[y_idx])/(_norm * dist_weight))
					_saliency += _dist * self.col_dist[x_idx,y_idx]*self.freq[y_idx]
			self.saliency[x_idx] = _saliency
		_min = np.min(self.saliency); _max = np.max(self.saliency);
		self.saliency -= _min
		self.saliency /= (_max-_min)
		print "Compute Saliency : ",time.time()-start_time
		
				
	def process(self):
		self.cur_frame = self._nextFrame()
		if self.cur_frame is None:
			self.finish = True
		else:
			self.quantize()
			(reg_labels,num_superpixels_found) = self.build_region()
			self.region_contrast(num_superpixels_found)
			_,self.saliency = cv2.threshold(self.saliency,0.4,1,cv2.THRESH_BINARY)
			for row in range(self.shape[0]):
				for col in range(self.shape[1]):
					self.p_frame[row,col,2]= np.uint8(self.saliency[reg_labels[row,col]] * 255)
			return cv2.addWeighted(self.cur_frame,0.6,self.p_frame,0.4,0.0);
			

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
		over_segmentation(sys.argv[1]);
	elif sys.argv.__len__()==3:
		over_segmentation(sys.argv[1],sys.argv[2]);

