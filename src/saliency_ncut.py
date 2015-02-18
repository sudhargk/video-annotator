import cv2,time
import numpy as np
from skimage.future import graph
from skimage.measure import regionprops
from skimage import  data, io, segmentation, color

class Saliency(object):	
	def __init__(self, shape, num_superpixels = 300,compactness = 40):
		self.PIXEL_MARGIN = 50;
		self.shape = shape
		self.num_superpixels = num_superpixels
		num_superpixels +=self.PIXEL_MARGIN
		self.compactness = compactness
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
		self.s_frame = color.label2rgb(labels,self.q_cur_frame, kind='avg')
		cv2.imwrite('outs.png',self.s_frame);
		print "Slic time : ",time.time()-start_time
		return labels;
	

	def normalized_graphcut(self,s_labels):
		start_time = time.time();
		_graph = graph.rag_mean_color(self.q_cur_frame, s_labels, mode='similarity')
		labels = graph.cut_normalized(s_labels,_graph)
		#self.c_frame = color.label2rgb(labels,self.q_cur_frame, kind='avg')
		cv2.imwrite('oucut.png',self.s_frame);
		print "Graph N Cut(preprocess) : ",time.time()-start_time
		
	#process input
	def process(self,cur_frame):
		self.quantize(cur_frame)
		reg_labels = self.build_region()
		self.normalized_graphcut(reg_labels)
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

