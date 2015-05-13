import cv2,numpy as np,time,os
from multiprocessing.pool import ThreadPool
from collections import deque
from skimage.feature import canny
from skimage.morphology import remove_small_objects
from skimage.measure import label
from scipy.ndimage.morphology import binary_fill_holes

from pylab import cm
from dnn_predict import get_instance as dnn_instance
from io_module.video_reader import VideoReader
from io_module.video_writer import VideoWriter;
from io_module.block_reader import BlockReader
from saliency import SaliencyProps,SaliencyMethods,get_instance as sal_instance
from bg_sub import get_instance as bg_instance,BGMethods
from tracker import get_instance as tracker_instance,TrackerMethods
from smoothing import get_instance as smooth_instance,SmoothMethods
from features.pixellete import allFeats as feats
from features.color import GRAY
from utils import normalize

sal = sal_instance(SaliencyMethods.REGION_CONTRAST,SaliencyProps())	
bg = bg_instance(BGMethods.FRAME_DIFFERENCING);
smoothner =  smooth_instance(feats,SmoothMethods.GMM_BASED);
tracker = tracker_instance(TrackerMethods.MIXTURE_BASED);

KERNEL = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3));
def __morphologicalOps__(mask):
	#_mask = binary_fill_holes(mask)
	_mask = cv2.medianBlur(np.uint8(mask),3)
	_mask = cv2.morphologyEx(_mask, cv2.MORPH_CLOSE,KERNEL)
	_mask = binary_fill_holes(_mask)
	_mask = remove_small_objects(_mask,min_size=128,connectivity=2)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
	_mask = cv2.dilate(np.uint8(_mask),kernel,iterations = 1)
	return _mask;
	
def load_labels(fileName):
	idx = 0; labels ={}
	with open(fileName,'r') as f:
		for line in f:
			labels[idx] = line[:-1];
			idx += 1;
	return labels;
	
def __draw_str__(dst, (x, y), s, fontface = cv2.FONT_HERSHEY_SIMPLEX,fontsize = 0.3, color=[255, 255, 255]):
	cv2.putText(dst, s, (x, y),fontface, fontsize, tuple(color), lineType=cv2.CV_AA)

def __draw_rect__(frame,window_center,window_size,sMask):
	sMask = __morphologicalOps__(sMask);
	ac_shape = frame.shape[:2];
	zero_frame= np.zeros(frame.shape,dtype = np.uint8);
	zero_frame[:,:,2]=sMask*255;
	frame = cv2.addWeighted(np.uint8(frame),0.6,zero_frame,0.4,0);
	window_size = np.uint8(window_size);
	left = np.array(window_center - window_size/2,dtype=np.int8);
	left[0] = min(left[0],ac_shape[0]-window_size[0]); left[0] = max(left[0],0)
	left[1] = min(left[1],ac_shape[1]-window_size[1]); left[1] = max(left[1],0)
	top_left = (left[0],left[1]); bottom_right = (left[0]+window_size[0],left[1]+window_size[1]);
	cv2.rectangle(frame,top_left, bottom_right, 255, 2)
	return np.uint8(frame);

def __crop_frame__(frame,window_center,window_size,rsz_shape):
	ac_shape = frame.shape[:2];
	zero_frame = np.zeros((rsz_shape[1],rsz_shape[0],3));
	left = np.array(window_center - window_size/2,dtype=np.int8);
	left[0] = min(left[0],ac_shape[0]-window_size[0]); left[0] = max(left[0],0)
	left[1] = min(left[1],ac_shape[1]-window_size[1]); left[1] = max(left[1],0)
	_frame = np.uint8(frame[left[1]:left[1]+window_size[1],left[0]:left[0]+window_size[0]])
	_ratio = rsz_shape[0]/float(window_size[0]);
	_frame = cv2.resize(_frame,None,fx=_ratio, fy=_ratio);
	_shape = _frame.shape; 
	x_l = max(rsz_shape[0]/2 - _shape[1]/2,0);  y_l = max(rsz_shape[1]/2 - _shape[0]/2,0);
	if _frame.shape[0] > rsz_shape[1]:
		zero_frame[y_l:y_l+_shape[0],x_l:x_l+_shape[1]]=_frame[:rsz_shape[1],:];
	else:
		zero_frame[y_l:y_l+_shape[0],x_l:x_l+_shape[1]]=_frame;
	return np.uint8(zero_frame).flatten()/float(255);

def sobel(frame):
	frame = GRAY(frame);
	gx = cv2.Sobel(frame, cv2.CV_32F, 1, 0)
	gy = cv2.Sobel(frame, cv2.CV_32F, 0, 1)
	mag, ang = cv2.cartToPolar(gx, gy)
	return normalize(mag);
	

def compute_fgbg_masks(saliency,bg_variation):
	saliency_prob =  normalize(6*bg_variation  + 4*saliency);
	_,fgmask = cv2.threshold(saliency_prob, 0.6, 1, cv2.THRESH_BINARY)
	_,bgmask = cv2.threshold(bg_variation, 0.1, 1, cv2.THRESH_BINARY_INV)
	return fgmask,bgmask
	
def compute_vas_masks(frames,num_prev_frames=1,num_blocks=4):
	# Finding true foreground and true background
	fgMasks = []; bgMasks = [];
	for frame in frames[:num_prev_frames]:
		bg_variation = sobel(frame); saliency = sal.process(frame);
		fgmask,bgmask = compute_fgbg_masks(saliency,bg_variation)
		fgMasks.extend([fgmask]); bgMasks.extend([bgmask])
	idx = num_prev_frames; num_frames = len(frames);
	while idx<num_frames:
		bg_variation = bg.process(frames[idx],frames[idx-num_prev_frames:idx]);
		saliency = sal.process(frames[idx]); idx += 1;
		fgmask,bgmask = compute_fgbg_masks(saliency,bg_variation)
		fgMasks.extend([fgmask]); bgMasks.extend([bgmask])
	
	#smoothening the mask	
	sMasks = [] ; idx = 2*num_blocks; 
	start_idx =  end_idx = 3*num_blocks/2;
	while(idx <= num_frames):
		start_idx = (0	if idx == 2*num_blocks else num_blocks/2);
		end_idx =  (num_frames if idx == num_frames else 3*num_blocks/2);
		s = idx-2*num_blocks; e = s + 2*num_blocks; idx += num_blocks
		sMasks.extend(smoothner.process(frames[s:e],fgMasks[s:e],bgMasks[s:e],range(start_idx,end_idx)));
	return sMasks
	
def processClassFrames(frames,contextSize):
	start = time.time();
	height,width = frames[0].shape[:2];
	sMasks = compute_vas_masks(frames); 	
	windowCenters = []; smoothenedMasks = [];
	prev_rect = np.array([width/4,height/4,width/2,height/2],dtype=np.float32);
	prev_center = np.array([width/2,height/2],	dtype=np.float32);
	windowSize = np.zeros(2);
	for frame_idx in range(len(frames)/contextSize):
		s = frame_idx; e = s + contextSize; 
		propWindows = tracker.track_object(frames[s:e],sMasks[s:e]);
		windowCenter = []; smoothenedMasks.extend([sMasks[s:e]]);
		for propWindow in propWindows:
			(window,lbl) = propWindow[0]
			rect = np.array([window[0],window[1],window[2]-window[0],window[3]-window[1]],dtype=np.float32);
			cv2.accumulateWeighted(rect,prev_rect,0.25,None);
			window_center = prev_rect[:2]+(prev_rect[2:]/2); 
			cv2.accumulateWeighted(window_center,prev_center,0.35,None);
			window_center = prev_center.copy();
			windowSize = np.maximum(windowSize,prev_rect[2:]+np.array([10,10]));
			windowCenter.extend([window_center]);
		windowCenters.extend([[windowCenter,windowSize]]);
	print "Time Taken ", (time.time()-start),"seconds"
	return windowCenters,smoothenedMasks;
		
class BlockTask(object):
	def __init__(self,frameBlock,processedWindowsBlock,smoothenedMasks,
						labels,cnt, rsz_shape=[80,60],block_size=5):
		self.frameBlock = frameBlock;
		self.processedWindowsBlock = processedWindowsBlock;
		self.smoothenedMasks = smoothenedMasks;
		self.labels = labels;
		self.cnt = cnt;
		self.rsz_shape=rsz_shape
		self.block_size = block_size;
	
	def __buildFrames__(self):
		dispFrames = []; blocksFeats = []; groupIdx = 0;
		for frameGroup,processedWindows,smoothenedMasksBlock in zip(self.frameBlock[:self.cnt],self.processedWindowsBlock[:self.cnt],self.smoothenedMasks[:self.cnt]):
			windowCenters,windowSize = processedWindows;
			blockFeats = [];
			for frame,windowCenter,smoothenedMask in zip(frameGroup,windowCenters,smoothenedMasksBlock):
				dispFrames.extend([__draw_rect__(frame,windowCenter,windowSize,smoothenedMask)]);
				blockFeats.extend(__crop_frame__(frame,windowCenter,windowSize,self.rsz_shape))
			blockFeats = np.array(blockFeats).reshape([self.block_size,self.rsz_shape[1],self.rsz_shape[0],3])
			blockFeats = np.swapaxes(blockFeats,1,3);
			blockFeats = np.swapaxes(blockFeats,2,3);	
			blocksFeats.extend([blockFeats]);
		feat_size = np.prod(self.rsz_shape[:2])*self.block_size
		for frameGroup in self.frameBlock[self.cnt:]:
			blockFeats = np.zeros([self.block_size,3,self.rsz_shape[1],self.rsz_shape[0]],dtype=float);
			blocksFeats.extend([blockFeats]);
		return dispFrames, np.array(blocksFeats);
		
	def process(self,vidwriter,predictor,designFrameBaner):
		scores = [];
		dispFrames,processed_frames = self.__buildFrames__();
		start = time.time();
		scores = predictor.get_score(processed_frames);
		for idx,frame in enumerate(dispFrames):
			blockIdx = idx/self.block_size;
			_frame = designFrameBaner(frame,scores[blockIdx],self.labels[blockIdx]);
			vidwriter.write(np.uint8(_frame));

class UCF50Processor(object):
	def __init__(self,vidfile,labelfile,modelFile,labelListFile,exportPath,
							perClassFrames=40,frameRate=20,context_size=5,
							banerWidth = 80,scale = 1):
		# input properties
		self.vidreader = VideoReader(vidfile);
		self.labelreader = open(labelfile,'r');
		self.N = self.vidreader.frames;
		self.width,self.height = self.vidreader.width,self.vidreader.height;
		self.context_size=context_size;
		self.perClassFrames = perClassFrames;
		self.labels = load_labels(labelListFile);
		self.n_outs = len(self.labels);
		self.flag_colors = [];
		for index in range(self.n_outs):
			self.flag_colors.extend([tuple(np.array(cm.jet(index/float(self.n_outs))[:3][::-1])*255)])
			
		with open(modelFile,'r') as fp:
			model = fp.readline();
			self.predictor = dnn_instance(model.strip());
		self.input_shape = [self.height,self.width,3]
		self.batch_size = self.predictor.batch_size
		
		#write properites
		self.banerWidth = banerWidth
		self.vidWriter = VideoWriter(exportPath,self.banerWidth+self.width,self.height,fps=frameRate);
		self.colors = np.random.randint(256, size=(len(self.labels), 3))
		self.scale = scale;
		
		#status
		self.frameIdx = 0;
		self.tasks = deque();
		self.isFinished = False;
		self.vidWriter.build();
	
	def __design_frame_banner__(self,frame,_score,_label,top=3):
		if not self.scale == 1:
			frame = cv2.resize(frame,None,fx=self.scale, fy=self.scale, interpolation = cv2.INTER_CUBIC)
		if frame.ndim == 2:
			frame = np.dstack((frame,frame,frame));
		assert(frame.ndim==3),"given frame not in shape"
		baner_frame = np.zeros((self.height,self.banerWidth,3));
		_indices = np.argsort(_score)[::-1];
		col = 5; row = 8; steps = ((self.height-30)/(top+1))-5;
		small_fface = cv2.FONT_HERSHEY_DUPLEX;
		__draw_str__(baner_frame,(col+3,row),">PREDICTION<",color=(255,255,255),fontsize=0.25,fontface=small_fface); row += steps
		for pos,classLbl in enumerate(_indices[:top]):
			_str = "{0}. {1}".format(pos+1,self.labels[classLbl]);
			__draw_str__(baner_frame,(col,row),_str,color=self.colors[classLbl],fontsize=0.25); row += steps
		__draw_str__(baner_frame,(col+3,row),">ACTUAL<",color=(255,255,255),fontsize=0.25,fontface=small_fface); row += steps
		if not _label is None:
			_str = "{0}".format(self.labels[_label]); 
			__draw_str__(baner_frame,(col,row),_str,color=self.colors[_label],fontsize=0.25);
			rank = list(_indices).index(_label); _str = "Rank : {0}".format(rank+1); row += steps;
			__draw_str__(baner_frame,(col+3,row),_str,color = (255,255,255),fontsize=0.25);
			cv2.rectangle(baner_frame,(8,self.height-12),(self.banerWidth-8,self.height-3),(255,255,255),1);
			cv2.rectangle(baner_frame,(10,self.height-10),(self.banerWidth-10,self.height-5),self.flag_colors[rank],-1);
		return np.hstack((baner_frame,frame));	
		
	def __readNextFrame__(self):
		if self.vidreader.has_next():
			frames = []; label = -1;
			for idx in range(self.context_size):
				frame = self.vidreader.read_next();
				label = int(self.labelreader.readline()); 
				frames.extend([frame]); 
			return (frames,label);
		else:
			return (None,-1);
	
	def process(self):
		blockReader = BlockReader(self.batch_size,self.input_shape,self.__readNextFrame__);
		#pool = ThreadPool(processes = 2);
		#task = pool.apply_async(self.__videoWriter__);
		p_frames_cnt = 0
		while not blockReader.isFinished:
			(frameCnt,frames,labels)=blockReader.readNextBlock();
			#print "READ A BLOCK...", len(self.tasks)
			if frameCnt > 0:
				processedWindowCenter,smoothenedMasks = self.__process_block__(frames[:frameCnt]);
				vidTask = BlockTask(frames,processedWindowCenter,smoothenedMasks,labels,frameCnt)
				#self.tasks.append(vidTask);
				vidTask.process(self.vidWriter,self.predictor,self.__design_frame_banner__);
		self.isFinished = True;	
		
	
	def __process_block__(self,frameBlock):
		windowCenters = []; smoothenedMasks = [];
		numGroupsPerClass = self.perClassFrames/self.context_size;
		for idx in range(len(frameBlock)/numGroupsPerClass):
			classFrames = [];
			for frameGroup in frameBlock[idx*numGroupsPerClass:(idx+1)*numGroupsPerClass]:
				for frame in frameGroup:
					classFrames.extend([frame]);
			c_windowCenters,sMask = processClassFrames(classFrames,self.context_size)
			windowCenters.extend(c_windowCenters);
			smoothenedMasks.extend(sMask);
		return windowCenters,smoothenedMasks;
		
if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Testing MNIST dataset with model')
	parser.add_argument("-input",nargs='?',help = "input path of data",default="sample_test_ucf");
	parser.add_argument("-group",nargs='?',help = "input path of data",default="BM");
	parser.add_argument("-output",nargs='?',help = "save path",default="testout.avi");
	
	args = parser.parse_args();
	inputVid = args.input + os.sep + args.group + os.sep + "test.avi"
	label  = args.input + os.sep + args.group + os.sep + "test.txt"
	model = args.input + os.sep + args.group + os.sep + "test.model"
	labelmap = args.input + os.sep + args.group + os.sep + "labels.txt"
	output = args.input + os.sep + args.group + os.sep + args.output
	vidProcessor = UCF50Processor(inputVid,label,model,labelmap,output);
	vidProcessor.process();
	print 'Processing UCF dataset [DONE]'		
		
