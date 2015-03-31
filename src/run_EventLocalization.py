import os,cv2,numpy as np,time;
from skimage.morphology import remove_small_objects
from skimage.measure import label
from scipy.ndimage.morphology import binary_fill_holes
from utils import DummyTask	
from multiprocessing.pool import ThreadPool
from collections import deque

from features.color import GRAY
from io_module.video_reader import VideoReader
from io_module.video_writer import VideoWriter
from saliency import SaliencyProps,SaliencyMethods,get_instance as sal_instance
from bg_sub import get_instance,BGMethods
from smoothing import get_instance as smooth_instance,SmoothMethods
from features.pixellete import allFeats as feats
from utils import normalize,create_folder_structure_if_not_exists

MAX_COLORS = 100
RANDOM_COLORS = np.random.randint(256, size=(MAX_COLORS, 3))
KERNEL = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3));
bgdModel = np.zeros((1,65),np.float64); fgdModel = np.zeros((1,65),np.float64)


def __morphologicalOps__(masks):
	new_masks = [];
	for mask in masks:
		#_mask = binary_fill_holes(mask)
		_mask = cv2.medianBlur(np.uint8(mask),3)
		_mask = cv2.morphologyEx(_mask, cv2.MORPH_OPEN, KERNEL)
		_mask = cv2.medianBlur(_mask,5)
		_mask = cv2.morphologyEx(_mask, cv2.MORPH_CLOSE,KERNEL)
		_mask = binary_fill_holes(_mask)
		_mask = remove_small_objects(_mask,min_size=128,connectivity=2)
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
		_mask = cv2.dilate(np.uint8(_mask),kernel,iterations = 2)
		new_masks.extend([_mask])
	return new_masks;

def compute_fgbg_masks(vidreader,sal,bg,num_prev_frames=1,threaded=False,num_blocks=20):
	start = time.time();
	def compute_mask(frameIdx,frames):
		num_frames = len(frames);	_idx = num_prev_frames
		fgMasks = []; bgMasks = [];
		while _idx<num_frames:
			prev_frames = frames[_idx-num_prev_frames:_idx]
			bg_variation = bg.process(frames[_idx],prev_frames);
			saliency = sal.process(frames[_idx]);	
			_idx += 1;	saliency_prob =  normalize(6*bg_variation  + 4*saliency);
			_,fgmask = cv2.threshold(saliency_prob ,0.5,1,cv2.THRESH_BINARY)
			_,bgmask = cv2.threshold(bg_variation,0.1,1,cv2.THRESH_BINARY_INV)
			fgMasks.extend([fgmask]); bgMasks.extend([bgmask])
		return (frameIdx,fgMasks,bgMasks);
	frameFgMasks = []; frameBgMasks = []; 
	frameIdx = num_prev_frames+num_blocks; N = vidreader.frames;
	threadn = cv2.getNumberOfCPUs();	pool = ThreadPool(processes = threadn/2);
	pending = deque();	N = vidreader.frames;
	while True:
		while len(pending) > 0 and pending[0].ready():
			task = pending.popleft()
			idx,fgmask,bgmask  = task.get()
			frameFgMasks.extend(fgmask); frameBgMasks.extend(bgmask);
			#print 'Computing Mask ... {0}%\r'.format((idx*100/N)),
		if len(pending) < threadn and frameIdx-num_prev_frames-num_blocks < N:
			(cnt,frames) = vidreader.read(frameIdx-num_prev_frames-num_blocks,num_prev_frames+num_blocks);
			if cnt >= num_prev_frames:
				if threaded:
					task = pool.apply_async(compute_mask,[min(frameIdx,N),frames]);
				else:
					task = DummyTask(compute_mask(min(frameIdx,N),frames));
				pending.append(task)
			frameIdx += num_blocks;	
		if len(pending) == 0:
			break;
	time_taken = time.time()-start;	
	#print "Computing Mask ... [DONE] in ",time_taken," seconds"
	return (frameFgMasks,frameBgMasks)

def perform_smoothing(vidreader,smoothner,fgMasks,bgMasks,num_blocks=4,skip_frames = 0):
	start = time.time();	frameNewMasks = []; 
	frame_idx = 2*num_blocks;  N = vidreader.frames;
	while(frame_idx < N):
		#print 'Smoothning Masks... {0}%\r'.format((frame_idx*100/N)),
		(num_frames,frames) = vidreader.read(skip_frames + frame_idx-2*num_blocks,2*num_blocks);
		if num_frames > num_blocks:
			start_idx = num_blocks/2; end_idx = min(3*num_blocks/2,num_frames);
			s = frame_idx-2*num_blocks; e = s + num_frames;
			newMasks = smoothner.process(frames,fgMasks[s:e],bgMasks[s:e],range(start_idx,end_idx));
			newMasks = __morphologicalOps__(newMasks);	
			frameNewMasks.extend(newMasks);
		frame_idx += num_blocks
	time_taken = time.time()-start;	
	#print "Smoothning Masks ... [DONE] in ",time_taken," seconds"
	return frameNewMasks;

	
def write_video(vidreader,out_path,num_prev_frames,num_blocks,smoothMasks):
	create_folder_structure_if_not_exists(out_path);
	start = time.time(); 
	vidwriter = VideoWriter(out_path,vidreader.width,vidreader.height)
	vidwriter.build();	vidreader.__reset__();	N = vidreader.frames;
	vidreader.skip_frames(num_prev_frames + num_blocks/2); frame_idx = 0	
	numFrames= len(smoothMasks);
	while(frame_idx < numFrames):
		#print 'Writing video ... {0}%\r'.format((frame_idx*100/N)),
		frame = vidreader.read_next();
		out_frame = GRAY(frame) * smoothMasks[frame_idx]
		out_frame = cv2.cvtColor(out_frame,cv2.COLOR_GRAY2RGB);
		vidwriter.write(np.uint8(out_frame))
		frame_idx += 1;
	vidwriter.close();
	time_taken = time.time()-start;	
	#print "Writing video .... [DONE] in ",time_taken," seconds"

def open_write_header(extract_path,shape,window):
	create_folder_structure_if_not_exists(extract_path);
	_file = open(extract_path,'w');		num_features = np.prod(shape)*window;
	_file.write(str(num_features));		_file.write(os.linesep)
	return _file;
	
def extract_feats(vidreader,extract_path,num_prev_frames,num_blocks,smoothMasks,window=5,overlap=2):
	assert(window>overlap),"Window size needs to be greater than overlap window"
	_file = open_write_header(extract_path,smoothMasks[0].shape,window);
	start = time.time(); 
	frame_idx = num_prev_frames + num_blocks/2	
	numFrames= len(smoothMasks);
	while(frame_idx < numFrames):
		#print 'Extracting Feats ... {0}%\r'.format((frame_idx*100/numFrames)),
		(cnt,frames) = vidreader.read(frame_idx,window);	
		if cnt==window:
			w_frames = [];
			for frame in frames:
				frame = GRAY(frame) * smoothMasks[frame_idx]; 
				w_frames.extend(frame.flatten());
			w_frames = np.array(w_frames)/float(255);
			w_frames.tofile(_file,' ','%0.3f');
			_file.write(os.linesep)
		frame_idx += window-overlap;
	time_taken = time.time()-start;	
	#print "Extracting Feats .... [DONE] in ",time_taken," seconds"

def process(vidreader,sal,bg,smoothner,num_prev_frames=3 ,num_blocks=6, write_path=None,
				extract_path=None,window_size=5,overlap=2,threaded=False):
	fgMasks,bgMasks = compute_fgbg_masks(vidreader,sal,bg,num_prev_frames,threaded);
	smoothMasks = perform_smoothing(vidreader,smoothner,fgMasks,bgMasks,num_blocks,num_prev_frames);
	if not write_path is None:
		write_video(vidreader,write_path,num_prev_frames,num_blocks,smoothMasks);
	if not extract_path is None:
		extract_feats(vidreader,extract_path,num_prev_frames,num_blocks,smoothMasks,window_size,overlap)
	vidreader.close();
		

if __name__ == "__main__":	
	import argparse
	parser = argparse.ArgumentParser(description='Event tracking algorithm using saliency and bg subtraction')
	parser.add_argument("input",nargs='?',help = "input path",default="../examples/videos/sample_video1.avi");
	parser.add_argument("output",nargs='?',help = "output path",default="test_results/final.avi");
	parser.add_argument("--bg",nargs='?',help = "background subtraction method 1-FD, 2-ES, 3-MG default-2",default=1,type=int);
	parser.add_argument("--sal",nargs='?',help = "saliency method 1-CF, 2-CA, 3-RC, 4-SD default-3",default=3,type=int);
	parser.add_argument("--smoothner",nargs='?',help = "smoothing method 1-Eigen, 2-GMM, 3-SSL,  default-2",default=2,type=int);
	parser.add_argument("--write",nargs='?',help = "write path  default-Dont write",default="test_results/final.avi",type=str);
	parser.add_argument("--extract",nargs='?',help = "extract path default-Dont extract",default="test_results/final.feats",type=str);
	parser.add_argument("--num_prev_frames",nargs='?',help = "num prev frames default-3",default=3,type=int);
	parser.add_argument("--num_blocks",nargs='?',help = "num blocks default-6",default=6,type=int);
	parser.add_argument("--window_size",nargs='?',help = "window size default-5",default=5,type=int);
	parser.add_argument("--overlap",nargs='?',help = "overlap default-2",default=2,type=int);
	args = parser.parse_args()
	inp = args.input
	vidreader = VideoReader(inp)
	sal = sal_instance(args.sal,SaliencyProps())	
	bg = get_instance(args.bg);
	smoothner = smooth_instance(feats,args.smoothner);
	process(vidreader,sal,bg,smoothner,args.num_prev_frames,args.num_blocks,args.write,args.extract,
				args.window_size,args.overlap);
	print "Event Localization ...",inp,"[DONE]"
