import cv2,numpy as np,time;
from skimage.morphology import remove_small_objects
from skimage.measure import label
from scipy.ndimage.morphology import binary_fill_holes
from utils import DummyTask	
from multiprocessing.pool import ThreadPool
from collections import deque

from io_module.video_reader import VideoReader
from io_module.video_writer import VideoWriter
from saliency import SaliencyProps,SaliencyMethods,get_instance as sal_instance
from bg_sub import get_instance,BGMethods
from tracker import get_instance as tracker_instance
from smoothing import get_instance as smooth_instance,SmoothMethods
from features.pixellete import allFeats as feats
from utils import normalize

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
		_mask = cv2.medianBlur(_mask,3)
		_mask = cv2.morphologyEx(_mask, cv2.MORPH_CLOSE,KERNEL)
		_mask = binary_fill_holes(_mask)
		_mask = remove_small_objects(_mask,min_size=128,connectivity=2)
		new_masks.extend([_mask])
	return np.uint8(new_masks);


def detect_object(newMask,oldMask=None,delta = 10):
	window = []; (lbls,num) = label(newMask,connectivity=2,neighbors=4,return_num=True,background=0)
	for lbl in range(np.max(lbls)+1):
		pixels = np.where(lbls==lbl); _max = np.max(pixels,1); _min = np.min(pixels,1)
		area = np.prod(_max - _min);
		if  (4 * np.sum(newMask[_min[0]:_max[0],_min[1]:_max[1]]) > area) and \
			((oldMask is None) or (np.sum(oldMask[_min[0]:_max[0],_min[1]:_max[1]]) > 10)):
			rect = np.array([_min[1],_min[0],_max[1],_max[0]],dtype=int);
			window.extend([rect]);
	
	if (len(window)>1):				#mergin along x-axis
		order = np.array(window)[:,0].argsort(); prev_idx = 0; cur_idx = 1;
		new_window = [window[order[0]]];
		while(cur_idx < len(window)):
			if new_window[prev_idx][2] + delta > window[order[cur_idx]][0] and \
				((new_window[prev_idx][1] < window[order[cur_idx]][3] and new_window[prev_idx][1] > window[order[cur_idx]][1]) or \
				(new_window[prev_idx][3] < window[order[cur_idx]][3] and new_window[prev_idx][3] > window[order[cur_idx]][1])):
				new_window[prev_idx][2]= max(new_window[prev_idx][2],window[order[cur_idx]][2]);
				new_window[prev_idx][1]= min(new_window[prev_idx][1],window[order[cur_idx]][1]);
				new_window[prev_idx][3]= max(new_window[prev_idx][3],window[order[cur_idx]][3]);
			else:
				new_window.extend([window[order[cur_idx]]])
				prev_idx += 1
			cur_idx += 1;
		window = new_window;
	if (len(window)>1):						#mergin along y-axis 
		order = np.array(window)[:,1].argsort(); prev_idx = 0; cur_idx = 1;
		new_window = [window[order[0]]];
		while(cur_idx < len(window)):
			if new_window[prev_idx][3] + delta > window[order[cur_idx]][1] and \
				((new_window[prev_idx][0] < window[order[cur_idx]][2] and new_window[prev_idx][0] > window[order[cur_idx]][0]) or \
				(new_window[prev_idx][2] < window[order[cur_idx]][2] and new_window[prev_idx][2] > window[order[cur_idx]][0])):
				new_window[prev_idx][2]= max(new_window[prev_idx][2],window[order[cur_idx]][2]);
				new_window[prev_idx][1]= min(new_window[prev_idx][1],window[order[cur_idx]][1]);
				new_window[prev_idx][3]= max(new_window[prev_idx][3],window[order[cur_idx]][3]);
			else:
				new_window.extend([window[order[cur_idx]]])
				prev_idx += 1
			cur_idx += 1;
		window = new_window;
	
	return window;
	
def write_block(vidwriter,frame,newMask,oldMask,windows,window_lbl):
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
	newMask = cv2.dilate(newMask,kernel,iterations = 2)
	zero_frame  = np.zeros((newMask.shape[0],newMask.shape[1],3),dtype=np.float32);
	zero_frame[:,:,2] = np.float32(oldMask*255);
	out_frame1 = cv2.addWeighted(np.float32(frame),0.6,zero_frame,0.4,0.0);
	out_frame2 = frame
	"""
	#zero_frame  = np.zeros((newMask.shape[0],newMask.shape[1]),dtype=np.float32);
	#for (idx,window) in enumerate(windows):
	#	zero_frame[window[1]:window[3],window[0]:window[2]] = newMask[window[1]:window[3],window[0]:window[2]];
		#cv2.rectangle(out_frame2, (window[0],window[1]), (window[2],window[3]), tuple(RANDOM_COLORS[window_lbl[idx],:]))
		#zero_frame[window[1]:window[3],window[0]:window[2]] = 1;
		#cv2.putText(out_frame2,str(window_lbl[idx]), ((window[0]+window[2])/2,(window[1]+window[3])/2), cv2.FONT_HERSHEY_DUPLEX, 0.25, 255)
	#zero_frame[:,:] = np.float32(newMask);
	#out_frame2 = cv2.addWeighted(np.float32(frame),0.6,zero_frame,0.4,0.0);
	"""
	out_frame2 = frame * newMask[:,:,None]
	out_frame = np.hstack((out_frame1,out_frame2))
	vidwriter.write(np.uint8(out_frame))

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
			_,fgmask = cv2.threshold(saliency_prob ,0.6,1,cv2.THRESH_BINARY)
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
			print 'Computing Mask ... {0}%\r'.format((idx*100/N)),
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
	print "Computing Mask ... [DONE] in ",time_taken," seconds"
	return (frameFgMasks,frameBgMasks)

def perform_smoothing(vidreader,smoothner,fgMasks,bgMasks,num_blocks=4,skip_frames = 0):
	start = time.time();	frameNewMasks = []; 
	frame_idx = 2*num_blocks;  N = vidreader.frames;
	while(frame_idx < N):
		print 'Smoothning Masks... {0}%\r'.format((frame_idx*100/N)),
		(num_frames,frames) = vidreader.read(skip_frames + frame_idx-2*num_blocks,2*num_blocks);
		if num_frames > num_blocks:
			start_idx = num_blocks/2; end_idx = min(3*num_blocks/2,num_frames);
			s = frame_idx-2*num_blocks; e = s + num_frames;
			newMasks = smoothner.process(frames,fgMasks[s:e],bgMasks[s:e],range(start_idx,end_idx));
			newMasks = __morphologicalOps__(newMasks);	
			frameNewMasks.extend(newMasks);
		frame_idx += num_blocks
	time_taken = time.time()-start;	
	print "Smoothning Masks ... [DONE] in ",time_taken," seconds"
	return frameNewMasks;


def process(vidreader,out_path,sal,bg,smoothner,num_prev_frames=3  ,num_blocks=4,threaded=False):
	fgMasks,bgMasks = compute_fgbg_masks(vidreader,sal,bg,num_prev_frames,threaded);
	smoothMasks = perform_smoothing(vidreader,smoothner,fgMasks,bgMasks,num_blocks,num_prev_frames);
	tracker = tracker_instance(0);  start = time.time(); 
	vidwriter = VideoWriter(out_path,2*vidreader.width,vidreader.height)
	
	vidwriter.build();	vidreader.__reset__();	N = vidreader.frames;
	vidreader.skip_frames(num_prev_frames + num_blocks/2); frame_idx = 0	
	numFrames= len(smoothMasks);
	while(frame_idx < numFrames):
		print 'Tracking ... {0}%\r'.format((frame_idx*100/N)),
		frame = vidreader.read_next();
		window = detect_object(smoothMasks[frame_idx],fgMasks[frame_idx])
		window_label = tracker.track_object(frame,window,smoothMasks[frame_idx]);
		write_block(vidwriter,frame,smoothMasks[frame_idx],fgMasks[frame_idx],window,window_label);
		frame_idx += 1;
	vidreader.close();
	vidwriter.close();
	time_taken = time.time()-start;	
	print "Tracking .... [DONE] in ",time_taken," seconds"


if __name__ == "__main__":	
	import argparse
	parser = argparse.ArgumentParser(description='Event tracking algorithm using saliency and bg subtraction')
	parser.add_argument("input",nargs='?',help = "input path",default="../examples/videos/sample_video1.avi");
	parser.add_argument("output",nargs='?',help = "output path",default="test_results/final.avi");
	parser.add_argument("--bg",nargs='?',help = "background subtraction method 1-FD, 2-ES, 3-MG default-2",default=1,type=int);
	parser.add_argument("--sal",nargs='?',help = "saliency method 1-CF, 2-CA, 3-RC, 4-SD default-3",default=3,type=int);
	parser.add_argument("--smoothner",nargs='?',help = "smoothing method 1-Eigen, 2-GMM, 3-SSL,  default-2",default=2,type=int);
	args = parser.parse_args()
	inp = args.input;	out = args.output
	vidreader = VideoReader(inp)
	sal = sal_instance(args.sal,SaliencyProps())	
	bg = get_instance(args.bg);
	smoothner = smooth_instance(feats,args.smoothner);
	process(vidreader,out,sal,bg,smoothner)
