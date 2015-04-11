import cv2,numpy as np,time,os
from multiprocessing.pool import ThreadPool
from collections import deque

from io_module.video_reader import VideoReader
from io_module.video_writer import VideoWriter
from saliency import SaliencyProps,SaliencyMethods,get_instance as sal_instance
from bg_sub import get_instance,BGMethods
from tracker import get_instance as tracker_instance,TrackerMethods
from smoothing import get_instance as smooth_instance,SmoothMethods
from features.pixellete import allFeats as feats
from features.color import GRAY
from utils import normalize,DummyTask,create_folder_structure_if_not_exists

MAX_COLORS = 100
RANDOM_COLORS = np.random.randint(256, size=(MAX_COLORS, 3))

"""
def write_block(vidwriter,frame,newMask,oldMask,window_centers,window_size):
	shape = frame.shape;
	zero_frame  = np.zeros((newMask.shape[0],newMask.shape[1],3),dtype=np.float32);
	zero_frame[:,:,2] = np.float32(oldMask*255);
	out_frame1 = cv2.addWeighted(np.float32(frame),0.6,zero_frame,0.4,0.0);
	out_frame2 = frame
	for (idx,_center) in enumerate(window_centers):
		left = _center - window_size[idx]/2;
		left[0] = min(left[0],shape[1]-window_size[idx][0]); left[0] = max(left[0],0)
		left[1] = min(left[1],shape[0]-window_size[idx][1]); left[1] = max(left[1],0)
		rect = np.array([left[0],left[1],window_size[idx][0],window_size[idx][1]],dtype=np.float32);
		window = (rect[0],rect[1],rect[0]+rect[2],rect[1]+rect[3]);
		cv2.rectangle(out_frame2, (window[0],window[1]), (window[2],window[3]), tuple(RANDOM_COLORS[idx,:]))
		cv2.putText(out_frame2,str(idx), (int((window[0]+window[2])/2),int((window[1]+window[3])/2)), cv2.FONT_HERSHEY_DUPLEX, 0.25, 255)
	zero_frame[:,:,2] = np.float32(newMask*255);
	out_frame2 = cv2.addWeighted(np.float32(frame),0.6,zero_frame,0.4,0.0);
	out_frame = np.hstack((out_frame1,out_frame2))
	vidwriter.write(np.uint8(out_frame))
"""

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
			frameNewMasks.extend(newMasks);
		frame_idx += num_blocks
	time_taken = time.time()-start;	
	#print "Smoothning Masks ... [DONE] in ",time_taken," seconds"
	return frameNewMasks;

def perform_tracking(vidreader,smoothMasks,num_blocks,num_prev_frames,num_objects=1):
	start = time.time(); 
	tracker = tracker_instance(TrackerMethods.MIXTURE_BASED); 	tracker.numMixtures = num_objects;
	prev_rects = [np.array([vidreader.width/4,vidreader.height/4,\
						vidreader.width/2,vidreader.height/2],\
						dtype=np.float32) for idx in range(num_objects)];
	vidreader.__reset__();	N = vidreader.frames;
	skip_frames = num_prev_frames + num_blocks/2;
	frame_idx = 0; numFrames= len(smoothMasks);
	frame_all_window_center = []; window_size = np.zeros((num_objects,2));
	while(frame_idx < numFrames):
		#print 'Tracking ... {0}%\r'.format((frame_idx*100/N)),
		(cnt,frames) = vidreader.read(frame_idx+skip_frames,num_blocks);
		if cnt > 1:
			window_frames = tracker.track_object(frames,smoothMasks[frame_idx:frame_idx+cnt]);
			s = frame_idx; e = s + cnt; 
			for window_frame in window_frames:
				all_window_center = [];
				for (idx,_window) in enumerate(window_frame):
					(window,lbl) = _window
					rect = np.array([window[0],window[1],window[2]-window[0],window[3]-window[1]],dtype=np.float32);
					cv2.accumulateWeighted(rect,prev_rects[idx],0.1,None);
					window_center = prev_rects[idx][:2]+(prev_rects[idx][2:]/2); 
					window_size[idx] = np.maximum(window_size[idx],prev_rects[idx][2:]);
					all_window_center.extend([window_center]);
				frame_all_window_center.extend([all_window_center]);
		frame_idx += num_blocks;
	time_taken = time.time()-start;	
	#print "Tracking .... [DONE] in ",time_taken," seconds"
	return frame_all_window_center,window_size
	
def crop_frame(frame,window_center,window_size,ac_shape,rsz_shape):
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
	return np.uint8(zero_frame);

def write_video(write_path,vidreader,smoothMasks,window_centers,window_size,num_blocks,
					num_prev_frames,rsz_shape=[80,60]):
	create_folder_structure_if_not_exists(write_path);
	shape = [vidreader.width, vidreader.height];
	vidwriter = VideoWriter(write_path,rsz_shape[0],rsz_shape[1]);
	vidwriter.build();	vidreader.__reset__();	
	skip_frames = num_prev_frames + num_blocks/2; vidreader.skip_frames(skip_frames);
	frame_idx = 0; numFrames= len(smoothMasks); 	
	while(frame_idx < numFrames):
		frame = vidreader.read_next();
		if frame is None:
			break;
		frame = crop_frame(frame,window_centers[frame_idx][0],window_size,shape,rsz_shape);
		vidwriter.write(np.uint8(frame))
	vidwriter.close();	

def open_write_header(extract_path):
	create_folder_structure_if_not_exists(extract_path);
	_file = open(extract_path,'w');
	return _file;



def extract_feats(extract_path,vidreader,smoothMasks,window_centers,window_size,num_blocks,num_prev_frames,
						rsz_shape=[80,60],write_gray=False,bgImpl=None,window=5,overlap=2):
	_file = open_write_header(extract_path);
	shape = [vidreader.width, vidreader.height]
	skip_frames = num_prev_frames + num_blocks/2;
	frame_idx = 0; numFrames= len(smoothMasks); 
	_frame = np.zeros((window_size[1],window_size[0],3))
	_ratio = rsz_shape[0]/float(window_size[0]); _frame = cv2.resize(_frame,None,fx=_ratio, fy=_ratio)
	_shape = _frame.shape; 	x_l =  rsz_shape[0]/2 - _shape[1]/2; y_l =  rsz_shape[1]/2 - _shape[0]/2;
	zero_frame = np.zeros((rsz_shape[1],rsz_shape[0],3));
	while(frame_idx < numFrames):
		if not bgImpl is None:
			(cnt,frames) = vidreader.read(frame_idx+skip_frames,window+1);	
			prev_frame = crop_frame(frames[0],window_centers[frame_idx][0],window_size,shape,rsz_shape); 
			cnt -= 1; frame_idx += 1; frames =frames[1:]
		else:
			(cnt,frames) = vidreader.read(frame_idx+skip_frames,window);
				
		if cnt==window:
			w_frames = [];
			for frame in frames:
				frame = crop_frame(frame,window_centers[frame_idx][0],window_size,shape,rsz_shape);
				if write_gray:
					feats = GRAY(frame);
				else:
					feats = frame;
				if not bgImpl is None:
					bg_variation = bgImpl.process(frame,[prev_frame]);
					bgmask = bgImpl.threshold_mask(bg_variation)*255;
					prev_frame = frame; feats = np.dstack((feats,bgmask));
				w_frames.extend(feats.flatten())	
			w_frames = np.array(w_frames)/float(255);
			w_frames.tofile(_file,' ','%0.3f'); _file.write(os.linesep)
		frame_idx += window-overlap;
	_file.close();
	time_taken = time.time()-start;	
	
def process(vidreader,sal,bg,smoothner,num_prev_frames=3 ,num_blocks=4, write_path=None,
				extract_path=None,write_gray=False,write_bgsub=False,window=5,overlap=2,
				rsz_shape=[80,60],threaded=False):
	fgMasks,bgMasks = compute_fgbg_masks(vidreader,sal,bg,num_prev_frames,threaded);
	smoothMasks = perform_smoothing(vidreader,smoothner,fgMasks,bgMasks,num_blocks,num_prev_frames);
	(window_centers,window_size) = perform_tracking(vidreader,smoothMasks,num_blocks,num_prev_frames)
	if write_bgsub:
		bgImpl = bg;
	else:
		bgImpl = None;
		
	if not write_path is None:
		write_video(write_path,vidreader,smoothMasks,window_centers,np.uint8(window_size[0]),
					num_blocks,num_prev_frames,rsz_shape)
	if not extract_path is None:
		extract_feats(extract_path,vidreader,smoothMasks,window_centers,np.uint8(window_size[0]),
					num_blocks,num_prev_frames,rsz_shape,write_gray,bgImpl,window,overlap);
		
	"""
	frame_idx = 0; numFrames= len(smoothMasks);
	vidwriter = VideoWriter(out_path,2*vidreader.width,vidreader.height)
	vidwriter.build();
	vidreader.__reset__();	N = vidreader.frames; skip_frames = num_prev_frames + num_blocks/2;
	vidreader.skip_frames(skip_frames);
	while(frame_idx < numFrames):
		frame = vidreader.read_next();
		if frame is None:
			break;
		write_block(vidwriter,frame,smoothMasks[frame_idx],fgMasks[frame_idx + num_blocks/2],\
						window_centers[frame_idx],window_size);
		frame_idx += 1
	vidreader.close();
	"""
	
	


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
	parser.add_argument("--num_prev_frames",nargs='?',help = "num prev frames default-3",default=2,type=int);
	parser.add_argument("--num_blocks",nargs='?',help = "num blocks default-6",default=6,type=int);
	parser.add_argument("--rsz_shape",nargs='+',help = "overlap default-[80 60]",default=[80,60],type=int);
	parser.add_argument("--write_gray",nargs='?',help = "Write gray value  default-False",default=False,type=bool);
	parser.add_argument("--write_bgsub",nargs='?',help = "Write bgsub default-False",default=False,type=bool);
	parser.add_argument("--window",nargs='?',help = "window size default-5",default=5,type=int);
	parser.add_argument("--overlap",nargs='?',help = "overlap default-2",default=2,type=int);
	
	args = parser.parse_args()
	inp = args.input;	out = args.output
	vidreader = VideoReader(inp)
	sal = sal_instance(args.sal,SaliencyProps())	
	bg = get_instance(args.bg);
	smoothner = smooth_instance(feats,args.smoothner);
	start = time.time();
	process(vidreader,sal,bg,smoothner,args.num_prev_frames,args.num_blocks,args.write,args.extract,
				args.write_gray,args.write_bgsub,args.window,args.overlap,args.rsz_shape);
	print "Event Tracker ...",inp,"[DONE] in",(time.time()-start),"seconds";
