import cv2,sys,os
import numpy as np
from utils import mkdirs
from bg_sub import get_instance,BGMethods
from io_module.video_reader import VideoReader
from io_module.video_writer import VideoWriter
from utils import DummyTask
import time
from multiprocessing.pool import ThreadPool
from collections import deque


def bgsub_process(bgsubImpl,frames,idx):
	N = frames.__len__();
	shape = frames[0].shape
	diff = bgsubImpl.process(frames[N-1],frames[:N-1]);
	mask = bgsubImpl.threshold_mask(diff)
	zero_frame  = np.zeros(shape,dtype=np.float32);
	zero_frame[:,:,2] = mask*255;
	out_frame = cv2.addWeighted(np.float32(frames[N-1]),0.6,zero_frame,0.4,0.0)
	return np.uint8(out_frame),idx;
	
def process_video(bgsubImpl,vidreader,vidwriter,num_blocks=4, threaded = False):
	vidwriter.build();
	threadn = cv2.getNumberOfCPUs();	pool = ThreadPool(processes = threadn);
	pending = deque();	N = vidreader.frames; frameIdx = num_blocks;
	while True:
		while len(pending) > 0 and pending[0].ready():
			task = pending.popleft()
			frame, idx = task.get()
			vidwriter.write(frame);
			print 'Proceesing ... {0}%\r'.format((idx*100/N)),	
		if len(pending) < threadn and frameIdx < N:
			(cnt,frames) = vidreader.read(frameIdx-num_blocks,num_blocks);
			if cnt == num_blocks:
				if threaded:
					task = pool.apply_async(bgsub_process,[bgsubImpl,frames,frameIdx]);
				else:
					task = DummyTask(bgsub_process(bgsubImpl,frames,frameIdx));
				pending.append(task)
			frameIdx += 1;	
		if len(pending) == 0:
			break;


def test_bgsub_fd(inp):
	vidreader = VideoReader(inp)
	vidwriter = VideoWriter("test_results/bg_sub_fd.avi",vidreader.width,vidreader.height);
	bgsub = get_instance(BGMethods.FRAME_DIFFERENCING);
	start = time.time();
	process_video(bgsub,vidreader,vidwriter,num_blocks=2);
	time_taken = time.time() - start;
	print "Tested Background Subtraction (Frame Differencing)...    [DONE] in " + str(time_taken) +" seconds"


def test_bgsub_es(inp):
	vidreader = VideoReader(inp)
	vidwriter = VideoWriter("test_results/bg_sub_es.avi",vidreader.width,vidreader.height);
	bgsub = get_instance(BGMethods.EIGEN_SUBSTRACTION);
	start = time.time();
	process_video(bgsub,vidreader,vidwriter);
	time_taken = time.time() - start;
	print "Tested Background Subtraction (Eigen Subtraction)...    [DONE] in " + str(time_taken) +" seconds"
	
def test_bgsub_mog(inp):
	vidreader = VideoReader(inp)
	vidwriter = VideoWriter("test_results/bg_sub_mog.avi",vidreader.width,vidreader.height);
	bgsub = get_instance(BGMethods.MOG_SUBSTRACTION);
	start = time.time();
	process_video(bgsub,vidreader,vidwriter);
	time_taken = time.time() - start;
	print "Tested Background Subtraction (Mixture of Gaussian)...    [DONE] in " + str(time_taken) +" seconds"
	
def test(vid_path="../examples/videos/sample_video1.avi"):
	mkdirs("test_results")
	start = time.time();
	test_bgsub_fd(vid_path);
	test_bgsub_es(vid_path);
	test_bgsub_mog(vid_path);
	time_taken = time.time() - start;
	print "Tested all Background Subtraction methods ...   [DONE] in " + str(time_taken) +" seconds"
