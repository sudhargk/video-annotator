import cv2,sys,os
import cv2
import numpy as np
from utils import mkdirs
from bg_sub import get_instance,BGMethods
from io_module.video_reader import VideoReader
from io_module.video_writer import VideoWriter
from sklearn.cluster import MeanShift,DBSCAN, estimate_bandwidth

#Default values
MIN_FRAMES_COUNT = 1000
MIN_POINTS_TO_CLUSTER = 10
MAX_CLUSTERS = 100
SKIP_FRAMES = 0
ALPHA = 0.35
DO_CLUSTERING = False
DO_CROPPING = False
DO_LABELLING = True
DO_BOUNDING_BOX = False	
CROP_D = (80,160)
	
def process_video(bgsubImpl,vidreader,vidwriter):
	random_colors = np.random.randint(256, size=(MAX_CLUSTERS, 3))
	#Clustering models
	if DO_CLUSTERING:
		#model = MeanShift(bandwidth=None, bin_seeding=True)
		model = DBSCAN(eps=5, min_samples=35)
	
	vidreader.skip_frames(SKIP_FRAMES);
	bgsubImpl.setShape((vidreader.height,vidreader.width))
	
	if DO_CROPPING:
		vidwriter.shape = CROP_D
	
	vidwriter.build();
	N = min(vidreader.num_remaining_frames(),MIN_FRAMES_COUNT)

	prv_mean = None; frame_idx = 0
	while(True):
		frame_idx += 1;
		mask_frame = bgsubImpl.process()
		mask_frame = cv2.medianBlur(mask_frame,5)
		mask_frame = cv2.medianBlur(mask_frame,3)
		print 'Proceesing ... {0}%\r'.format((frame_idx*100/N)),
		if bgsubImpl.isFinish() or frame_idx>MIN_FRAMES_COUNT:
			break;
			
		img = bgsubImpl.cur_frame.copy();

		if DO_CLUSTERING :		#clustering logic
			points = np.where(mask_frame==1)
			points = np.column_stack(points)
			points_len = points.shape[0]
			if points_len > MIN_POINTS_TO_CLUSTER:
				model.fit(points);
				for idx,lbl in zip(range(points_len),model.labels_):
					mask_frame[points[idx][0]][points[idx][1]]=lbl+1;
		
		if DO_LABELLING:		#labelling logic
			for lbl,val in enumerate(np.unique(mask_frame)):
				if lbl == 0:
					continue;
				if lbl >= MAX_CLUSTERS:
					break;
				color = tuple(random_colors[lbl])
				for point_x,point_y in np.column_stack(np.where(mask_frame==val)):
					cv2.circle(img,(point_y,point_x), 2,color, 1)
		
		if DO_BOUNDING_BOX:		#bounding box logic
			points =  np.column_stack(np.where(mask_frame==1))
			if points.shape[0] > 0:
				mean = points.mean(axis=0)
				if not prv_mean is None:
					mean = ALPHA*mean + (1-ALPHA)*prv_mean
				(y,x)=np.int32(mean);
				if x-w_crop/2>0 and x+w_crop/2<w and y-h_crop/2>0 and y+h_crop/2<h :
					if DO_CROPPING:
						img = img[y-h_crop/2:y+h_crop/2,x-w_crop/2:x+w_crop/2];
					else:
						cv2.rectangle(img,(x-w_crop/2,y-h_crop/2),(x+w_crop/2,y+h_crop/2),(255,0,0),2);
					vidwriter.write(img);
				prv_mean = np.int32(mean);
		else:
			vidwriter.write(img);
	vidreader.close();
	vidwriter.close()

def test_bgsub_fd(inp):
	vidreader = VideoReader(inp)
	vidwriter = VideoWriter("test_results/bg_sub_fd.avi",vidreader.width,vidreader.height);
	bgsub = get_instance(BGMethods.FRAME_DIFFERENCING,vidreader.read_next);
	process_video(bgsub,vidreader,vidwriter);
	print "Tested Background Subtraction (Frame Differencing)...    [DONE]"

def test_bgsub_ma(inp):
	vidreader = VideoReader(inp)
	vidwriter = VideoWriter("test_results/bg_sub_ma.avi",vidreader.width,vidreader.height);
	bgsub = get_instance(BGMethods.MOVING_AVERAGE,vidreader.read_next);
	process_video(bgsub,vidreader,vidwriter);
	print "Tested Background Subtraction (Moving Average)...    [DONE]"

def test_bgsub_es(inp):
	vidreader = VideoReader(inp)
	vidwriter = VideoWriter("test_results/bg_sub_es.avi",vidreader.width,vidreader.height);
	bgsub = get_instance(BGMethods.EIGEN_SUBSTRACTION,vidreader.read_next);
	process_video(bgsub,vidreader,vidwriter);
	print "Tested Background Subtraction (Eigen Subtraction)...    [DONE]"
	
def test_bgsub_mog(inp):
	vidreader = VideoReader(inp)
	vidwriter = VideoWriter("test_results/bg_sub_mog.avi",vidreader.width,vidreader.height);
	bgsub = get_instance(BGMethods.GMG_SUBSTRACTION,vidreader.read_next);
	process_video(bgsub,vidreader,vidwriter);
	print "Tested Background Subtraction (Mixture of Gaussian)...    [DONE]"
	
def test(vid_path="../examples/videos/sample_video1.avi"):
	mkdirs("test_results")
	test_bgsub_fd(vid_path);
	test_bgsub_ma(vid_path);
	test_bgsub_es(vid_path);
	test_bgsub_mog(vid_path);
	print "Tested all Background Subtraction methods ...   [DONE]"
