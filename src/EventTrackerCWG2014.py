import time ,os, os.path
import cv2,numpy as np;
from skimage.morphology import remove_small_objects
from skimage.measure import label
from scipy.ndimage.morphology import binary_fill_holes
from multiprocessing.pool import ThreadPool
from collections import deque

from io_module.frame_folder_reader import FrameFolderReader
from saliency import SaliencyProps,SaliencyMethods,get_instance as sal_instance
from bg_sub import get_instance,BGMethods
from smoothing import get_instance as smooth_instance,SmoothMethods
from features.pixellete import allFeats as feats
from utils import normalize,create_folder_structure_if_not_exists,DummyTask	
from utils.stats import comparator,updateConfusion,getNewConfusion,Stats

KERNEL = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3));

def getMask(frame):
	frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY);
	return np.where(frame==255,1,0);

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
		EL_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
		_mask = cv2.dilate(np.uint8(_mask),EL_KERNEL,iterations = 2)
		new_masks.extend([_mask])
	return new_masks;

def compute_fgbg_masks(vidreader,gtreader,sal,bg,num_prev_frames=1,skip_frames=0,
							last_frame=1000,threaded=False,num_blocks=10,):
	start = time.time();
	def compute_mask(frameIdx,frames,gtframes):
		num_frames = len(frames);	_idx = num_prev_frames
		fgMasks = []; bgMasks = []; gtMasks = [];
		while _idx<num_frames:
			prev_frames = frames[_idx-num_prev_frames:_idx]
			gtmask = getMask(gtframes[_idx-num_prev_frames]);
			bg_variation = bg.process(frames[_idx],prev_frames);
			saliency = sal.process(frames[_idx]);	
			_idx += 1;	saliency_prob =  normalize(7*bg_variation  + 3*saliency);
			_,fgmask = cv2.threshold(saliency_prob ,0.5,1,cv2.THRESH_BINARY)
			fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, KERNEL)
			EL_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
			fgmask = cv2.dilate(np.uint8(fgmask),EL_KERNEL,iterations = 2)
			fgmask = cv2.medianBlur(fgmask,3)
			fgmask = binary_fill_holes(fgmask)
			fgmask = cv2.dilate(np.uint8(fgmask),EL_KERNEL,iterations = 2)
			fgmask = cv2.morphologyEx(np.uint8(fgmask), cv2.MORPH_CLOSE,KERNEL)	
			_,bgmask = cv2.threshold(bg_variation,0.1,1,cv2.THRESH_BINARY_INV)
			zero_frame= np.zeros(frames[0].shape,dtype = np.uint8);
			zero_frame[:,:,2]=fgmask*255;
			zero_frame[:,:,1]=bgmask*255;
			_prob = cv2.cvtColor(saliency_prob*255,cv2.COLOR_GRAY2RGB);
			fgMasks.extend([fgmask]); bgMasks.extend([bgmask]); gtMasks.extend([gtmask])
		return (frameIdx,fgMasks,bgMasks,gtMasks);
	frameFgMasks = []; frameBgMasks = []; frameGTMasks = [];
	frameIdx = skip_frames+num_prev_frames+num_blocks; N = vidreader.frames;
	threadn = cv2.getNumberOfCPUs();	pool = ThreadPool(processes = threadn/2);
	pending = deque();	N = min(vidreader.frames,last_frame);
	while True:
		while len(pending) > 0 and pending[0].ready():
			task = pending.popleft()
			idx,fgmask,bgmask,gtmask  = task.get()
			frameFgMasks.extend(fgmask); frameBgMasks.extend(bgmask); frameGTMasks.extend(gtmask);
			print 'Computing Mask ... {0}%\r'.format((idx*100/N)),
		if len(pending) < threadn and frameIdx-num_prev_frames-num_blocks < N:
			(cnt,frames) = vidreader.read(frameIdx-num_prev_frames-num_blocks,num_prev_frames+num_blocks);
			(_,gtframes) = gtreader.read(frameIdx-num_blocks,num_blocks);
			if cnt >= num_prev_frames:
				if threaded:
					task = pool.apply_async(compute_mask,[min(frameIdx,N),frames,gtframes]);
				else:
					task = DummyTask(compute_mask(min(frameIdx,N),frames,gtframes));
				pending.append(task)
			frameIdx += num_blocks;	
		if len(pending) == 0:
			break;
	time_taken = time.time()-start;	
	print "Computing Mask ... [DONE] in ",time_taken," seconds"
	return (frameFgMasks,frameBgMasks,frameGTMasks)

def perform_localization(vidreader,roi_mask,smoothner,fgMasks,bgMasks,gtMasks,
							num_blocks=10,skip_frames = 0,last_frame=1000):
	confusion = getNewConfusion();
	start = time.time();	frameNewMasks = []; 
	frame_idx = skip_frames + 2*num_blocks;  N = min(vidreader.frames,last_frame);
	status = 'Localizing Video ' + vidreader.foldername  +' ... {0}\r'
	out_dir=os.path.dirname(vidreader.foldername)+os.sep+'output'+os.sep
	create_folder_structure_if_not_exists(out_dir);
	while(frame_idx +num_blocks  < N):
		print status.format((str(frame_idx*100/N)+'%')),
		(num_frames,frames) = vidreader.read(frame_idx-2*num_blocks,2*num_blocks);
		if num_frames > num_blocks:
			start_idx = num_blocks/2; end_idx = min(3*num_blocks/2,num_frames);
			s = frame_idx-2*num_blocks-skip_frames; e = s + num_frames;
			newMasks = smoothner.process(frames,fgMasks[s:e],bgMasks[s:e],range(start_idx,end_idx));
			#newMasks = __morphologicalOps__(newMasks);
			groundTruthMasks = gtMasks[s+num_blocks/2:s+num_blocks/2+newMasks.__len__()];	
			_frames = frames[num_blocks/2:num_blocks/2+newMasks.__len__()];
			tmp_idx= 0;
			for (gt,my) in zip(groundTruthMasks,newMasks):
				zero = np.zeros(gt.shape[:2],dtype=np.uint8)
				my_mask = np.uint8(np.dstack((zero,zero,my*255)))
				out = cv2.addWeighted(_frames[tmp_idx],0.6,my_mask,0.4,0);
				path = out_dir+os.sep+'sal_'+str(frame_idx-2*num_blocks-skip_frames+tmp_idx)+'.png'
				cv2.imwrite(path,out)
				gt_mask = np.uint8(np.dstack((zero,zero,gt*255)));
				out = cv2.addWeighted(_frames[tmp_idx],0.6,gt_mask,0.4,0);
				path = out_dir+os.sep+'gt_'+str(frame_idx-2*num_blocks-skip_frames+tmp_idx)+'.png'
				cv2.imwrite(path,out)
				frame_confusion=comparator(gt,my,roi_mask);
				updateConfusion(confusion,frame_confusion);
				tmp_idx += 1
			
		frame_idx += num_blocks
	time_taken = time.time()-start;	
	print status.format('[Done] in '),time_taken," seconds"
	return confusion;
	
def process(vidreader,sal,bg,smoothner,num_prev_frames ,num_blocks,
				gtreader,roi_mask,startFrame=0,endFrame=-1):
	startFrame = max(startFrame-(num_blocks/2)-num_prev_frames,1);
	endFrame = min(endFrame+num_blocks/2,vidreader.frames)
	fgMasks,bgMasks,gtMasks = compute_fgbg_masks(vidreader,gtreader,sal,bg,num_prev_frames,
					startFrame,endFrame);
	confusion = perform_localization(vidreader,roi_mask,smoothner,fgMasks,bgMasks,
					gtMasks,num_blocks,startFrame+num_prev_frames,endFrame);
	vidreader.close();
	gtreader.close();
	return confusion;

def compareWithGroundtruth(videoPath,sal,bg,smoothner,num_prev_frames,num_blocks,resize=(160,120)):
	inputPath = os.path.join(videoPath, 'input')
	vidreader = FrameFolderReader(inputPath,resize)
	groundtruthPath = os.path.join(videoPath, 'groundtruth')
	gtreader = FrameFolderReader(groundtruthPath,resize)
	ROIPath = os.path.join(videoPath, 'ROI.bmp');
	#roi_img = cv2.resize(cv2.imread(ROIPath), (0,0),fx=resize[0],fy=resize[1]);
	roi_img = cv2.resize(cv2.imread(ROIPath),resize);
	roi_mask = getMask(roi_img);
	temporalROI = os.path.join(videoPath, 'temporalROI.txt');
	with open(temporalROI) as f:
		line = f.readline();
		indices  = [int(strnum) for strnum in line.split()];
	confusion = process(vidreader,sal,bg,smoothner,num_prev_frames,num_blocks,
							gtreader,roi_mask,startFrame=indices[0],endFrame =indices[1]);
	return confusion

def getDirectories(path):
	"""Return a list of directories name on the specifed path"""
	return [file for file in os.listdir(path)
			if os.path.isdir(os.path.join(path, file))]
	
def processFolder(datasetPath,sal,bg,smoothner,num_prev_frames,num_blocks):
	stats = Stats(datasetPath) 
	for category in getDirectories(datasetPath):
		stats.addCategories(category)
		categoryPath = os.path.join(datasetPath, category)
		for video in getDirectories(categoryPath):
			videoPath = os.path.join(categoryPath, video)
			confusionMatrix = compareWithGroundtruth(videoPath,sal,bg,smoothner,
					num_prev_frames,num_blocks)
			stats.update(category, video, confusionMatrix)
		stats.writeCategoryResult(category)
	stats.writeOverallResults()

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description='Event tracking algorithm using saliency and bg subtraction')
	parser.add_argument("input",nargs='?',help = "directory chanage detection dataset",default="");
	parser.add_argument("output",nargs='?',help = "output path",default="test_results/");
	parser.add_argument("--bg",nargs='?',help = "background subtraction method 1-FD, 2-ES, 3-MG default-2",default=1,type=int);
	parser.add_argument("--sal",nargs='?',help = "saliency method 1-CF, 2-CA, 3-RC, 4-SD default-3",default=3,type=int);
	parser.add_argument("--smoothner",nargs='?',help = "smoothing method 1-Eigen, 2-GMM, 3-SSL,  default-2",default=2,type=int);
	parser.add_argument("--num_prev_frames",nargs='?',help = "num prev frames default-3",default=3,type=int);
	parser.add_argument("--num_blocks",nargs='?',help = "num blocks default-6",default=8,type=int);
	args = parser.parse_args()
	sal = sal_instance(args.sal,SaliencyProps())	
	bg = get_instance(args.bg);
	smoothner = smooth_instance(feats,args.smoothner);
	processFolder(args.input,sal,bg,smoothner,args.num_prev_frames,args.num_blocks);
		
		
