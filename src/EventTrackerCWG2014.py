import time ,os, os.path
import cv2,numpy as np;
from io_module.frame_folder_reader import FrameFolderReader
from io_module.video_writer import VideoWriter
from saliency import SaliencyProps,SaliencyMethods,get_instance as sal_instance
from smoothing import get_instance as smooth_instance
from features.pixellete import allFeats as feats
from bg_sub import get_instance,BGMethods
from utils import normalize
from skimage.morphology import remove_small_objects
from skimage.measure import label
from scipy.ndimage.morphology import binary_fill_holes
from utils.cwg import comparator,updateConfusion,getNewConfusion
from utils.cwg.Stats import Stats	
KERNEL = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3));

def getMask(frame):
	frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY);
	return np.where(frame==255,1,0);
	
def process_next_block(gtreader,sal_rc,bg_es,blockSize=4):
	frameCnt = 0;	blocks = []; gt_mask_blocks = []
	bg_mask_blocks = [];	fg_mask_blocks = [];
	while(frameCnt < blockSize):
		mask_es = bg_es.process();
		if not bg_es.isFinish():
			frameCnt += 1
			gt_mask = getMask(gtreader.read_next());
			blocks.append(np.array(bg_es.cur_frame,dtype=np.uint8))
			sal_rc.process(bg_es.cur_frame);
			saliency_prob =  bg_es.variation  + sal_rc.saliency;
			saliency_prob = normalize(saliency_prob)
			_,fgmask = cv2.threshold(saliency_prob,0.5,1,cv2.THRESH_BINARY)
			_,bgmask = cv2.threshold(bg_es.variation,0.1,1,cv2.THRESH_BINARY_INV)
			bg_mask_blocks.append(bgmask);
			fg_mask_blocks.append(fgmask);
			gt_mask_blocks.append(gt_mask);
		else:
			break;
	return (frameCnt,blocks,fg_mask_blocks,bg_mask_blocks,gt_mask_blocks);

def __morphologicalOps__(masks,staticMasks):
	new_masks = [];
	for (mask,_staticMask) in zip(masks,staticMasks):
		#_mask = binary_fill_holes(mask)
		#_mask = np.uint8(mask * (1-_staticMask));
		_mask = mask
		_mask = cv2.medianBlur(_mask,3)
		_mask = cv2.morphologyEx(np.uint8(_mask), cv2.MORPH_OPEN, KERNEL)
		_mask = cv2.medianBlur(_mask,3)
		_mask = cv2.morphologyEx(_mask, cv2.MORPH_CLOSE,KERNEL)
		_mask = binary_fill_holes(_mask)
		_mask = remove_small_objects(_mask,min_size=128,connectivity=2)
		new_masks.extend([_mask])
	return np.uint8(new_masks);

"""		
def write_block(vidwriter,frames,newMasks,oldMasks,staticMasks):
	zero_frame  = np.zeros((newMasks[0].shape[0],newMasks[0].shape[1],3),dtype=np.float32);
	zero = np.zeros((newMasks[0].shape[0],newMasks[0].shape[1]),dtype=np.float32);
	for (frame,newMask,oldMask,staticMask) in zip(frames,newMasks,oldMasks,staticMasks):
		#newMask = cv2.cvtColor(newMask*255,cv2.COLOR_GRAY2RGB);
		zero_frame[:,:,2] = oldMask*255;
		zero_frame[:,:,1] = staticMask*255;
		out_frame1 = cv2.addWeighted(np.float32(frame),0.6,zero_frame,0.4,0.0);
		zero_frame[:,:,2] = newMask*255;
		zero_frame[:,:,1] = zero
		out_frame2 = cv2.addWeighted(np.float32(frame),0.6,zero_frame,0.4,0.0);
		out_frame = np.hstack((out_frame1,out_frame2))
		vidwriter.write(np.uint8(out_frame))

def process(vidreader,out_path,batch=4):
	vidwriter.build();
	vidreader.skip_frames(790)
	sal_rc = sal_instance(SaliencyMethods.REGION_CONTRAST,SaliencyProps())	
	bg_es = get_instance(BGMethods.FRAME_DIFFERENCING,vidreader.read_next)
	tracker = tracker_instance(0);
	bg_es.setShape((vidreader.height,vidreader.width))
	smoothner = smooth_instance(feats,1);	
	frame_idx = 0; N = vidreader.num_remaining_frames();
	batchsize = batch * 2
	_frames = []; _fgmasks = []; _bgmasks = [];
	while(vidreader.has_next()):
		print 'Proceesing Video... {0}%\r'.format((frame_idx*100/N))
		(cnt,frames,fgmasks,bgmasks) = process_next_block(sal_rc,bg_es,batchsize);
		if cnt > 0:
			frame_idx += cnt;
			_frames.extend(frames); _fgmasks.extend(fgmasks); _bgmasks.extend(bgmasks); 
			newMasks = smoothner.process(_frames,_fgmasks,_bgmasks,range(batch/2,3*batch/2));
			frames = _frames[batch/2:batch/2+newMasks.__len__()]
			staticMasks = _bgmasks[batch/2:batch/2+newMasks.__len__()];
			oldMasks = _fgmasks[batch/2:batch/2+newMasks.__len__()]
			newMasks = __morphologicalOps__(newMasks,staticMasks);
			write_block(vidwriter,frames,newMasks,oldMasks,staticMasks);
			_frames = _frames[batch:];_fgmasks = _fgmasks[batch:];  _bgmasks = _bgmasks[batch:]; 
		else:
			break;
		batchsize = batch;
	vidreader.close();
	vidwriter.close()
"""
def process(vidreader,gtreader,roi_mask,startFrame=0,endFrame=-1,batch=4):
	confusion = getNewConfusion();
	startFrame = max(startFrame-(batch/2),1);
	endFrame = min(endFrame+(batch/2),vidreader.frames)
	vidreader.skip_frames(startFrame-1)
	gtreader.skip_frames(startFrame)
	sal_rc = sal_instance(SaliencyMethods.REGION_CONTRAST,SaliencyProps())	
	bg_es = get_instance(BGMethods.FRAME_DIFFERENCING,vidreader.read_next)
	bg_es.setShape((vidreader.height,vidreader.width))
	smoothner = smooth_instance(feats,1);	
	frame_idx = 0; N = endFrame-gtreader.read_frames+1; batchsize = batch * 2
	_frames = []; _fgmasks = []; _bgmasks = []; _gtmasks = [];
	status = 'Proceesing Video ' + vidreader.foldername  +' ... {0}\r'
	while(vidreader.has_next()):
		print status.format((str(frame_idx*100/N)+'%')),
		(cnt,frames,fgmasks,bgmasks,gtmasks) = process_next_block(gtreader,sal_rc,bg_es,batchsize);
		if cnt > 0:
			frame_idx += cnt;
			_frames.extend(frames); _fgmasks.extend(fgmasks); 
			_bgmasks.extend(bgmasks); _gtmasks.extend(gtmasks);
			newMasks = smoothner.process(_frames,_fgmasks,_bgmasks,range(batch/2,3*batch/2));
			frames = _frames[batch/2:batch/2+newMasks.__len__()]
			staticMasks = _bgmasks[batch/2:batch/2+newMasks.__len__()];
			groundTruthMasks = _gtmasks[batch/2:batch/2+newMasks.__len__()];
			newMasks = __morphologicalOps__(newMasks,staticMasks);
			for (gt,my) in zip(groundTruthMasks,newMasks):
				updateConfusion(confusion,comparator(gt,my,roi_mask));
			_frames = _frames[batch:];	_fgmasks = _fgmasks[batch:];  
			_bgmasks = _bgmasks[batch:]; _gtmasks = _gtmasks[batch:]
		else:
			break;
		if vidreader.read_frames >= endFrame:
			break;
		batchsize = batch;
	print status.format('[Done]')
	vidreader.close();
	gtreader.close();
	return confusion;

def compareWithGroundtruth(videoPath,resize=(160,120)):
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
	confusion = process(vidreader,gtreader,roi_mask,startFrame=indices[0],endFrame =indices[1]);
	return confusion

def getDirectories(path):
	"""Return a list of directories name on the specifed path"""
	return [file for file in os.listdir(path)
			if os.path.isdir(os.path.join(path, file))]
	
def processFolder(datasetPath):
	stats = Stats(datasetPath) 
	for category in getDirectories(datasetPath):
		stats.addCategories(category)
		categoryPath = os.path.join(datasetPath, category)
		for video in getDirectories(categoryPath):
			videoPath = os.path.join(categoryPath, video)
			confusionMatrix = compareWithGroundtruth(videoPath)
			stats.update(category, video, confusionMatrix)
		stats.writeCategoryResult(category)
	stats.writeOverallResults()

if __name__ == "__main__":
	import sys;	
	if sys.argv.__len__()<=1:
		print "dataset path not provided........."
	else :
		processFolder(sys.argv[1]);
		
		
