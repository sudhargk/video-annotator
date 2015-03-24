import time 
import cv2,numpy as np;
from io_module.video_reader import VideoReader
from io_module.video_writer import VideoWriter
from saliency import SaliencyProps,SaliencyMethods,get_instance as sal_instance
from smoothing import get_instance as smooth_instance
from features.pixellete import allFeats as feats
from bg_sub import get_instance,BGMethods
from utils import normalize
from skimage.morphology import remove_small_objects
from skimage.measure import label
from tracker import get_instance as tracker_instance
from scipy.ndimage.morphology import binary_fill_holes

MAX_COLORS = 100
RANDOM_COLORS = np.random.randint(256, size=(MAX_COLORS, 3))
KERNEL = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3));
bgdModel = np.zeros((1,65),np.float64); fgdModel = np.zeros((1,65),np.float64)


def process_next_block(sal_rc,bg_es,blockSize=4):
	frameCnt = 0;	blocks = []; bg_mask_blocks = [];	fg_mask_blocks = [];
	while(frameCnt < blockSize):
		mask_es = bg_es.process();
		if not bg_es.isFinish():
			frameCnt += 1
			blocks.append(np.array(bg_es.cur_frame,dtype=np.uint8))
			sal_rc.process(bg_es.cur_frame);
			saliency_prob =  bg_es.variation  + sal_rc.saliency;
			saliency_prob = normalize(saliency_prob)
			"""
			#mask_cut = np.zeros(bg_es.cur_frame.shape[:2],np.uint8)
			#mask_cut[saliency_prob >= 0.2] = 2
			#mask_cut[saliency_prob >= 0.6] = 3
			#mask_cut[saliency_prob >= 0.8] = 1
			#print np.sum(mask_cut==0),np.sum(mask_cut==1),np.sum(mask_cut==2),np.sum(mask_cut==3)
			#cv2.grabCut(bg_es.cur_frame,mask_cut,None,bgdModel,fgdModel,2,cv2.GC_INIT_WITH_MASK)
			#fgmask = np.float32(np.where((mask_cut==1)|(mask_cut==3),1,0))
			#bgmask = np.float32(np.where((mask_cut==0),1,0))
			#print np.sum(fgmask==1),np.sum(bgmask==1)
			"""
			_,fgmask = cv2.threshold(saliency_prob,0.6,1,cv2.THRESH_BINARY)
			_,bgmask = cv2.threshold(bg_es.variation,0.1,1,cv2.THRESH_BINARY_INV)
			bg_mask_blocks.append(bgmask);
			fg_mask_blocks.append(fgmask);
		else:
			break;
	return (frameCnt,blocks,fg_mask_blocks,bg_mask_blocks);

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


def detect_object(newMask,oldMask=None):
	window = []; (lbls,num) = label(newMask,connectivity=2,neighbors=4,return_num=True,background=0)
	for lbl in range(np.max(lbls)+1):
		pixels = np.where(lbls==lbl); _max = np.max(pixels,1); _min = np.min(pixels,1)
		area = np.prod(_max - _min);
		if  (4 * np.sum(newMask[_min[0]:_max[0],_min[1]:_max[1]]) > area) and \
			((oldMask is None) or (np.sum(oldMask[_min[0]:_max[0],_min[1]:_max[1]]) > 10)):
			rect = np.array([_min[1],_min[0],_max[1],_max[0]],dtype=np.uint8);
			window.extend([rect]);
	return window;
	
def write_block(vidwriter,frame,newMask,oldMask,windows,window_lbl):
	zero_frame  = np.zeros((newMask.shape[0],newMask.shape[1],3),dtype=np.float32);
	zero_frame[:,:,2] = np.float32(oldMask*255);
	out_frame1 = cv2.addWeighted(np.float32(frame),0.6,zero_frame,0.4,0.0);
	out_frame2 = frame
	for (idx,window) in enumerate(windows):
		cv2.rectangle(out_frame2, (window[0],window[1]), (window[2],window[3]), tuple(RANDOM_COLORS[window_lbl[idx],:]))
		cv2.putText(out_frame2,str(window_lbl[idx]), ((window[0],window[1])), cv2.FONT_HERSHEY_DUPLEX, 0.25, 255)
	zero_frame[:,:,2] = np.float32(newMask*255);
	out_frame2 = cv2.addWeighted(np.float32(frame),0.6,zero_frame,0.4,0.0);
	out_frame = np.hstack((out_frame1,out_frame2))
	vidwriter.write(np.uint8(out_frame))
	
def process(vidreader,out_path,batch=4):
	sal_rc = sal_instance(SaliencyMethods.REGION_CONTRAST,SaliencyProps())	
	bg_es = get_instance(BGMethods.FRAME_DIFFERENCING,vidreader.read_next)
	tracker = tracker_instance(0);
	bg_es.setShape((vidreader.height,vidreader.width))
	smoothner = smooth_instance(feats,1);	
	frame_idx = 0; N = vidreader.frames;
	batchsize = batch * 2
	_frames = []; _fgmasks = []; _bgmasks = [];
	frameOldMasks = []; frameMasks = [];
	while(vidreader.has_next()):
		print 'Proceesing Video... {0}%\r'.format((frame_idx*100/N))
		(cnt,frames,fgmasks,bgmasks) = process_next_block(sal_rc,bg_es,batchsize);
		if cnt > 0:
			frame_idx += cnt;
			_frames.extend(frames); _fgmasks.extend(fgmasks); _bgmasks.extend(bgmasks);
			newMasks = smoothner.process(_frames,_fgmasks,_bgmasks,range(batch/2,3*batch/2));
			newMasks = __morphologicalOps__(newMasks);	
			oldMasks = _fgmasks[batch/2:batch/2+newMasks.__len__()]
			frameOldMasks.extend(oldMasks); frameMasks.extend(newMasks);
			#print newMasks[0].shape,oldMasks[0].shape
			_frames = _frames[batch:];_fgmasks = _fgmasks[batch:]; _bgmasks = _bgmasks[batch:];
		else:
			break;
		batchsize = batch;
	vidwriter = VideoWriter(out_path,2*vidreader.width,vidreader.height)
	vidwriter.build();
	print "Tracking process starts..."
	vidreader.__reset__();
	vidreader.skip_frames(batch/2); frameIdx = -1;	
	numFrames= frameMasks.__len__();
	while(frameIdx + 1 < numFrames):
		frame = vidreader.read_next(); frameIdx += 1;
		window = detect_object(frameMasks[frameIdx],frameOldMasks[frameIdx])
		window_label = tracker.track_object(frame,window,frameMasks[frameIdx]);
		write_block(vidwriter,frame,frameMasks[frameIdx],frameOldMasks[frameIdx],window,window_label);
	vidreader.close();
	vidwriter.close()
	
if __name__ == "__main__":
	import sys;	
	if sys.argv.__len__()<=1:
		print "input path not provided........."
	else :
		inp = sys.argv[1];
		out = "test_results/final.avi";
		vidreader = VideoReader(inp)
		process(vidreader,out)
