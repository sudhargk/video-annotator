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
from tracker import Tracker


MAX_COLORS = 100
RANDOM_COLORS = np.random.randint(256, size=(MAX_COLORS, 3))

def process_next_block(sal_rc,bg_es,blockSize=4):
	frameCnt = 0;	blocks = []; bg_mask_blocks = [];	fg_mask_blocks = [];
	while(frameCnt < blockSize):
		mask_es = bg_es.process();
		if not bg_es.isFinish():
			frameCnt += 1
			blocks.append(np.array(bg_es.cur_frame,dtype=np.uint8))
			#start_time = time.time();
			sal_rc.process(bg_es.cur_frame);
			saliency_prob =  bg_es.variation + sal_rc.saliency;
			saliency_prob = normalize(saliency_prob)
			_,fgmask = cv2.threshold(saliency_prob,0.6,1,cv2.THRESH_BINARY)
			_,bgmask = cv2.threshold(saliency_prob,0.4,1,cv2.THRESH_BINARY_INV)
			#print "saliency : ",time.time()-start_time	
			bg_mask_blocks.append(bgmask);
			fg_mask_blocks.append(fgmask);
		else:
			break;
	return (frameCnt,blocks,fg_mask_blocks,bg_mask_blocks);

def __morphologicalOps__(mask,kernel):
	mask = cv2.medianBlur(mask,5)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	mask = cv2.medianBlur(mask,5)
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,kernel)
	mask = remove_small_objects(mask==1,min_size=128,connectivity=2)
	
	return 	mask;
	
def write_block(vidwriter,frames,newMasks,oldMasks,mask_frame,tracker):
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5));
	num_frames = newMasks.__len__();
	frameWindows = [];
	for idx in range(num_frames):
		window = []; newMasks[idx] = __morphologicalOps__(newMasks[idx],kernel);
		(lbls,num) = label(newMasks[idx],connectivity=2,neighbors=4,return_num=True,background=0)
		for lbl in range(np.max(lbls)+1):
			pixels = np.where(lbls==lbl); _max = np.max(pixels,1); _min = np.min(pixels,1)
			if np.sum(oldMasks[idx][_min[0]:_max[0],_min[1]:_max[1]]) > 10:
				rect = np.array([_min[1],_min[0],_max[1],_max[0]],dtype=np.uint8);
				window.extend([rect]);
		frameWindows.extend([window]);
	window_labels = tracker.track_object(frames,frameWindows,newMasks);
	
	for (frame,newMask,oldMask,windows,window_lbl) in zip(frames,newMasks,oldMasks,frameWindows,window_labels):
		mask_frame[:,:,2] = np.float32(oldMask*255);
		out_frame1 = cv2.addWeighted(np.float32(frame),0.6,mask_frame,0.4,0.0);
		out_frame2 = frame
		for (idx,window) in enumerate(windows):
			cv2.rectangle(out_frame2, (window[0],window[1]), (window[2],window[3]), tuple(RANDOM_COLORS[window_lbl[idx],:]))
			cv2.putText(out_frame2,str(window_lbl[idx]), ((window[0],window[1])), cv2.FONT_HERSHEY_DUPLEX, 0.25, 255)
		mask_frame[:,:,2] = np.float32(newMask*255);
		out_frame2 = cv2.addWeighted(np.float32(frame),0.6,mask_frame,0.4,0.0);
		out_frame = np.hstack((out_frame1,out_frame2))
		vidwriter.write(np.uint8(out_frame))

def process(vidreader,vidwriter,batch=12):
	vidwriter.build();
	sal_rc = sal_instance(SaliencyMethods.REGION_CONTRAST,SaliencyProps())	
	bg_es = get_instance(BGMethods.FRAME_DIFFERENCING,vidreader.read_next)
	tracker = Tracker();
	bg_es.setShape((vidreader.height,vidreader.width))
	smoothner = smooth_instance(feats,1);	
	frame_idx = 0; N = vidreader.frames;
	batchsize = batch * 2
	_frames = []; _fgmasks = []; _bgmasks = [];
	mask_frame  = np.zeros((vidreader.height,vidreader.width,3),dtype=np.float32);
	while(vidreader.has_next()):
		#print 'Proceesing ... {0}%\r'.format((frame_idx*100/N)),
		(cnt,frames,fgmasks,bgmasks) = process_next_block(sal_rc,bg_es,batchsize);
		if cnt > 0:
			frame_idx += cnt;
			_frames.extend(frames); _fgmasks.extend(fgmasks); _bgmasks.extend(bgmasks);
			newMasks = smoothner.process(_frames,_fgmasks,_bgmasks,range(batch/2,3*batch/2));
			curblockFrames = _frames[batch/2:batch/2+newMasks.__len__()]
			oldMasks = _fgmasks[batch/2:batch/2+newMasks.__len__()]
			write_block(vidwriter,curblockFrames,newMasks,oldMasks,mask_frame,tracker);
			_frames = _frames[batch:];_fgmasks = _fgmasks[batch:]; _bgmasks = _bgmasks[batch:];
		else:
			break;
		batchsize = batch;
	vidreader.close()
	vidwriter.close()
	
if __name__ == "__main__":
	import sys;	
	if sys.argv.__len__()<=1:
		print "input path not provided........."
	else :
		inp = sys.argv[1];
		out = "test_results/final.avi";
		vidreader = VideoReader(inp)
		vidwriter = VideoWriter(out,2*vidreader.width,vidreader.height)
		process(vidreader,vidwriter)
