import cv2,numpy as np;
from io_module.video_reader import VideoReader
from io_module.video_writer import VideoWriter
from saliency import SaliencyProps,SaliencyMethods,get_instance as sal_instance
from smoothing import get_instance as smooth_instance
from features.pixellete import allFeats as feats

def process_next_block(vidreader,sal_rc,blockSize=4):
	frameCnt = 0;	blocks = []; bg_mask_blocks = [];	fg_mask_blocks = [];
	while(frameCnt < blockSize):
		if vidreader.has_next():
			frameCnt += 1
			frame = vidreader.read_next();
			blocks.append(np.array(frame,dtype=np.uint8))
			sal_rc.process(frame)
			bg_mask_blocks.append(sal_rc.bgmask);
			fg_mask_blocks.append(sal_rc.mask);
		else:
			break;
	return (frameCnt,blocks,fg_mask_blocks,bg_mask_blocks);

def write_block(vidwriter,frames,newMasks,oldMasks,mask_frame):
	for (frame,newMask,oldMask) in zip(frames,newMasks,oldMasks):
		mask_frame[:,:,2] = np.float32(oldMask*255);
		out_frame1 = cv2.addWeighted(np.float32(frame),0.6,mask_frame,0.4,0.0);
		mask_frame[:,:,2] = np.float32(newMask*255);
		out_frame2 = cv2.addWeighted(np.float32(frame),0.6,mask_frame,0.4,0.0);
		out_frame = np.hstack((out_frame1,out_frame2))
		vidwriter.write(np.uint8(out_frame))

def process(vidreader,vidwriter,batch=4):
	vidwriter.build();
	sal_rc = sal_instance(SaliencyMethods.REGION_CONTRAST,SaliencyProps())	
	smoothner = smooth_instance(feats,0);	
	frame_idx = 0; N = vidreader.frames;
	batchsize = batch * 2
	_frames = []; _fgmasks = []; _bgmasks = [];
	mask_frame  = np.zeros((vidreader.height,vidreader.width,3),dtype=np.float32);
	while(vidreader.has_next()):
		#print 'Proceesing ... {0}%\r'.format((frame_idx*100/N)),
		(cnt,frames,fgmasks,bgmasks) = process_next_block(vidreader,sal_rc,batchsize);
		if cnt > 0:
			frame_idx += cnt;
			_frames.extend(frames); _fgmasks.extend(fgmasks); _bgmasks.extend(bgmasks);
			newMasks = smoothner.process(_frames,_fgmasks,_bgmasks,range(batch/2,3*batch/2));
			curblockFrames = _frames[batch/2:batch/2+newMasks.__len__()]
			oldMasks = _fgmasks[batch/2:batch/2+newMasks.__len__()]
			write_block(vidwriter,curblockFrames,newMasks,oldMasks,mask_frame);
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
