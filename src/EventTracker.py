import cv2,numpy as np;
from io_module.video_reader import VideoReader
from io_module.video_writer import VideoWriter
from skimage import  segmentation, color
#from skimage.future import graph
from saliency import SaliencyProps,SaliencyMethods,get_instance as sal_instance
from bg_sub import get_instance,BGMethods
from utils import normalize
#from sklearn.cluster import MeanShift,DBSCAN, estimate_bandwidth
def process(vidreader,vidwriter):
	vidwriter.build();
	bg_es = get_instance(BGMethods.FRAME_DIFFERENCING,vidreader.read_next)
	bg_es.setShape((vidreader.height,vidreader.width))
	sal_rc = sal_instance(SaliencyMethods.REGION_CONTRAST,SaliencyProps())
	zero = np.zeros((vidreader.height,vidreader.width),np.float32)
	zero3 = np.zeros((vidreader.height,3*vidreader.width),np.float32)
	frame_idx = 0; N = vidreader.frames;
	prev_saliency = np.zeros((vidreader.height,vidreader.width),np.float32)
	#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	prev_rect = np.array([vidreader.width/4,vidreader.height/4,vidreader.width/2,vidreader.height/2],dtype=np.float32)
	kernel = np.ones((5,5),np.uint8)
	bgdModel = np.zeros((1,65),np.float64); fgdModel = np.zeros((1,65),np.float64)
	while(True):
		frame_idx += 1
		
		mask_es = bg_es.process();
		mask_es = cv2.medianBlur(mask_es,5)
		mask_es = cv2.medianBlur(mask_es,3)
		print 'Proceesing ... {0}%\r'.format((frame_idx*100/N)),
		if bg_es.isFinish():	
			break;
		sal_rc.process(bg_es.cur_frame);
		
		#mask_lbls = np.unique(mask_es * sal_rc.cut_labels)[1:]
		#if 	len(mask_lbls)>0:
		#	mask = sum([sal_rc.cut_labels==lbl for lbl in mask_lbls])
		#	mask = np.uint8(mask*255)
		#else:
		#	mask = zero
		#out_frame = sal_rc.ncut*mask[:,:,None]
		#frame = np.uint8(sal_rc.ncut * mask[:,:,None]);
		#mask_es = np.uint8(mask_es*255);
		#cv2.accumulateWeighted(sal_rc.saliency,prev_saliency,0.7,None)
		#saliency_prob =  bg_es.variation + prev_saliency;
		#_,mask = cv2.threshold(prev_mask,0.8 * np.max(prev_mask),255,cv2.THRESH_BINARY)
		
		#mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
		#mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
	
		#bgdModel = np.zeros((1,65),np.float64); fgdModel = np.zeros((1,65),np.float64)
		saliency_prob =  bg_es.variation + sal_rc.saliency;
		saliency_prob = normalize(saliency_prob)
		
		_,mask = cv2.threshold(saliency_prob,0.6,1,cv2.THRESH_BINARY)
		#mask = cv2.medianBlur(mask,3)
		pixels = np.where(mask==1); _max = np.max(pixels,1); _min = np.min(pixels,1)
		rect = np.array([_min[1],_min[0],_max[1]-_min[1],_max[0]-_min[0]],dtype=np.float32);
		#out_frame = np.uint8(bg_es.cur_frame.copy());
		#cv2.accumulateWeighted(rect,prev_rect,0.3,None);
		rect = tuple(rect)
		
		mask_cut = np.zeros(bg_es.cur_frame.shape[:2],np.uint8)
		#mask_cut[_min[1]:_max[1]-_min[1],_min[0]:_max[0]-_min[0]]=3
		mask_cut[saliency_prob >= 0.2] = 2
		mask_cut[saliency_prob >= 0.6] = 3
		mask_cut[saliency_prob >= 0.8] = 1
		#_,mask_cut = cv2.threshold(saliency_prob,0.8,1,cv2.THRESH_BINARY)
		
		#mask_cut = np.uint8(mask_cut)
		
		#count = np.array([np.sum((mask_cut%2)==lbl) for lbl in [0,1]]);
		#Laplcian features
		#_frame = np.uint8(cv2.Laplacian(bg_es.cur_frame,3))
		gray = cv2.cvtColor(bg_es.cur_frame,cv2.COLOR_RGB2GRAY)
		#eigen = cv2.cornerEigenValsAndVecs(gray,15,3);
		#eigen = eigen.reshape(gray.shape[0], gray.shape[1], 3, 2)
		#texture_mag = normalize(np.sqrt(eigen[:,:,0,0]**2 +  eigen[:,:,0,1]**2))*255
		#texture_dir1 = normalize(np.arctan2(eigen[:,:,1,1],eigen[:,:,1,0]))*255
		#texture_dir2 = normalize(np.arctan2(eigen[:,:,2,1],eigen[:,:,2,0]))*255
		
		t_frame = bg_es.cur_frame#np.uint8(np.dstack((gray,gray,gray)));
		regions = segmentation.slic(t_frame,50,20,convert2lab=True,multichannel=True)
		_frame = color.label2rgb(regions,bg_es.cur_frame, kind='avg')		
		#_frame = np.uint8(np.dstack((gray,gray,texture_mag)));
		#print 'texture flow',texture_flow.shape
		
		max_iter = 4
		#_frame = np.uint8(np.dstack((luminescence,texture_flow,laplacian)));
		#if np.all(count>4):
		cv2.grabCut(_frame,mask_cut,rect,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_MASK)
		#else:
		#cv2.grabCut(bg_es.cur_frame,mask_cut,rect,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_RECT)
		for _iter  in range(max_iter):	
			mask = np.uint8(np.where((mask_cut==2)|(mask_cut==0),0,1))
			mask_erode = cv2.erode(mask,kernel,1)
			mask_dilate = cv2.dilate(mask,kernel,1)
			mask_cut = np.uint8(3 * mask_dilate - 2 * mask_erode)
			if np.unique(mask_cut).__len__()==1:
				break;
			cv2.grabCut(_frame,mask_cut,rect,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_MASK)
		#out_frame = cv2.cvtColor(stacked*255,cv2.COLOR_GRAY2RGB);	
		#_,mask = cv2.threshold(out_frame[:,:,0],0.8*255,255,cv2.THRESH_BINARY)
		mask = np.float32(np.where((mask_cut==2)|(mask_cut==0),0,1))
		mask[mask_cut==3] = 0.8
		mask = cv2.medianBlur(np.uint8(mask),5)
		mask = cv2.medianBlur(mask,3)
		
		
		salProbColor = cv2.cvtColor(saliency_prob*255,cv2.COLOR_GRAY2RGB);
		_,salMask = cv2.threshold(sal_rc.saliency,0.8,1,cv2.THRESH_BINARY)
		mask_frame = np.float32(np.dstack((zero,salMask*255,zero)))
		salProbColor = cv2.addWeighted(np.float32(salProbColor),0.6,mask_frame,0.4,0.0);
		pixels = np.where(salMask==1); _max = np.max(pixels,1); _min = np.min(pixels,1)
		rect = np.array([_min[1],_min[0],_max[1]-_min[1],_max[0]-_min[0]],dtype=np.float32);
		cv2.accumulateWeighted(rect,prev_rect,0.3,None);
		rect = tuple(prev_rect)
		cv2.rectangle(salProbColor, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (0,0,255))
		
		mask_frame = np.float32(np.dstack((zero,mask*255,zero)))
		out_frame = cv2.addWeighted(np.float32(_frame),0.6,mask_frame,0.4,0.0);
		#out_frame = np.uint8( bg_es.cur_frame* mask[:,:,None])
		frame = np.hstack((salProbColor,out_frame))
		frame = cv2.resize(frame,(0,0),fx=0.25, fy=0.25)
		vidwriter.write(np.uint8(frame));
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
		vidwriter = VideoWriter(out,2*vidreader.width/4,vidreader.height/4)
		process(vidreader,vidwriter)
