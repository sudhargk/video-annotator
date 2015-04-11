import cv2,numpy as np
from skimage.measure import label


class TrackerMethods(object):
	CORRELATION_BASED = 0
	MIXTURE_BASED = 1
	KALMAN_BASED = 2

def get_instance(method):
	if method == TrackerMethods.CORRELATION_BASED:
		from tracker.correlation_based import CorrelationBased as Tracker;
	elif method == TrackerMethods.MIXTURE_BASED:
		from tracker.mixture_based import MixtureBased as Tracker
	else:
		raise NotImplementedError;
	"""
	elif method == TrackerMethods.KALMAN_BASED:
		from tracker.kalman import Kalman as Tracker
	"""
	
	return Tracker();	

class Tracker(object):
	def track_object(self,frames,frameWindows,masks):
		raise NotImplementedError;
	
	def __detect_object__(self,newMask,delta = 10):
		(lbls,num) = label(newMask,connectivity=2,neighbors=4,return_num=True,background=0)
		window = []; 
		for lbl in range(np.max(lbls)+1):
			pixels = np.where(lbls==lbl); _max = np.max(pixels,1); _min = np.min(pixels,1)
			rect = np.array([_min[1],_min[0],_max[1],_max[0]],dtype=np.uint8);
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
		
		
		
	
