import numpy as np
import cv2
from utils import normalize,pdist2
HSV_LOW = 0 
HSV_MAX = 180

class Tracker(object):
	def __init__(self):
		self.n_bins = 180;
		self.threshold = 0.1
		self.activeWindowFeats =  None;

	def __computeHist__(self,hsv_frame,window,mask):
		hsv_roi = hsv_frame[window[1]:window[3],window[0]:window[2],:];
		_mask = mask[window[1]:window[3],window[0]:window[2]]
		hist_val = cv2.calcHist([hsv_roi],[0],np.uint8(_mask),[self.n_bins],[HSV_LOW,HSV_MAX]).flatten()
		return normalize(hist_val);
		
	def track_object(self,frames,frameWindows,masks):
		num_frames = frameWindows.__len__()
		hsv_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) for frame in frames]
		if self.activeWindowFeats is None:
			self.activeWindowFeats = [self.__computeHist__(hsv_frames[0],window,masks[0]) for window in frameWindows[0]]
			self.windowMarkers = range(self.activeWindowFeats.__len__());
			self.windowCounters = [1] * self.activeWindowFeats.__len__();
			self.nextIdx = self.activeWindowFeats.__len__()
			
		_labels = [];
		for frameIdx in range(0,num_frames):
			mark_map = None
			num_activeWindows = self.activeWindowFeats.__len__()
			curWindowFeats = [self.__computeHist__(hsv_frames[frameIdx],window,masks[frameIdx]) for window in frameWindows[frameIdx]]
			num_windows = curWindowFeats.__len__();
			marker = -1 * np.ones(num_windows,dtype=np.int32)
			if num_windows > 0:
				dist =  (pdist2(np.array(self.activeWindowFeats),np.array(curWindowFeats))/np.sqrt(self.n_bins))
				(x,y) = np.meshgrid(range(num_activeWindows),range(num_windows)); 
				x = x.flatten(); y = y.flatten(); order = np.argsort(dist.flatten());
				mark_map = -1 * np.ones(num_activeWindows,dtype=np.int32)
				rev_mark_map = -1 * np.ones(num_windows,dtype=np.int32)
				for (_prev,_cur) in zip(x[order],y[order]):
					if mark_map[_prev]==-1 and rev_mark_map[_cur]==-1:
						mark_map[_prev]=_cur;
						rev_mark_map[_cur]=_prev;
			#Updating counters
			for _id in range(num_activeWindows):
				if not (mark_map is None) and (mark_map[_id]!=-1):
					_map = mark_map[_id];
					self.activeWindowFeats[_id]=curWindowFeats[_map];
					marker[_map] = self.windowMarkers[ _id];
					self.windowCounters[_id] += 1
				else:
					self.windowCounters[_id] -= 1
			#print frameIdx,	self.windowMarkers,self.windowCounters
			#Eliminating windows
			self.activeWindowFeats = [self.activeWindowFeats[idx] 
							for idx in range(num_activeWindows) if self.windowCounters[idx] > -1]
			self.windowMarkers = [self.windowMarkers[idx] 
							for idx in range(num_activeWindows) if self.windowCounters[idx] > -1]
			self.windowCounters = [self.windowCounters[idx] 
							for idx in range(num_activeWindows) if self.windowCounters[idx] > -1]
			# Adding windows
			for idx in range(num_windows):
				if marker[idx] == -1:
					self.activeWindowFeats.extend([curWindowFeats[idx]]);
					self.windowMarkers.extend([self.nextIdx]);
					marker[idx]=self.nextIdx;
					self.windowCounters.extend([1]); 
					self.nextIdx += 1;
			_labels.extend([marker]);
		return _labels;
		
		
		
		
	
