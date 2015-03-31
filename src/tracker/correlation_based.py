import numpy as np
import cv2
from utils import normalize,pdist2
from tracker import Tracker
HSV_LOW = 0; HSV_MAX = 180
class CorrelationBased(Tracker):
	def __init__(self):
		self.n_bins = 180;
		self.threshold = 0.1
		self.activeWindowFeats =  None;
		self.activeWindowVolume = {}
		

	def __computeFeats__(self,hsv_frame,window,mask,shape):
		hsv_roi = hsv_frame[window[1]:window[3],window[0]:window[2],:];
		_mask = mask[window[1]:window[3],window[0]:window[2]]		
		shape_val = np.array([window[1]/shape[0],window[0]/shape[1],
					(window[3]-window[1])/shape[0],
					(window[2]-window[0])/shape[1]]);
		center_dist = pow(0.5*(shape[1]-(window[2]+window[0]))/shape[1],2);
		center_dist += pow(0.5*(shape[0]-(window[3]+window[1]))/shape[0],2);
		center_dist = np.sqrt(center_dist)
		
		hist_val = cv2.calcHist([hsv_roi],[0],np.uint8(_mask),[self.n_bins],[HSV_LOW,HSV_MAX]).flatten()
		hist_val = cv2.calcHist([hsv_roi],[0],None,[self.n_bins],[HSV_LOW,HSV_MAX]).flatten()
		return np.hstack((normalize(hist_val),shape_val,center_dist));
		
	def track_object(self,frame,frameWindow,mask):
		shape = frame.shape[:2];
		hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		if self.activeWindowFeats is None:
			self.activeWindowFeats = [self.__computeFeats__(hsv_frame,window,mask,shape) for window in frameWindow]
			self.windowMarkers = range(len(self.activeWindowFeats));
			self.windowCounters = [1] * len(self.activeWindowFeats);
			self.nextIdx = len(self.activeWindowFeats)
		
		mark_map = None; 	num_activeWindows = len(self.activeWindowFeats)
		curWindowFeats = [self.__computeFeats__(hsv_frame,window,mask,shape) for window in frameWindow]
		num_windows = len(curWindowFeats);
		marker = -1 * np.ones(num_windows,dtype=np.int32)
		if num_windows > 0 and num_activeWindows > 0:
			hist_dist =  (pdist2(np.array(self.activeWindowFeats)[:,:self.n_bins],np.array(curWindowFeats)[:,:self.n_bins])/np.sqrt(self.n_bins))
			rect_dist = (pdist2(np.array(self.activeWindowFeats)[:,self.n_bins:self.n_bins+4],np.array(curWindowFeats)[:,self.n_bins:self.n_bins+4])/np.sqrt(self.n_bins))
			centre_dist = (pdist2(np.array(self.activeWindowFeats)[:,self.n_bins+4:],np.array(curWindowFeats)[:,self.n_bins+4:])/np.sqrt(self.n_bins))
			dist = hist_dist*rect_dist*centre_dist
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
				#if self.activeWindowVolume[_id] is None:
				#	self.activeWindowVolume[_id] = 
				
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
		return marker;
		
		
		
		
	
