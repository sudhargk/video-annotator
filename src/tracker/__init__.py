TRACK_CORRELATION_BASED = 0
TRACK_MATCHING_BASED = 1
TRACK_KALMAN_BASED = 2

def get_instance(method):
	if method == TRACK_CORRELATION_BASED:
		from tracker.correlation_based import CorrelationBased as Tracker
	elif method == TRACK_MATCHING_BASED:
		from tracker.matching_based import MatchingBased as Tracker
	elif method == TRACK_KALMAN_BASED:
		from tracker.kalman import Kalman as Tracker
	return Tracker();	

class Tracker(object):
	def track_object(self,frames,frameWindows,masks):
		raise NotImplementedError;
		
		
		
		
	
