from numpy import float32
from cv2 import accumulateWeighted,convertScaleAbs
from bg_sub import BGSubtractionImpl

"""
	A Simple frame differencing approach when single frame given otherwise  computes moving average
	Args:
		alpha (float) : values [0-1] specifies the importance to current frame, default 0.3
"""
class FrameDifferencingImpl(BGSubtractionImpl):
	def __init__(self,alpha=0.0,threshold = 0.2):
		super(FrameDifferencingImpl,self).__init__(threshold)
		self.alpha = alpha;
		
	def process(self,cur_frame,prev_frames):
		assert(isinstance(prev_frames,list)),"prev_frames is not a list"
		acum_frame = float32(prev_frames[0])
		for frame in prev_frames[1:]:
			accumulateWeighted(float32(frame),acum_frame,self.alpha,None);
		absScaleFrame = convertScaleAbs(acum_frame)
		diff = self.__frame_differencing__(absScaleFrame,cur_frame)
		return diff;
