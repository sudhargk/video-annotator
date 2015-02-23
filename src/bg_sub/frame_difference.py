from bg_sub import BGSubtractionImpl

"""
	A Simple frame differencing approach
	Args:
		threshold (int) : values [0-256] integer threshold on difference between foreground 
					and background, default = 50
"""
class FrameDifferencingImpl(BGSubtractionImpl):
	def __init__(self,_nextFrame,threshold=50):
		super(FrameDifferencingImpl,self).__init__(_nextFrame)
		self.prev_frame = None
		self.cur_frame = None
		self.threshold = threshold
	
	def process(self):
		if self.prev_frame is None:
			self.prev_frame = self._nextFrame()
		self.cur_frame = self._nextFrame()
		if self.cur_frame is None:
			self.finish = True
		else:
			diff = self.frame_differencing(self.prev_frame,self.cur_frame,threshold = self.threshold)
			self.prev_frame = self.cur_frame
			return diff;
