import cv2
class VideoReader(object):
	def __init__(self,filename):
		self.filename = filename;
		self.cap = cv2.VideoCapture(filename) 
		self.width = self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
		self.height = self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
		self.frames = self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
		self.read_frames = 0;
		
	def read_next(self):
		if self.cap.isOpened():
			ret, frame = self.cap.read()
			self.read_frames += 1;
			if ret:
				return frame
			return None
		
	def skip_frames (self,num_frames=0):
		frame = None;
		while(num_frames>0):
			frame = self.read_next();
			num_frames -= 1;
		return frame;
	
	def num_remaining_frames(self):
		return self.frames - self.read_frames;
	
	def close(self):
		self.cap.release();
	
		
