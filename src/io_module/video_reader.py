import cv2
class VideoReader(object):
	def __init__(self,filename,resize=(120,90)):
		self.filename = filename;
		self.cap = cv2.VideoCapture(filename) 
		self.resize = resize
		self.width = self.resize[0]#self.width = self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)*self.resize[0]
		self.height = self.resize[1]#self.height = self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)* self.resize[1]
		self.frames = self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
		self.read_frames = 0;
	
	def __reset__(self):
		self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,0);
		self.read_frames = 0;
		
	def read_next(self):
		if self.cap.isOpened():
			ret, frame = self.cap.read()
			self.read_frames += 1;
			if ret:
				#frame = cv2.resize(frame,(0,0),fx=self.resize[0],fy=self.resize[1]);
				frame = cv2.resize(frame,self.resize);
				return frame
			return None
	
	def read(self,startFrame=0,numFrames=1):
		frames = [];	self.read_frames = startFrame - 1
		self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,startFrame);
		while(numFrames>0):
			frame = self.read_next();
			if frame is None:# or self.read_frames > self.frames:
				break;
			frames.extend([frame]);numFrames -=1;
		return (len(frames),frames);
		
	def skip_frames (self,num_frames=0):
		frame = None;
		while(num_frames>0):
			frame = self.read_next();
			num_frames -= 1;
		return frame;
	
	def num_remaining_frames(self):
		return self.frames - self.read_frames;
	
	
	def has_next(self):
		return self.num_remaining_frames()>0;
	
	
	def close(self):
		self.cap.release();
	
		
