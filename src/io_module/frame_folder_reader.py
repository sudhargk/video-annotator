import cv2
import os, os.path
class FrameFolderReader(object):
	def __init__(self,foldername,resize=(160,120)):
		assert(os.path.isdir(foldername)),foldername + " is not a directory";
		self.foldername = foldername;
		self.resize = resize
		self.fileNames = [os.path.join(self.foldername,name) for name in os.listdir(self.foldername) \
						if os.path.isfile(os.path.join(self.foldername, name))]
		self.fileNames.sort();
		tmp_frame = cv2.imread(self.fileNames[0]);
		self.width 	= self.resize[0]#self.width = tmp_frame.shape[1] * resize[0]
		self.height	= self.resize[1]#self.height = tmp_frame.shape[0] * resize[1]
		self.frames = len(self.fileNames)
		self.read_frames = 0;
	
	def __reset__(self):
		self.read_frames = 0;
		
	def read_next(self):
		if self.read_frames < self.frames:
			frame = cv2.imread(self.fileNames[self.read_frames]);
			#frame = cv2.resize(frame, (0,0), fx=self.resize[0], fy=self.resize[1])
			frame = cv2.resize(frame, self.resize)
			self.read_frames += 1
			return frame
		return None
		
	def skip_frames (self,num_frames=0):
		self.read_frames = min(self.read_frames+num_frames,self.frames)
		return None
	
	def num_remaining_frames(self):
		return self.frames - self.read_frames;
	
	
	def has_next(self):
		return self.num_remaining_frames()>0;
	
	
	def close(self):
		self.fileNames = []
	
		
