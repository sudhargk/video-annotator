from features.color import GRAY
import numpy as np

from bg_sub import BGSubtractionImpl
"""
	A  background subtraction based on eigen subtraction
	Args:
		K (int) : Number of eigen direction neccesarily less than N, default = 2
"""

class EigenBGSubImpl(BGSubtractionImpl):
	def __init__(self,K = 2,threshold=0.25):
		super(EigenBGSubImpl,self).__init__(threshold)
		self.K = K;
		
	def process(self,cur_frame,prev_frames):
		N = prev_frames.__len__();
		assert(self.K<=N ),("length of prev frames less than " + str(self.K))
		shape = cur_frame.shape[:2];
		prev_frames = [GRAY(frame).flatten() for frame in prev_frames];		
		cur_frame = GRAY(cur_frame)			
		mean = np.mean(prev_frames,axis=0)
		mean_subtracted = [frame - mean for frame in prev_frames];
		mean_subtracted = np.asarray(mean_subtracted)
		eigv,eigt = np.linalg.eig(np.cov(mean_subtracted));
		eigt = np.dot(mean_subtracted.T,eigt); 
		eigt = eigt / np.linalg.norm(eigt,axis=0)
		idx = np.argsort(eigv)[::-1]
		eigt = eigt[:,idx]; eigv = eigv[idx]	
		score = np.dot(cur_frame.flatten()-mean,eigt[:,:self.K])		
		recon = np.dot(eigt[:,:self.K],score)+mean
		recon = np.uint8(recon.reshape(shape))
		diff = self.__frame_differencing__(recon,cur_frame)
		return diff;

