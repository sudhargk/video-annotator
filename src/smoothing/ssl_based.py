import time,cv2,numpy as np
from sklearn.semi_supervised import label_propagation

from  smoothing import Smoothing

class SSLBased(Smoothing):
	def __init__(self,_feats, max_iter = 10):
		super(SSLBased,self,).__init__(_feats)
		self.max_iter =  max_iter
		self.ssl_model = label_propagation.LabelSpreading(kernel='rbf',n_neighbors=10,max_iter = self.max_iter);
		
	def __build_model__(self,frame_feats,target,num_examples = 1000):
		#sel = np.int8(np.linspace(0, len(target), num=num_examples))
		sel = np.random.permutation(range(len(target)))[:num_examples]
		frame_feats = frame_feats[sel,:]; target = target[sel];
		#print frame_feats.shape,target.shape
		self.ssl_model.fit(frame_feats,target);
		return self.ssl_model;
		
	def __get_score__ (self,ssl_model,frame_feats,shape):
		assert(shape[0]*shape[1] == frame_feats.__len__())
		frames_mask = ssl_model.predict(frame_feats);
		frames_mask =  (frames_mask).reshape((shape[0],shape[1]))
		return np.uint8(frames_mask);
		
	def process(self,blocks,fgMasks,bgMasks,smoothFrames=None):
		numBlocks = len(blocks);
		assert(numBlocks>0)
		shape = blocks[0].shape;	frameSize = np.prod(shape[:2])
		blockFeats = np.vstack([self.feats(block) for block in blocks])
		blockMask = np.hstack([(2*fgmask + bgmask -1).flatten() for (fgmask,bgmask) in zip(fgMasks,bgMasks)])
		finalMask = np.int8(blockMask)
		ssl_model = self.__build_model__(blockFeats,finalMask);
		if smoothFrames is None:
			smoothFrames = range(numBlocks);
		else:
			smoothFrames = [idx for idx in smoothFrames if idx < numBlocks];			
		newMasks = [self.__get_score__(ssl_model,
							blockFeats[frameIdx*frameSize:(frameIdx+1)*frameSize],shape) 
							for frameIdx in smoothFrames]
		newMasks= self.__post_process__(fgMasks,newMasks);
		return newMasks
