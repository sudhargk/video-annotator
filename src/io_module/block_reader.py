import logging,numpy as np

logger = logging.getLogger(__name__)

class BlockReader(object):
	def __init__(self,blockSize,inputShape,readNextFrameFn):
		self.blockSize = blockSize
		self.readNextFrameFn = readNextFrameFn;
		self.inputShape = inputShape;
		self.defaultZero = np.zeros(self.inputShape,dtype=np.float64);
		self.isFinished = False
		
	def readNextBlock(self):
		numFrames = 0;	frameBlock = []; labelBlock = [];
		while(numFrames<self.blockSize):
			(nextFrame,lbl) = self.readNextFrameFn();
			if nextFrame is None:
				self.isFinished = True
				break;
			frameBlock.extend([nextFrame]);
			labelBlock.extend([lbl]);
			numFrames += 1;
		padFrames = 0				#pad_zeros
		while(padFrames+numFrames<self.blockSize):
			frameBlock.extend([self.defaultZero]);
			labelBlock.extend([-1]);
			padFrames += 1
		return (numFrames,frameBlock,labelBlock)
