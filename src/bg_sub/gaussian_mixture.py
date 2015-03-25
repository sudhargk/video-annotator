from cv2 import BackgroundSubtractorMOG,getStructuringElement,morphologyEx
from cv2 import MORPH_ELLIPSE,MORPH_CLOSE,MORPH_OPEN
from bg_sub import BGSubtractionImpl

"""
	A background subtraction technique using gausian mixture technique
"""
class BackgroundSubtractorMOGImpl(BGSubtractionImpl):
	def __init__(self,threshold=0.05):
		super(BackgroundSubtractorMOGImpl,self).__init__(threshold)
		
	def process(self,cur_frame,prev_frames):
		fgbg = BackgroundSubtractorMOG()
		kernel = getStructuringElement(MORPH_ELLIPSE,(3,3))
		[fgbg.apply(prev_frame) for prev_frame in prev_frames];
		fgmask = fgbg.apply(cur_frame)
		fgmask = morphologyEx(fgmask, MORPH_CLOSE, kernel)
		fgmask = morphologyEx(fgmask, MORPH_OPEN, kernel)
		return fgmask*255;
