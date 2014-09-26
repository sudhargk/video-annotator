import cv2
import numpy as np


class IPMethod(object):
	SIFT = 1
	HARRIS_CORNER = 2
	TOMASI = 3

"""
	Extracts the interest points for the given input - numpy array or file
"""
def get_instace(method):
		return InterestPointImpl.get_instace(method);
		


class InterestPointImpl(object):
	"""loads the file """
	@staticmethod	
	def __loadfile__(inp):
		 return cv2.imread(inp);
	"""Returns the grayscale """
	@staticmethod 	
	def __gray__(inp):
		gray = cv2.cvtColor(inp, cv2.COLOR_BGR2GRAY)
		return gray
	"""loads input"""
	@staticmethod
	def __load__(inp):
		if type(inp) is str:
			inp = __loadfile__(inp)
		assert(type(inp) is np.ndarray)
		return inp;
				
	@staticmethod
	def get_instace(method):
		assert(type(method) is int);
		if method == IPMethod.SIFT:
			return SIFT_Impl()
		elif method == IPMethod.TOMASI:
			return TOMASI_Impl()
		elif method == IPMethod.HARRIS_CORNER:
			return HARRIS_CORNER_Impl();
		else:
			raise InvalidConfigType		
		
	def detect(self,inp):
		raise NotImplementedError;
		
	"""Extract SIFT features for deafault corner detectros"""
	def compute(self,inp,kp=None):
		inp = self.__load__(inp);
		if kp is None:
			kp = self.detect(inp);
		sift = cv2.SIFT();
		return sift.compute(__gray__(inp),kp);

	def detectAndCompute(self):
		inp = self.__load__(inp);
		kp = self.detect(inp);
		kpDesc = self.compute(inp,kp);
		return kp,kpDesc;
	

class TOMASI_Impl(InterestPointImpl):
	def __init__(self,maxCorners=30,minDistance=10,qualityLevel=0.01):
		self.maxCorners = maxCorners;
		self.qualityLevel = qualityLevel
		self.minDistance = minDistance
		self.maxCorners = maxCorners
		
	def detect(self,inp):
		inp = self.__load__(inp);
		kp = cv2.goodFeaturesToTrack(self.__gray__(inp),self.maxCorners,self.qualityLevel,self.minDistance);
		numCorners = kp.shape[0];
		kp = kp.reshape(numCorners,2);
		kp = [cv2.KeyPoint(kp[ind][0],kp[ind][1],self.minDistance) for ind in xrange(numCorners)]
		return kp
		

class HARRIS_CORNER_Impl(InterestPointImpl):
	def __init__(self,block_size=3,ap_size=3,harris_param=0.04,qualityLevel=0.01):
		self.block_size = block_size;
		self.ap_size=ap_size;
		self.k = harris_param;
		self.qualityLevel = qualityLevel;
		
	def detect(self,inp):
		#Harris corner dector works on single channel
		inp = self.__load__(inp);
		dest = cv2.cornerHarris(self.__gray__(inp),self.block_size,self.ap_size,self.k);
		dest=cv2.dilate(dest,None)		
		x,y = np.where(dest>self.qualityLevel*dest.max());
		kp = [cv2.KeyPoint(_y,_x,dest[_x][_y]) for _x,_y in zip(x,y)]
		return kp

class SIFT_Impl(InterestPointImpl):
	def __init__(self):
		self.sift = cv2.SIFT()
		
	def detect(self,inp):
		inp = self.__load__(inp);
		kp = self.sift.detect(self.__gray__(inp),None)
		return kp
		
	def compute(self,inp,kp=None):
		inp = self.__load__(inp);
		inp = self.__gray__(inp);
		if kp is None:
			kp = self.sift.detect(inp,None)
		kpDesc = self.sift.compute(inp,kp);
		return 	kpDesc


		
	
	



	
	

