import cv2
import numpy as np


class IPMethod(object):
	SIFT = 1
	HARRIS_CORNER = 2
	TOMASI = 3

"""
	Extracts the interest points for the given input - numpy array or file
"""
def extract (method,inp):
		_ip_impl = InterestPointImpl.get_instace(method,inp);
		return _ip_impl.extract();


class InterestPointImpl(object):
	def __init__(self, inp):	
		if type(inp) is str:
			inp = __loadfile__(inp)
		assert(type(inp) is np.ndarray)
		self.inp=inp;
		self.output = None;
		
	def __loadfile__(self,inp):
		 return cv2.imread(inp);
		 
	"Returns the grayscale "
	def __gray__(self):
		gray = cv2.cvtColor(self.inp, cv2.COLOR_BGR2GRAY)
		#return np.float32(gray);
		return gray
		
	@staticmethod
	def get_instace(method,inp):
		assert(type(method) is int);
		if method == IPMethod.SIFT:
			return SIFT_Impl(inp)
		elif method == IPMethod.TOMASI:
			return TOMASI_Impl(inp)
		elif method == IPMethod.HARRIS_CORNER:
			return HARRIS_CORNER_Impl(inp);
		else:
			raise InvalidConfigType		
		
		
	def extract(self):
		raise NotImplementedError;
	

class TOMASI_Impl(InterestPointImpl):
	def __init__(self,inp,maxCorners=30,minDistance=10,qualityLevel=0.01):
		super(TOMASI_Impl,self).__init__(inp);
		self.maxCorners = maxCorners;
		self.qualityLevel = qualityLevel
		self.minDistance = minDistance
		self.maxCorners = maxCorners
	def extract(self):
		kp = cv2.goodFeaturesToTrack(self.__gray__(),self.maxCorners,self.qualityLevel,self.minDistance);
		numCorners = kp.shape[0];
		kp = kp.reshape(numCorners,2);
		kp = [cv2.KeyPoint(kp[ind][0],kp[ind][1],self.minDistance) for ind in xrange(numCorners)]
		return kp

class HARRIS_CORNER_Impl(InterestPointImpl):
	def __init__(self,inp,block_size=3,ap_size=3,harris_param=0.04,qualityLevel=0.01):
		super(HARRIS_CORNER_Impl,self).__init__(inp);
		self.block_size = block_size;
		self.ap_size=ap_size;
		self.k = harris_param;
		self.qualityLevel = qualityLevel;
		
	def extract(self):
		#Harris corner dector works on single channel
		self.out = cv2.cornerHarris(self.__gray__(),self.block_size,self.ap_size,self.k);
		dest=cv2.dilate(self.out,None)		
		x,y = np.where(dest>self.qualityLevel*dest.max());
		kp = [cv2.KeyPoint(_x,_y,dest[_x][_y]) for _x,_y in zip(x,y)]
		return kp

class SIFT_Impl(InterestPointImpl):
	def __init__(self,inp):
		super(SIFT_Impl,self).__init__(inp);
	def extract(self):
		sift = cv2.SIFT()
		return sift.detect(self.__gray__(),None)
		



		
	
	



	
	

