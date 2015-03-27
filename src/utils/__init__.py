import numpy as np
import os,errno
APPROX_ZERO = 0.0000001
"""
	Normalizes the _input vector/matrix based on maximum and minimum value
"""
def normalize(_input):
	_max = np.max(_input); _min = np.min(_input)
	return (_input-_min)/np.float32((_max-_min+APPROX_ZERO));
	

"""
	Normalizes matrix along column or row
		axis = 0 across column,
		axis = 1 across row
"""
def normalize2D(_input,axis):
	assert((axis==1)|(axis==0))
	_max = np.max(_input,axis); _min = np.min(_input,axis)
	if (axis==0):
		return (_input-_min)/np.float32((_max-_min));
	else:
		return (_input-_min[:,None])/np.float32((_max-_min+APPROX_ZERO)[:,None]);

"""
	Computes the pairwise euclidean distance across every pair of vector in A & B
"""
def pdist2(A,B):
	A_2 = np.outer(np.sum(np.square(A),1),np.ones(B.shape[0]))
	B_2 = np.outer(np.ones(A.shape[0]),np.sum(np.square(B),1))
	AoutB = np.dot(A,B.transpose())
	return np.sqrt(A_2 + B_2 - 2*AoutB)
	
def getDirectories(path):
	"""Return a list of directories name on the specifed path"""
	return [file for file in os.listdir(path)
			if os.path.isdir(os.path.join(path, file))]


"""
	Creates directory if dirpath is valid or 
"""
def mkdirs(dirpath):
	if dirpath is None:
		raise EnvironmentError('dirpath is not provided');
	try:
		os.makedirs(dirpath)
	except OSError as exc: # Python >2.5
		if exc.errno == errno.EEXIST and os.path.isdir(dirpath):
			pass
		else: 
			raise

"""
	Its a dummy task object which nullifies asynchs
"""		
class DummyTask:
    def __init__(self, data):
        self.data = data
    def ready(self):
        return True
    def get(self):
        return self.data
