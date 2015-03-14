from cv2 import cvtColor
from cv2 import COLOR_RGB2GRAY,COLOR_RGB2YUV,COLOR_RGB2LAB

"""
	Convert from RGB to LAB
"""
def LAB(image):
	return cvtColor(image,COLOR_RGB2LAB);
	

"""
	Convert from RGB to YUV
"""
def YUV(image):
	return cvtColor(image,COLOR_RGB2YUV);

"""
	Convert from RGB to GRAY
"""
def GRAY(image):
	return cvtColor(image,COLOR_RGB2GRAY);

