import cv2,numpy as np
from texture import eigenBasedFeats
from color import LAB,YUV
from utils import normalize2D

"""
	Extracts the  texture feats (3 dimension)
		(texture-mag,texture-dir1,texture-dir2)
"""
def texture(image):
	imageDim = np.prod(image.shape[:2])
	return eigenBasedFeats(image).reshape(imageDim,3);

	
"""
	Extract color based feats returns LAB and YUV (6 dimension)
"""
def color(image):
	imageDim = np.prod(image.shape[:2])
	_rgb = image.reshape(imageDim,3)
	_lab = LAB(image).reshape(imageDim,3);
	_yuv = YUV(image).reshape(imageDim,3);
	_color = np.hstack((_lab,_yuv));
	#_color = _rgb
	return _color

"""
	Returns x_pos and y_pos feats (2 dimension)
"""
def position(image):
	shape = image.shape
	x,y  = np.meshgrid(range(shape[0]),range(shape[1]))
	x = x.flatten(); y = y.flatten();
	return np.vstack((x,y)).transpose();


def gradient(image,bin_n=16):
	imageDim = np.prod(image.shape[:2])
	gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
	gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
	mag, ang = cv2.cartToPolar(gx, gy)
	grad = np.hstack((mag.reshape(imageDim,3),ang.reshape(imageDim,3)))    
	return grad
"""
	Return all feats
"""
def allFeats(image):
	_color = color(image);
	_texture = texture(image);
	_position = position(image);
	#_grad = gradient(image);
	#allFeats = _color
	allFeats = np.hstack((_color,_texture))
	#allFeats = np.hstack((_color,_grad,_texture))
	return normalize2D(allFeats,0);
