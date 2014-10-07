import cv2
import numpy as np
import math


def get_features(src_pt,dst_pt):
	out =[];
	#out.extend(src_pt);
	#out.extend(dst_pt);
	dist_x = src_pt[0]-dst_pt[0];
	dist_y = src_pt[1]-dst_pt[1];
	#out.extend([math.sqrt(dist_y**2 + dist_x**2)])
	out.extend([math.atan2(dist_y,dist_x)])
	return out;
	



	
	

