import numpy as np

def comparator(gt_mask,ac_mask,roi_mask):
	tp = np.sum((gt_mask==1) & (ac_mask==1) & (roi_mask==1));
	fp = np.sum((gt_mask==0) & (ac_mask==1) & (roi_mask==1));
	fn = np.sum((gt_mask==1) & (ac_mask==0) & (roi_mask==1));
	tn = np.sum((gt_mask==0) & (ac_mask==0) & (roi_mask==1));
	return [tp,fp,fn,tn,0]

def updateConfusion(accumulator,confusion):
	for idx in range(5):
		accumulator[idx] += confusion[idx];

def getNewConfusion():
	return [0.0]*5;
