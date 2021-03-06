import cv2,sys,os
import numpy as np
from saliency import get_instance,SaliencyProps,SaliencyMethods
def overlay_write(_input,mask_frame,saliency,filename):
	zero = np.zeros(mask_frame.shape,np.uint8);
	saliency = cv2.cvtColor(saliency*255,cv2.COLOR_GRAY2RGB);
	mask_frame = np.uint8(mask_frame*255);
	mask_frame = np.dstack((zero,zero,mask_frame))
	out_frame = cv2.addWeighted(_input,0.4,mask_frame,0.6,0.0);
	out_frame = np.hstack((saliency,out_frame));
	cv2.imwrite(filename,out_frame);
	
def test_saliency_cf(inp,props):
	sal = get_instance(SaliencyMethods.COLOR_FREQUENCY,props);
	saliency = sal.process(inp);
	mask_frame = sal.saliency_cut(inp,saliency);
	overlay_write(inp,mask_frame,saliency,"test_results/cf_final.png");
	print "Tested saliency(color frequency)...    [DONE]"

def test_saliency_ca(inp,props):
	sal = get_instance(SaliencyMethods.CONTEXT_AWARE,props);
	saliency = sal.process(inp);
	mask_frame = sal.saliency_cut(inp,saliency);
	overlay_write(inp,mask_frame,saliency,"test_results/ca_final.png");
	print "Tested saliency(context aware)...    [DONE]"

def test_saliency_rc(inp,props):
	sal = get_instance(SaliencyMethods.REGION_CONTRAST,props);
	saliency = sal.process(inp);
	mask_frame = sal.saliency_cut(inp,saliency);
	overlay_write(inp,mask_frame,saliency,"test_results/rc_final.png");
	print "Tested saliency(region contrast)...    [DONE]"
	
def test_saliency_sd(inp,props):
	sal = get_instance(SaliencyMethods.SPECTRAL_DISTRIBUTION,props);
	saliency = sal.process(inp);
	mask_frame = sal.saliency_cut(inp,saliency);
	overlay_write(inp,mask_frame,saliency,"test_results/sd_final.png");		
	print "Tested saliency(spectral distribution)...    [DONE]"
	
def test(img_path = "../examples/images/sample_image1.png"):
	os.environ.setdefault("PROFILE_PATH","test_results");	
	inp = cv2.imread(img_path);
	props = SaliencyProps()
	test_saliency_cf(inp,props);
	test_saliency_ca(inp,props);
	test_saliency_rc(inp,props);
	test_saliency_sd(inp,props);
	print "Tested all saliency methods ...   [DONE]"

if __name__ == "__main__":
	import sys;
	if sys.argv.__len__()<=1:
		test();
	else :
		test(sys.argv[1]);
