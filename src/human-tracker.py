import cv2
import numpy as np
from io_module.video_reader import VideoReader
from io_module.video_writer import VideoWriter
#A box r inside the other box q
def inside(r, q):
	rx, ry, rw, rh = r
	qx, qy, qw, qh = q
	return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

#Draw rectangle and store on img
def draw_detections(img, rects, thickness = 1,shrinkage = 0.05):
	for x, y, w, h in rects:
		# the HOG detector returns slightly larger rectangles than the real objects.
		# so we slightly shrink the rectangles to get a nicer output.
		pad_w, pad_h = int(shrinkage*w), int(shrinkage*h)
		cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)

def tracker(inPath,outPath="test_results/tracker.avi"):
	#initialization for default value
	hog = cv2.HOGDescriptor()
	hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector())
	vidreader = VideoReader(inPath)
	vidwriter = VideoWriter(outPath,vidreader.width,vidreader.height)
	vidwriter.build();
	frame_idx =0; N = vidreader.frames;
	while vidreader.num_remaining_frames() > 0:
		frame_idx += 1;
		cur_frame = vidreader.read_next();
		print 'Perform human tracking.... {0}%\r'.format((frame_idx*100/N)),
		if cur_frame is None:
			break;
		gray_cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
		found, w = hog.detectMultiScale(gray_cur_frame, winStride=(8,8), padding=(32,32), scale=1.05)
		found_filtered = []
		#discarding the bounding box on within other
		for ri, r in enumerate(found):
			for qi, q in enumerate(found):
				if ri != qi and inside(r, q):
					break
				else:
					found_filtered.append(r)
		draw_detections(cur_frame, found_filtered, 1)
		vidwriter.write(cur_frame);
	vidreader.close();
	vidwriter.close()
	cv2.destroyAllWindows()
	print "Implemented Human Tracking.............   [Done]"
	
if __name__ == "__main__":
	import sys;
	_len = sys.argv.__len__()
	if _len == 1:
		raise NotEmptyError("Atleast one arguments must be provided")
	elif _len == 2:
		tracker(sys.argv[1])
	elif _len >= 3:
		tracker(sys.argv[1],sys.argv[2])	
