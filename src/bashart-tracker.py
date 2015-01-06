import cv2
import numpy as np
import features.interest_points as ip
import features.motion as motion
from sklearn.cluster import MeanShift, estimate_bandwidth


def tracker(path):
	#initialization for default value
	if path=='0':
		path=0;
	
	cap = cv2.VideoCapture(path)
	ip_method = ip.get_instace(ip.IPMethod.TOMASI);
	
	#FLANN Properties
	MIN_FRAMES_COUNT = 120
	SKIP_FRAMES = 60
	MIN_MERGE_FRAMES = 5;
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 10)
	search_params = dict(checks = 50)
	flann = cv2.FlannBasedMatcher(index_params, search_params)
	DO_RESIZE=False
	new_sz = (180,120)
	#Initialization of inputs
	frames =[];					#Frames
	kp = [];					#Key points
	all_matches = []; 			#All good matches
	match_count = [];			#match_count
	labels = [];
	frame_cnt=0;	
	
	print "Extracting frames...................."
	ret, prev_frame = cap.read()
	kp1,desc1 = ip_method.detectAndCompute(prev_frame);
	num_matches = np.zeros(kp1.__len__())
	
	#storing frames
	frames.append(prev_frame);
	kp.append(kp1)
	match_count.append(num_matches);
	
	
	while(cap.isOpened()):
		SKIP_FRAMES=SKIP_FRAMES-1;
		ret, prev_frame = cap.read()
		if not ret or SKIP_FRAMES<0:
			break;
	
	while(cap.isOpened()):
			
		ret, cur_frame = cap.read()
		if not ret:
			break;
		kp2,desc2 = ip_method.detectAndCompute(cur_frame);
		matches = flann.knnMatch(desc1,desc2,k=	2)		
		# Ratio test as per Lowe's paper 
		good_matches = []; distances = []
		for (m,n) in matches:
			
			if m.distance < 0.7*n.distance and m.distance > 4:
				good_matches.append(m);
				distances.append(m.distance);
				
		# Bashart's Displacement filtering
		mean = np.mean(distances); std = np.std(distances)
		good_matches[:] = [match for match in good_matches if abs(match.distance - mean) <  5 * std]
		kp1 = kp2; desc1 = desc2;
		
		num_matches = np.zeros(kp1.__len__())
		for match in good_matches:
			num_matches[match.trainIdx]=match_count[-1][match.queryIdx]+1
	
		all_matches.append(good_matches);		
		
		#storing frames
		frames.append(cur_frame);
		kp.append(kp1)
		match_count.append(num_matches);
		
		if frame_cnt > MIN_FRAMES_COUNT:
			break;
		frame_cnt = frame_cnt +1;
	cap.release()
	
	
	print "Labeling the keypoints................."
	max_label=0;
	MIN_POINTS_TO_CLUSTER = 20
	MAX_CLUSTERS = 100
	#Forward Labeling Pass
	for rng in xrange(0,MIN_MERGE_FRAMES+1):
		labels.append([-1]*kp[rng].__len__());
	for rng in xrange(MIN_MERGE_FRAMES+1,frame_cnt):
		motion_feats = []; feat_indices = [];
		labels.append([-1]*kp[rng].__len__());
		for match in all_matches[rng-1]:
			if match_count[rng-1][match.queryIdx]>=MIN_MERGE_FRAMES: 
				if labels[rng-1][match.queryIdx]==-1:
					src_pt = np.int32(kp[rng-1][match.queryIdx].pt)
					dst_pt = np.int32(kp[rng][match.trainIdx].pt)
					motion_feats.append(motion.get_features(src_pt,dst_pt));
					feat_indices.append(match.trainIdx)
				else :
					labels[rng][match.trainIdx]=labels[rng-1][match.queryIdx]
		
		if(motion_feats.__len__()>=MIN_POINTS_TO_CLUSTER):
			#Clustering mean-shift
			motion_feats = np.asarray(motion_feats)
			bandwidth = estimate_bandwidth(motion_feats, quantile=0.1,random_state=200)
			ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
			ms.fit(motion_feats);
			for idx,lbl in zip(feat_indices,ms.labels_):
				labels[rng][idx]=lbl+max_label;
			max_label = max(labels[rng])+1;
	
	
	random_colors = np.random.randint(256, size=(MAX_CLUSTERS, 3))
	print "Writing the video................."
	fourcc = cv2.cv.CV_FOURCC(*'XVID')
	w = prev_frame.shape[0]; h = prev_frame.shape[1]
	if DO_RESIZE:
		vidout = cv2.VideoWriter('out.avi',fourcc,20,new_sz)
	else:
		vidout = cv2.VideoWriter('out.avi',fourcc,20,(h,w))
	for frame_idx in xrange(MIN_MERGE_FRAMES*2,frame_cnt):
		cur_frame = frames[frame_idx];
		for rng in xrange(frame_idx-MIN_MERGE_FRAMES,frame_idx):
			for match in all_matches[rng-1]:
				if match_count[rng-1][match.queryIdx]>=MIN_MERGE_FRAMES \
						and not (labels[rng-1][match.queryIdx]==-1 or labels[rng-1][match.queryIdx]>=MAX_CLUSTERS):
					#print "i m not here"
					src_pt = np.int32(kp[rng-1][match.queryIdx].pt)
					dst_pt = np.int32(kp[rng][match.trainIdx].pt)
					color = tuple(random_colors[labels[rng-1][match.queryIdx]])
					cv2.line(cur_frame,tuple(src_pt),tuple(dst_pt),color,2);	
		if DO_RESIZE:
			cur_frame=cv2.resize(cur_frame,new_sz);
		vidout.write(cur_frame);
	vidout.release()
	cv2.destroyAllWindows()
	
if __name__ == "__main__":
	import sys;
	tracker(sys.argv[1]);
