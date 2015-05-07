import os,cv2,numpy as np;

from io_module.video_reader import VideoReader
from io_module.video_writer import VideoWriter
from utils import normalize,create_folder_structure_if_not_exists

def createVideo(datapath,group,outfileName,shape,duration=120,framerate=20,vidLen=32):
	vidOutFileName = outfileName + 'test.avi'
	labelOutFileName = outfileName + 'test.txt'
	values = [];
	create_folder_structure_if_not_exists(vidOutFileName);
	with open(datapath+os.sep+group+".txt",'r') as fp:
		for line in fp:
			values.extend([line.split()]);
	numClass = len(values);
	numExamples = (duration*framerate)/(vidLen);
	randomNumbers = np.random.random_integers(0,numClass-1,numExamples);
	fileWriter = open(labelOutFileName,'w');
	vidWriter = VideoWriter(vidOutFileName,shape[0],shape[1]);
	vidWriter.build();
	for idx,random_idx in enumerate(randomNumbers):
		print 'Creating UCF 50 video ... {0}%\r'.format((idx*100/numExamples)),
		classDetails = values[random_idx];
		classPath = datapath + os.sep + classDetails[0] + os.sep
		classExamples = [os.path.join(classPath, _file) for _file in os.listdir(classPath)
			if os.path.isfile(os.path.join(classPath, _file))];
		chosenExample = classExamples[np.random.randint(len(classExamples))];
		#print chosenExample
		vidreader = VideoReader(chosenExample,None);
		num_frames = vidreader.frames;
		cnt,frames = vidreader.read(np.random.randint(num_frames-vidLen),vidLen);
		fileWriter.writelines("%d\n" % int(item) for item in [classDetails[2]]*cnt) 
		for frame in frames:
			vidWriter.write(frame);
	vidWriter.close();	
	fileWriter.close();
		
if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description='Creating a sample UCF50 video')
	parser.add_argument("-datapath",nargs='?',help = "input path of data",default="/media/sudhar/Data/mtech/SemIV/data/UCF50/UCF50/");
	parser.add_argument("-group",nargs='?',help = "category of video",default="ALL");
	parser.add_argument("-out",nargs='?',help = "output path of the label",default="sample_test_ucf/");
	parser.add_argument("-shape",nargs='*',help = "shape of the video",default=[320,240],type=int);
	parser.add_argument("-fps",nargs='?',help = "framerate of created video",default=20,type=int);
	parser.add_argument("-vidseq",nargs='?',help = "length of individual video sequence",default=40,type=int);
	parser.add_argument("-duration",nargs='?',help = "duration of output video",default=60,type=int);
	args = parser.parse_args();
	out = args.out + os.sep + args.group + os.sep
	createVideo(args.datapath,args.group,out,args.shape,args.duration,args.fps,args.vidseq);
	print 'Creating UCF 50 video [DONE]'
		
		

