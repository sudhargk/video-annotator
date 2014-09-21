import numpy as np
import cv2


def vidread(path):
	cap = cv2.VideoCapture(path)
	print cap.isOpened();
	while(cap.isOpened()):
		ret, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		cv2.imshow('frame',gray)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	import sys;
	vidread(sys.argv[1]);


