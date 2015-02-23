import test;
if __name__ == "__main__":
	import sys;
	if sys.argv.__len__()<=1:
		test.run_all_test();
	elif sys.argv[1]=='sal':
		test.run_saliency_test(sys.argv[2:]);
	elif sys.argv[1]=='bgsub':
		test.run_bg_sub_test(sys.argv[2:]);
	else:
		raise NotImplementedError('Invalid method... supports only {sal,bgsub}')
