def run_saliency_test(argv):
	from test import test_saliency
	if argv.__len__()>=1:
		test_saliency.test(argv[0]);
	else:
		test_saliency.test();

def run_bg_sub_test(argv):
	from test import test_bg_sub
	if argv.__len__()>=1:
		test_bg_sub.test(argv[0]);
	else:
		test_bg_sub.test();


def run_all_test():
	run_saliency_test();
	run_bg_sub_test();

