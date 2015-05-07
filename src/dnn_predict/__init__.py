import numpy as np

from pythonDnn.utils.load_conf import load_model
from theano.tensor.shared_randomstreams import RandomStreams

def get_instance(model_config_file):
	modelConfig = load_model(model_config_file);
	nnetType = modelConfig ['nnetType']
	if nnetType == 'CNN':
		from dnn_predict.cnn import CNNPredictor as PredictorInstance
	elif nnetType == 'CNN3D':
		from dnn_predict.cnn3d import CNN3dPredictor as PredictorInstance
	elif nnetType == 'DNN':
		from dnn_predict.dnn import DNNPredictor as PredictorInstance
	else :
		raise NotImplementedError('Unknown nnet Type')
		exit(2)
	return PredictorInstance(modelConfig);

class Predictor(object):
	def __init__(self,model_config,model_type):
		if type(model_config) is dict:
			self.model_config = model_config
		else :
			self.model_config = load_model(model_config,model_type)
		self.model_type = model_type
		
		#generating Random
		self.numpy_rng = np.random.RandomState(model_config['random_seed'])
		self.theano_rng = RandomStreams(self.numpy_rng.randint(2 ** 30))
		
		#model properties
		self.batch_size = model_config['batch_size'];
		self.input_shape = model_config['input_shape'];
		self.batch_shape = [self.batch_size];
		self.batch_shape.extend(self.input_shape)

	def get_score(self,feats):
		assert(self.batch_shape==list(feats.shape)),"Feats are not in proper shape"
		return self.scoreFn(feats);
	
	def __load_model__(self,model_file,num_pretrained_layers):
		try:
			self.model.load(filename=model_file,max_layer_num = num_pretrained_layers, withfinal=True)
			self.scoreFn = self.model.getScoreFunction()	#setting score fn
		except KeyError, e:
			print 'IGNORING : Pretrained network missing in working directory, skipping model loading'
		except IOError, e:
			raise IOError('Model cannot be initialize from input file ')
			exit(2)	
			
	
	
	
	
	
	




