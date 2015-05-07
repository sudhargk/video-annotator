from dnn_predict import Predictor
from pythonDnn.utils.load_conf import load_conv_spec
from pythonDnn.utils.utils import parse_activation
from pythonDnn.models.cnn3d import CNN3D


class CNN3dPredictor(Predictor):
	def __init__(self,model_config):
		super(CNN3dPredictor, self).__init__(model_config,'CNN3d');
		conv_config,conv_layer_config,mlp_config = load_conv_spec(self.model_config['nnet_spec'],
														self.batch_size,
														self.model_config['input_shape'])
		activationFn = parse_activation(mlp_config['activation']);
		
		self.model = CNN3D(self.numpy_rng,self.theano_rng,conv_layer_configs = conv_layer_config, 
			batch_size = self.batch_size, n_outs= self.model_config['n_outs'],
			hidden_layer_configs=mlp_config,hidden_activation = activationFn,
			l1_reg = mlp_config['l1_reg'],l2_reg = mlp_config['l1_reg'],
			max_col_norm = mlp_config['max_col_norm'])
		
		self.__load_model__(self.model_config['input_file'],mlp_config['pretrained_layers']);

