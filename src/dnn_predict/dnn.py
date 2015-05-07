from dnn_predict import Predictor
from pythonDnn.utils.load_conf import load_dnn_spec
from pythonDnn.utils.utils import parse_activation
from pythonDnn.models.dnn import DNN3,DNN_Dropout

class DNNPredictor(Predictor):
	def __init__(self,model_config):
		super(DNNPredictor, self).__init__(model_config,'DNN');
		mlp_config = load_dnn_spec(self.model_config['nnet_spec'])
		activationFn = parse_activation(mlp_config['activation']);
		n_ins = model_config['n_ins']
		n_outs = model_config['n_outs']
		max_col_norm = mlp_config['max_col_norm']
		l1_reg = mlp_config['l1_reg']
		l2_reg = mlp_config['l2_reg']	
		adv_activation = mlp_config['adv_activation']
		hidden_layers_sizes = mlp_config['hidden_layers']
		do_dropout = mlp_config['do_dropout']
		
		if do_dropout:
			dropout_factor = dnn_config['dropout_factor']
			input_dropout_factor = dnn_config['input_dropout_factor']
			self.model = DNN_Dropout(numpy_rng=self.numpy_rng, theano_rng=self.theano_rng, 
							n_ins=n_ins, hidden_layers_sizes=hidden_layers_sizes, n_outs=n_outs,
							activation = activationFn, dropout_factor = dropout_factor,
							input_dropout_factor = input_dropout_factor, adv_activation = adv_activation,
							max_col_norm = max_col_norm, l1_reg = l1_reg, l2_reg = l2_reg)
		else:
			self.model = DNN(numpy_rng=self.numpy_rng,theano_rng=self.theano_rng, n_ins=n_ins, 
							hidden_layers_sizes=hidden_layers_sizes, n_outs=n_outs,
							activation = activationFn, adv_activation = adv_activation,
							max_col_norm = max_col_norm, l1_reg = l1_reg, l2_reg = l2_reg)
							
		self.__load_model__(self.model_config['input_file'],mlp_config['pretrained_layers']);
