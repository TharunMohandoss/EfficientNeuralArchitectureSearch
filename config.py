
class Config:
	def __init__(self):
		#define what features of the cnn we predict per layer
		self.layer_feature_list = ['kernel_size','no_of_filters','stride','continue']#prediction order
		self.feature_values_dict = dict()
		self.feature_values_dict['kernel_size'] = [1,3,5,7]
		self.feature_values_dict['no_of_filters'] = [4,8,16,32]
		self.feature_values_dict['stride'] = [1,2,4]
		self.feature_values_dict['continue'] = [0,1]

		#maximum layers
		self.max_no_of_layers = 10
		self.state_feature_size = 35
		self.controller_learning_rate = 0.01
		self.max_no_of_sampled_networks = 10000
		self.batch_size_controller = 10
		self.CNN_epochs = 50
		self.CNN_lr = 0.01
		self.entropy_weight = 200




		