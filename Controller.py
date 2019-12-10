import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
import torch
import pickle
import sys

def convertToString(action):
	string_val = ''
	for layer in action:
		for feat in layer:
			# print('feat : ',feat)
			string_val += (feat+':'+str(layer[feat]))
	return string_val

def getRewardFromStatistics(acc,flops,params):
	reward = acc - flops*1.0/(1000000000.0) - params*1.0/4000000.0
	return reward

class NASCell_Layer(nn.Module):
	def __init__(self,output_size,state_size):
		super(NASCell_Layer, self).__init__()
		self.dense_probability_prediction = nn.Linear(2*state_size,output_size)
		self.dense_next_state = nn.Linear(2*state_size,state_size)

	def forward(self,previous_state,previous_prediction):
		#concatenate the vectors to pass forward
		concatenated_vector = torch.cat((previous_state,previous_prediction),1)
		# print('shape of concatenated : ',concatenated_vector.shape)
		#predict next state
		next_state = F.relu(self.dense_next_state(concatenated_vector))
		#predict the probailities of picking features
		predicted_probabilities = F.softmax(self.dense_probability_prediction(concatenated_vector),dim=1)

		return next_state,predicted_probabilities

class NASCell(nn.Module):
	def __init__(self,config,embeddings):
		super(NASCell, self).__init__()
		self.config = config
		self.embeddings_dict = embeddings
		self.layer_list = []

		#for each type of feature, create a NASCell layer instance and add them to the list
		for feature in self.config.layer_feature_list:
			current_layer = NASCell_Layer(\
				len(self.config.feature_values_dict[feature]),self.config.state_feature_size)
			self.layer_list.append(current_layer)
		self.layer_list = nn.ModuleList(self.layer_list)

		#define initial state and input_action values as trainable variables
		self.initial_state = Variable(torch.zeros(1, self.config.state_feature_size).cuda(), requires_grad=True)
		self.initial_prev_action = Variable(torch.zeros(1,self.config.state_feature_size).cuda(), requires_grad=True)




	def sample_action(self):
		sampled_action = []
		state,prev_action = self.initial_state,self.initial_prev_action
		for i in range(self.config.max_no_of_layers):
			layer_specification_dict = dict()
			for j in range(len(self.config.layer_feature_list)):
				current_feature = self.config.layer_feature_list[j]

				state,action_probabilities = self.layer_list[j](state,prev_action)
				action_probabilities_numpy = action_probabilities.detach().cpu().numpy()

				sample_random_val = random.uniform(0,1)
				# print('sample_random_val : ',sample_random_val)
				total_val = 0
				chosen_value = 0
				for i in range(len(action_probabilities_numpy[0])):
					total_val += action_probabilities_numpy[0][i]
					# print('total_val : ',total_val)
					if sample_random_val<=total_val:
						break
					# print('incrementing chosen_value to ',chosen_value,', max : ',len(action_probabilities_numpy[0]))
					chosen_value += 1
				# print('action_probabilities_numpy : ',action_probabilities_numpy)
				# print('chosen_value : ',chosen_value)
				# print('feature_values_dict[current_feature] : ',self.config.feature_values_dict[current_feature])
				chosen_action = self.config.feature_values_dict[current_feature][chosen_value]
				layer_specification_dict[current_feature] = chosen_action
			sampled_action.append(layer_specification_dict)
			if(layer_specification_dict['continue']==0):
				break

		return sampled_action

	def getProbability(self,action):
		state,prev_action = self.initial_state,self.initial_prev_action
		total_probability = torch.ones(1,1).cuda()
		log_probability_sum = torch.zeros(1,1).cuda()
		for i in range(len(action)):
			for j in range(len(self.config.layer_feature_list)):	
				current_feature = self.config.layer_feature_list[j]
			
				state,action_probabilities = self.layer_list[j](state,prev_action)
				chosen_action_index =\
				 self.config.feature_values_dict[current_feature].index(action[i][current_feature])
				total_probability = total_probability*action_probabilities[0][chosen_action_index]
				log_probability_sum += torch.log(action_probabilities[0][chosen_action_index])

		return total_probability,log_probability_sum





class Controller:
	def __init__(self,config):
		self.config  = config
		self.embeddings = self.init_random_embeddings()
		self.nascell = NASCell(config,self.embeddings).cuda()

		self.optimizer = torch.optim.SGD(self.nascell.parameters(),\
		 lr=self.config.controller_learning_rate,momentum=0.9)
		self.optimizer.zero_grad()		

		#running mean
		with open('running_mean.txt','r') as fp:
			self.running_mean = float(fp.readline().strip())

	def init_random_embeddings(self):
		embeddings_dict = dict()
		for feature in self.config.layer_feature_list:
			embeddings_dict[feature] = dict()
			for val in self.config.feature_values_dict[feature]:
				embeddings_dict[feature][val] = np.random.rand(1,self.config.state_feature_size)*2-1

		return embeddings_dict

	def get_action(self):
		return self.nascell.sample_action()

	def train_step(self,reward_dict,completed_action_list):
		if len(completed_action_list) > self.config.batch_size_controller:
			all_elements = [i for i in range(len(completed_action_list))]
			elements_to_take = random.sample(all_elements,self.config.batch_size_controller)


			total_loss = torch.zeros(1,1).cuda()
			avg_rew = 0
			for element in elements_to_take:
				curent_action = completed_action_list[element]
				statistics = reward_dict[convertToString(curent_action)]
				acc,flops,params = statistics[0],statistics[1],statistics[2]
				rew = getRewardFromStatistics(acc,flops,params) 
				current_reward = rew - self.running_mean
				avg_rew += rew
				probability,log_probability = self.nascell.getProbability(curent_action)
				total_loss += -log_probability*current_reward

			avg_rew = avg_rew/len(elements_to_take)
			self.running_mean = 0.99*self.running_mean + 0.01*avg_rew
			self.optimizer.zero_grad()
			total_loss.backward()
			self.optimizer.step()
		else:
			print('skipping train step due to lack of number of examples')

	def save_model(self):
		torch.save(self.nascell.state_dict(), './controller.pth')
		with open('embeddings.pickle', 'wb') as handle:
		    pickle.dump(self.embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)		
		#running mean
		with open('running_mean.txt','w') as fp:
			fp.write(str(self.running_mean))

	def load_model(self):
		self.nascell.load_state_dict(torch.load('./controller.pth'))
		with open('embeddings.pickle', 'rb') as handle:
		    self.embeddings = pickle.load(handle)




		