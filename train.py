from config import Config
from Controller import Controller
from network_manager import NetworkManager
import pickle
import sys

#store previously sample networks for experience replay and to avoid redundant training of the same network
reward_dict = dict()
completed_action_list = []

def convertToString(action):
	string_val = ''
	for layer in action:
		for feat in layer:
			# print('feat : ',feat)
			string_val += (feat+':'+str(layer[feat]))
	return string_val


def main():
	config = Config()
	controller = Controller(config)
	net_man = NetworkManager(epochs=config.CNN_epochs,\
	 child_batchsize=128, lr=config.CNN_lr, momentum=0.9, acc_beta=0.8)

	for i in range(config.max_no_of_sampled_networks):
		print('sampling ',i,'th network')
		action = controller.get_action()
		print(action)
		sys.stdout.flush()
		while convertToString(action) in reward_dict:
			action = controller.get_action()

		acc,flops,params = net_man.get_peformance_statistics(action)

		reward_dict[convertToString(action)] = (acc,flops,params)
		completed_action_list.append(action)
		with open('completed_action_list.pickle', 'wb') as handle:
		    pickle.dump(completed_action_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
		with open('reward_dict.pickle', 'wb') as handle:
		    pickle.dump(reward_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

		controller.train_step(reward_dict,completed_action_list)
		controller.save_model()





	
if __name__== "__main__":
	main()
