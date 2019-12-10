import numpy as np 
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models import resnet50
from thop import profile
import sys

class ChildNetwork(nn.Module):
	'''

	Creates a child network given the actions

	'''
	def __init__(self, action):
		super(ChildNetwork, self).__init__()
		self.action = action
		self.conv_layers = []
		in_c = 3
		self.dense_input_pixels = 32*32
		for dct in action:
			# print(dct)
			no_of_filters = int(dct['no_of_filters'])
			kernel_size = int(dct['kernel_size'])
			strideVal = int(dct['stride'])
			self.conv_layer = nn.Conv2d(in_channels=in_c, out_channels=no_of_filters, kernel_size=kernel_size, stride=strideVal, padding=int((kernel_size-1)/2))
			self.conv_layers.append(self.conv_layer)
			self.conv_layers.append(nn.ReLU())
			in_c = no_of_filters
			self.dense_input_pixels = max(1,int(self.dense_input_pixels/(strideVal*strideVal)))
		self.conv_layers = nn.ModuleList(self.conv_layers)
		self.dense = nn.Linear(self.dense_input_pixels*int(self.action[-1]['no_of_filters']), 10)

	def forward(self, x):
		for layer in self.conv_layers:
			x = layer(x)
		#conv_layers.append(nn.AvgPool2d())
		#print(x.shape)
		x = x.view(-1, self.dense_input_pixels*int(self.action[-1]['no_of_filters'])) 
		x = self.dense(x)
		#print(x.shape)
		#x = nn.Softmax(x)
		return x

class NetworkManager:
	'''
    
    Helper class to create a child network and train it to get reward,
    given the actions viz. child network parameters
    
    '''
	def __init__(self, epochs=5, child_batchsize=128, lr=0.001, momentum=0.9, acc_beta=0.8):
		'''
		Args:
		    epochs: number of epochs to train the child network
		    child_batchsize: batchsize for training the child network
		    lr: for SGD
		    momentum: ''
		'''
	# initialize training hyperparameters
		self.epochs = epochs
		self.batchsize = child_batchsize
		self.lr = lr
		self.momentum = momentum

		# parameters for updating moving accuracy
		self.beta = acc_beta
		self.exp_weighted_accuracy = 0.0

		# creating dataset
		transform = transforms.Compose(
		[transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
		self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
				                 download=True, transform=transform)
		self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batchsize,
		                                          shuffle=True, num_workers=2)
		self.testset = torchvision.datasets.CIFAR10(root='./data', train=False,
		                                       download=True, transform=transform)
		self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.batchsize,
		                                         shuffle=False, num_workers=2)


	def get_peformance_statistics(self, action):
		# print('action inside manager : ',action)
		model = ChildNetwork(action).cuda()
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.Adam(model.parameters(), lr = self.lr)

		max_accuracy = 0
		for epoch in range(self.epochs):  # loop over the dataset multiple times
			if epoch == 10:
				for param_group in optimizer.param_groups:
					param_group['lr'] = self.lr/10
					print('changed lr to : ',self.lr/10)

			running_loss = 0.0
			for i, data in enumerate(self.trainloader, 0):
				# get the inputs; data is a list of [inputs, labels]
				inputs, labels = data
				inputs = inputs.cuda()
				labels = labels.cuda()
				#print(inputs.shape)
				#print(labels.shape)
				# zero the parameter gradients
				optimizer.zero_grad()
				# forward + backward + optimize
				outputs = model(inputs)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()	
				# print statistics
				running_loss += loss.item()
				if i % 50 == 49:	# print every 50 mini-batches
					print('[%d, %5d] loss: %.3f' %
					(epoch + 1, i + 1, running_loss / 50))
					sys.stdout.flush()
					running_loss = 0.0

			with torch.no_grad():
				total = 0
				correct = 0
				for data in self.testloader:
					images, labels = data
					images = images.cuda()
					labels = labels.cuda()
					outputs = model(images)
					_, predicted = torch.max(outputs.data, 1)
					total += labels.size(0)
					correct += (predicted == labels).sum().item()
			acc = (1.0*correct)/total
			max_accuracy = max(max_accuracy,acc)
			print("epoch ",epoch,"Accuracy: ",  acc)
		print('==> Finished training the child network')
		with torch.no_grad():
			total = 0
			correct = 0
			for data in self.testloader:
				images, labels = data
				images = images.cuda()
				labels = labels.cuda()
				outputs = model(images)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
		acc = (1.0*correct)/total
		print("Accuracy: ",  max_accuracy)

		input_arbit = torch.randn(1,3,32,32).cuda()
		flops, params = profile(model, inputs=(input_arbit, ))
		print('flops : ',flops*1.0/(1000000000.0))
		print('params : ',params*1.0/4000000.0)
		print('rew : ',max_accuracy - flops*1.0/(1000000000.0) - params*1.0/4000000.0)

		# reward =  acc - self.exp_weighted_accuracy
		# self.exp_weighted_accuracy = self.beta * self.exp_weighted_accuracy + (1 - self.beta) * acc
		return max_accuracy,flops,params

if __name__ == '__main__':
	net_man = NetworkManager(epochs=1, child_batchsize=128, lr=0.001, momentum=0.9, acc_beta=0.8)
	# li = []
	# li += [{'no_of_filters': '64', 'kernel_size': '7'}]
	# li += [{'no_of_filters': '64', 'kernel_size': '7'}]
	# li += [{'no_of_filters': '64', 'kernel_size': '7'}]
	# li += [{'no_of_filters': '64', 'kernel_size': '7'}]
	# li += [{'no_of_filters': '64', 'kernel_size': '7'}]
	li = [{'kernel_size': 7, 'no_of_filters': 8}, {'kernel_size': 1, 'no_of_filters': 8}, {'kernel_size': 7, 'no_of_filters': 16}, {'kernel_size': 3, 'no_of_filters': 8}, {'kernel_size': 3, 'no_of_filters': 4}]
	net_man.get_peformance_statistics(li)