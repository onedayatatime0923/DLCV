import os
import sys
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data as Data
import torchvision 

from skimage import io
import skimage.transform

from sklearn.cluster import KMeans

import pickle

readdata = 0
batch_size = 64
num_epoch = 20


if(sys.argv[1] == 'c'):
	timestr = time.strftime("%m%d_%H%M", time.localtime())
	print ('building model...')
	model = torch.load(sys.argv[2]).eval()

	keys = []
	state_dict = {}
	weight_dict = {}
	#target_layer = []
	'''
	for key, value in model.state_dict().items():
		keys.append(key)

	print (keys)

	with open('keys_vgg16', 'wb') as f:
		pickle.dump(keys, f)
	'''
	with open('keys_vgg16', 'rb') as f:
	 	keys = pickle.load(f)

	with open('weight_dict_vgg16.pt', 'rb') as f:
	 	weight_dict = pickle.load(f)

	state_dict = torch.load('state_dict_vgg16.pt')

	#ResNet-34: 182
	#VGG-16: 92
	#50,100,150,182
	for i in range(50,75):
		key = keys[i]
		size = model.state_dict()[key].size()
		if len(list(size)) > 1 :
		#if key[9:13] == 'conv' or key[0:4] == 'conv' or key == 'fc.weight':
			print (key)
			size = model.state_dict()[key].size()
			print (size)
			params = model.state_dict()[key].view(-1,1).cpu().numpy()
			kmeans = KMeans(n_clusters=256, random_state=0, n_jobs=4).fit(params)
			quantized_weight = torch.from_numpy(kmeans.labels_ ).view(size)
			quantized_weight = torch.tensor(quantized_weight, dtype=torch.uint8)
			quantized_dict = dict(zip(range(256), kmeans.cluster_centers_))
			weight_dict[key] = quantized_dict
			state_dict[key] = quantized_weight
		else:
			state_dict[key] = model.state_dict()[key]


		
	with open('weight_dict_vgg16.pt', 'wb') as f:
		pickle.dump(weight_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

	torch.save(state_dict, 'state_dict_vgg16.pt')

if (sys.argv[1] == 'r'):
	print ('building model...')
	model = torch.load(sys.argv[2]).eval()
	state_dict = torch.load('state_dict.pt')
	reconstruct_state = {}
	with open('weight_dict.pt', 'rb') as f:
		weight_dict = pickle.load(f)

	with open('keys', 'rb') as f:
		keys = pickle.load(f)

	for i in range(len(keys)):
		key = keys[i]
		if key[9:13] == 'conv' or key[0:4] == 'conv':
			print (key)
			layer_dict = weight_dict[key]
			layer_weight = state_dict[key].float()
			size = list(layer_weight.size())
			layer_weight = layer_weight.view(size[0],size[1],size[2],size[3], 1)

			for i in range(size[0]):
				for j in range(size[1]):
					for k in range(size[2]):
						for l in range(size[3]):

							new_weight = torch.tensor(layer_dict[layer_weight[i][j][k][l].numpy().item()])
							#print (new_weight.size())
							layer_weight[i][j][k][l].copy_(new_weight)
			# for tensor_1 in layer_weight:
			# 	for tensor_2 in tensor_1:
			# 		for tensor_3 in tensor_2:
			# 			for tensor_4 in tensor_3:
			# 				tensor_4.data = torch.from_numpy(layer_dict[tensor_4.numpy().item()])
							
			layer_weight = layer_weight.squeeze()
			print (layer_weight.size())
			reconstruct_state[key] = layer_weight
			model.state_dict()[key].copy_(layer_weight)
		elif key == 'fc.weight':
			print (key)
			layer_dict = weight_dict[key]
			layer_weight = state_dict[key].float()
			size = list(layer_weight.size())
			layer_weight = layer_weight.view(size[0],size[1], 1)

			for i in range(size[0]):
				for j in range(size[1]):
					new_weight = torch.tensor(layer_dict[layer_weight[i][j].numpy().item()])
							#print (new_weight.size())
					layer_weight[i][j].copy_(new_weight)
			# for tensor_1 in layer_weight:
			# 	for tensor_2 in tensor_1:
			# 		for tensor_3 in tensor_2:
			# 			for tensor_4 in tensor_3:
			# 				tensor_4.data = torch.from_numpy(layer_dict[tensor_4.numpy().item()])
							
			layer_weight = layer_weight.squeeze()
			print (layer_weight.size())
			reconstruct_state[key] = layer_weight
			model.state_dict()[key].copy_(layer_weight)
		else:
			reconstruct_state[key] = state_dict[key]

	#torch.save(reconstruct_state, 'recon_state.pt')
	#model.state_dict().copy_(reconstruct_state)
	torch.save(model, 'recon_model.pt')


	
		
	


