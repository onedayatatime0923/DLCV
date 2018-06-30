
import os, sys, time, math
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
assert os and np and nn and F and optim and Data and torchvision and io and skimage
assert pickle and sys


class DataManager():
    def timeSince(self,since, percent):
        now = time.time()
        s = now - since
        es = s / (percent)
        rs = es - s
        return '%s (- %s)' % (self.asMinutes(s), self.asMinutes(rs))
    def asMinutes(self,s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)
    def save(self, model, path, cluster= 256):
        model = model.cpu()
        state_dict = model.state_dict()
        weight = {}
        kmeans= KMeans(n_clusters=cluster, random_state=0, n_jobs=4)

        #ResNet-34: 182
        #VGG-16: 92
        #50,100,150,182
        for key in state_dict:
            print(key)
            if state_dict[key].numel()< cluster:
                continue
            else:
                size = state_dict[key].size()
                params = state_dict[key].view(-1,1).numpy()
                kmeans.fit(params)
                quantized_table = list(kmeans.cluster_centers_.reshape((-1,)))
                quantized_weight = kmeans.labels_.reshape(size)
                weight[key] = (quantized_table, quantized_weight)

        with open(path, 'wb') as f:
                pickle.dump(weight, f, protocol=pickle.HIGHEST_PROTOCOL)


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
                        #       for tensor_2 in tensor_1:
                        #               for tensor_3 in tensor_2:
                        #                       for tensor_4 in tensor_3:
                        #                               tensor_4.data = torch.from_numpy(layer_dict[tensor_4.numpy().item()])
                                                        
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
                        #       for tensor_2 in tensor_1:
                        #               for tensor_3 in tensor_2:
                        #                       for tensor_4 in tensor_3:
                        #                               tensor_4.data = torch.from_numpy(layer_dict[tensor_4.numpy().item()])
                                                        
                        layer_weight = layer_weight.squeeze()
                        print (layer_weight.size())
                        reconstruct_state[key] = layer_weight
                        model.state_dict()[key].copy_(layer_weight)
                else:
                        reconstruct_state[key] = state_dict[key]

        #torch.save(reconstruct_state, 'recon_state.pt')
        #model.state_dict().copy_(reconstruct_state)
        torch.save(model, 'recon_model.pt')
            '''

if __name__ == '__main__':
    dm = DataManager()
    model = torch.load('./model/squeezenet.pt').eval()
    print(model)
    dm.save(model, './')
