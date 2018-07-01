
import time, math, os
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import pickle
import torch.nn as nn
from torchvision import models
from resnet import resnet50
from huffman.huffman import HuffmanCoding
assert KMeans, MiniBatchKMeans


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
    def save(self, model, dir, cluster= 256):
        model = model.cpu().eval()
        if not os.path.exists(dir):
            os.makedirs(dir)
        state_dict = model.state_dict()
        weight = {}
        kmeans= MiniBatchKMeans(n_clusters=cluster, random_state=0)

        layer_list = []
        for key in state_dict:
            length = state_dict[key].numel()
            print('\rsaving model layer {}, size {}...     '.format(key, length),end='')
            if state_dict[key].numel()<= cluster:
                weight[key] = state_dict[key].numpy()
            else:
                params = state_dict[key].view(-1,1).numpy()
                kmeans.fit(params)
                quantized_table = kmeans.cluster_centers_.reshape((-1,))
                quantized_weight = kmeans.labels_.astype(np.uint8)
                layer_list.append(list(quantized_weight))
                weight[key] = (quantized_table, len(layer_list)-1)
        with open('{}/table.pt'.format(dir), 'wb') as f:
                pickle.dump(weight, f, protocol=pickle.HIGHEST_PROTOCOL)
        huffman = HuffmanCoding()
        huffman.compress(layer_list, '{}/weight.pt'.format(dir), '{}/code.pt'.format(dir))
    def load(self, dir, model):
        with open('{}/table.pt'.format(dir), 'rb') as f:
            weight= pickle.load(f)
        huffman = HuffmanCoding()
        layer_list=huffman.decompress('{}/weight.pt'.format(dir), '{}/code.pt'.format(dir))
        state_dict = {}
        model_dict = model.state_dict()
        for key in weight:
            print('\rsaving model layer {}...     '.format(key),end='')
            if isinstance(weight[key], np.ndarray):
                print(weight[key].shape)
                state_dict[key] = torch.from_numpy(weight[key])
            else:
                quantized_table = weight[key][0]
                quantized_weight = np.array(layer_list[weight[key][1]])
                print(quantized_weight.shape)
                state_dict[key] = torch.from_numpy(quantized_table[quantized_weight]).view_as(model_dict[key])
        model.load_state_dict(state_dict)
        return model

class CNN_squeezenet(nn.Module):
    def __init__(self, pretrained=False):
        super(CNN_squeezenet, self).__init__()
        self.features = models.squeezenet1_1(pretrained=pretrained).features
        self.final_conv = nn.Conv2d(512, 2630, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            self.final_conv,
            nn.ReLU(inplace=True),
            nn.AvgPool2d((13, 11), stride=1)
        )
        self._initialize_weights()
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        return x
    def save(self, path):
        torch.save(self,path)
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is self.final_conv:
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def _initialize_weights_vgg(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    def _initialize_weights_densenet(self):
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

class CNN_vgg16(nn.Module):
    def __init__(self, pretrained=False):
        super(CNN_vgg16, self).__init__()
        self.conv = models.vgg16_bn(pretrained=pretrained).features
        self.fc = nn.Sequential(
            nn.Linear(512 * 6 * 5, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 2360),
        )
        self._initialize_weights_vgg()
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x
    def save(self, path):
        torch.save(self,path)
    def _initialize_weights_vgg(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    def _initialize_weights_densenet(self):
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    dm = DataManager()
    '''
    model = torch.load('./model/resnet50.pt')
    dm.save(model, './model/resnet')
    '''
    model = resnet50()
    torch.save(dm.load('./model/resnet', model),
            './model/model_recover.pt')
