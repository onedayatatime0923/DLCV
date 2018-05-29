
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torchvision
from torchvision import models
import collections, os, skimage.transform, csv
from skvideo import io
assert torch and Dataset and DataLoader

class DataManager():
    def getVideoList(self,data_path):
        '''
        @param data_path: ground-truth file path (csv files)

        @return: ordered dictionary of videos and labels {'Action_labels', 'Nouns', 'End_times', 'Start_times', 'Video_category', 'Video_index', 'Video_name'}
        '''
        result = {}

        with open (data_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                for column, value in row.items():
                    result.setdefault(column,[]).append(value)

        od = collections.OrderedDict(sorted(result.items()))
        return od
    def readShortVideo(self,video_path, video_category, video_name, downsample_factor=12, rescale_factor=1):
        '''
        @param video_path: video directory
        @param video_category: video category (see csv files)
        @param video_name: video name (unique, see csv files)
        @param downsample_factor: number of frames between each sampled frame (e.g., downsample_factor = 12 equals 2fps)
        @param rescale_factor: float of scale factor (rescale the image if you want to reduce computations)

        @return: (T, H, W, 3) ndarray, T indicates total sampled frames, H and W is heights and widths
        '''

        filepath = video_path + '/' + video_category
        filename = [file for file in os.listdir(filepath) if file.startswith(video_name)]
        video = os.path.join(filepath,filename[0])

        videogen = io.vreader(video)
        frames = []
        for frameIdx, frame in enumerate(videogen):
            if frameIdx % downsample_factor == 0:
                frame = skimage.transform.rescale(frame, rescale_factor, mode='constant', preserve_range=True).astype(np.uint8)
                frames.append(frame)
            else:
                continue

        return np.array(frames).astype(np.uint8)
    def get_data(self, video_path, tag_path, save_path= None):
        if save_path != None:
            if os.path.isfile(save_path[0]) and os.path.isfile(save_path[1]):
                x= np.load(save_path[0])
                y= np.load(save_path[1])
                return x,y
        file_dict=(self.getVideoList(tag_path))
        x, y=[], []
        for i in range(len(file_dict['Video_index'])):
            x.append(self.readShortVideo(video_path, file_dict['Video_category'][i],file_dict['Video_name'][i]))
            y.append(int(file_dict['Action_labels'][i]))
            print('\rreading image from {}...{}'.format(video_path,i),end='')
        print('\rreading image from {}...finished'.format(video_path))
        if save_path != None:
            np.save(save_path[0],np.array(x))
            np.save(save_path[1],np.array(y))

class ResNet50_feature(nn.Module):
    def __init__(self):
        super(ResNet50_feature, self).__init__()
        original_model = models.resnet50(pretrained=True)
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool
        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4
        self.avgpool = original_model.avgpool
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class ImageDataset(Dataset):
    def __init__(self, image, label):
        self.image = image
        self.label = label
    def __getitem__(self, i):
        index= i// self.flip_n 
        flip = bool( i % self.flip_n )

        if self.mode=='vae':
            if flip == True: x= np.flip(self.image[index],2).copy()
            else: x= self.image[index]
            x=torch.FloatTensor(x)/255
            if self.rotate: x=torchvision.transforms.RandomRotation(5)
            if not isinstance(self.c, np.ndarray) : return x
            c=torch.FloatTensor(self.c[index][:])
            return x, c
        elif self.mode=='gan':
            if flip == True: x= np.flip(self.image[index],2).copy()
            else: x= self.image[index]
            x=(torch.FloatTensor(x)- 127.5)/127.5
            if self.rotate>0: x=torchvision.transforms.RandomRotation(5)
            if not isinstance(self.c, np.ndarray) : return x
            c=torch.FloatTensor(self.c[index][:])
            return x, c
        else: raise ValueError('Wrong mode.')
    def __len__(self):
        return len(self.image)*self.flip_n
