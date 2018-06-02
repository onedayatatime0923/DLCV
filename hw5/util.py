
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from torchvision import models
from tensorboardX import SummaryWriter 
import collections, os, skimage.transform, csv, time, math, random
from skvideo import io
from skimage.transform import resize
from PIL import Image
import matplotlib.pyplot as plt
assert torch and F and skimage and plt and resize

class DataManager():
    def __init__(self, path=None):
        if path== None: self.writer=None
        else: self.tb_setting(path)
    def tb_setting(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        for f in os.listdir(directory): 
            os.remove('{}/{}'.format(directory,f))
        self.writer = SummaryWriter(directory)
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
                #frame= resize(frame,(224, 224, 3))
                frame = skimage.transform.rescale(frame, rescale_factor, mode='constant', preserve_range=True).astype(np.uint8)
                frames.append(frame)
            else:
                continue

        return np.array(frames).astype(np.uint8)
    def get_data(self, video_path, tag_path, save_path= None, batch_size=32, shuffle= True):
        if save_path != None and os.path.isfile(save_path[0]) and os.path.isfile(save_path[1]):
                x= np.load(save_path[0])
                y= np.load(save_path[1])
                return ImageDataLoader(x,y, batch_size=batch_size, shuffle=shuffle)

        file_dict=(self.getVideoList(tag_path))
        x, y=[], []
        for i in range(len(file_dict['Video_index'])):
            x.append(self.readShortVideo(video_path, file_dict['Video_category'][i],file_dict['Video_name'][i]))
            y.append(int(file_dict['Action_labels'][i]))
            print('\rreading image from {}...{}'.format(video_path,i),end='')
        x= np.array(x)
        y= np.array(y)
        print('\rreading image from {}...finished'.format(video_path))
        if save_path != None:
            np.save(save_path[0],x)
            np.save(save_path[1],y)
        return ImageDataLoader(x,y, batch_size=batch_size, shuffle=shuffle)
    def train(self, model, dataloader, epoch, lr=1E-5, print_every= 10):
        start= time.time()
        model.train()
        
        optimizer = torch.optim.Adam(list(model.parameters())+[model.hidden],lr=1E-5)
        criterion= nn.CrossEntropyLoss()
        total_loss= 0
        batch_loss= 0
        total_correct= 0
        batch_correct= 0
        
        data_size= len(dataloader)
        for b, (x, i, y) in enumerate(dataloader):
            batch_index=b+1
            x, i, y= Variable(x).cuda(), Variable(i).cuda() , Variable(y).cuda()
            output= model(x,i)
            loss = criterion(output,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # loss
            batch_loss+= float(loss)
            total_loss+= float(loss)* len(x)
            # accu
            pred = output.data.argmax(1) # get the index of the max log-probability
            #print(y)
            #print(pred)
            correct = pred.eq(y.data).long().cpu().sum()
            batch_correct += correct/ len(x)
            total_correct += correct
            if batch_index% print_every== 0:
                print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)] | Loss: {:.6f} | Accu: {}% | Time: {}  '.format(
                            epoch , batch_index*len(x), data_size, 100. * batch_index*len(i)/ data_size,
                            batch_loss/ print_every, 100.* batch_correct/ print_every,
                            self.timeSince(start, batch_index*len(i)/ data_size)),end='')
                batch_loss= 0
                batch_correct= 0
        print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)] | Loss: {:.6f} | Accu: {}% | Time: {}  '.format(
                    epoch , batch_index*len(x), data_size, 100.,
                    float(total_loss)/ data_size, 100.*total_correct/ data_size,
                    self.timeSince(start, 1)))
        if self.writer != None:
            self.writer.add_scalar('Train Loss', float(total_loss)/ data_size, epoch)
            self.writer.add_scalar('Train Accu',  100.*total_correct/ data_size, epoch)
        return float(total_loss)/ data_size, 100. * total_correct/ data_size
    def val(self,model,dataloader, epoch, print_every= 10):
        start= time.time()
        model.eval()
        
        criterion= nn.CrossEntropyLoss()
        total_loss= 0
        batch_loss= 0
        total_correct= 0
        batch_correct= 0
        
        data_size= len(dataloader)
        for b, (x, i, y) in enumerate(dataloader):
            batch_index=b+1
            x, i, y= Variable(x).cuda(), Variable(i).cuda() , Variable(y).cuda()
            output= model(x,i)
            loss = criterion(output,y)
            # loss
            batch_loss+= float(loss)
            total_loss+= float(loss)* len(x)
            # accu
            pred = output.data.argmax(1) # get the index of the max log-probability
            correct = pred.eq(y.data).long().cpu().sum()
            batch_correct += correct/ len(x)
            total_correct += correct
            if batch_index% print_every== 0:
                print('\rVal Epoch: {} | [{}/{} ({:.0f}%)] | Loss: {:.6f} | Accu: {}% | Time: {}  '.format(
                            epoch , batch_index*len(x), data_size, 100. * batch_index*len(i)/ data_size,
                            batch_loss/ print_every, 100.* batch_correct/ print_every,
                            self.timeSince(start, batch_index*len(i)/ data_size)),end='')
                batch_loss= 0
                batch_correct= 0
        print('\rVal Epoch: {} | [{}/{} ({:.0f}%)] | Loss: {:.6f} | Accu: {}% | Time: {}  '.format(
                    epoch , batch_index*len(x), data_size, 100.,
                    float(total_loss)/ data_size, 100.*total_correct/ data_size,
                    self.timeSince(start, 1)))
        if self.writer != None:
            self.writer.add_scalar('Val Loss', float(total_loss)/ data_size, epoch)
            self.writer.add_scalar('Val Accu',  100.*total_correct/ data_size, epoch)
        return float(total_loss)/ data_size, 100. * total_correct/ data_size
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

class ResNet50_feature(nn.Module):
    def __init__(self, hidden_dim, label_dim, dropout=0.1):
        super(ResNet50_feature, self).__init__()
        original_model = models.resnet50(pretrained=True)
        self.dropout= nn.Dropout()
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool
        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4
        self.avgpool = original_model.avgpool
        self.classifier = nn.Sequential(
                nn.Linear( 16384,hidden_dim),
                nn.SELU(),
                nn.Dropout(),
                nn.Linear( hidden_dim,hidden_dim),
                nn.SELU(),
                nn.Dropout(),
                nn.Linear( hidden_dim,label_dim))
    def forward(self, x, i):
        packed_data= nn.utils.rnn.pack_padded_sequence(x, i, batch_first=True)
        #print(i)
        #print(sort_i)
        #print(sort_x.size())
        #print(packed_data.data.size())
        z = self.conv1(packed_data.data)
        z = self.bn1(z)
        z = self.relu(z)
        z = self.maxpool(z)
        z = self.dropout(z)

        z = self.layer1(z)
        z = self.dropout(z)
        z = self.layer2(z)
        z = self.dropout(z)
        z = self.layer3(z)
        z = self.dropout(z)
        z = self.layer4(z)
        z = self.dropout(z)

        z = self.avgpool(z)
        z = z.view(z.size(0), -1)
        #print(z.size())
        packed_data=nn.utils.rnn.PackedSequence(z, packed_data.batch_sizes)
        z = nn.utils.rnn.pad_packed_sequence(packed_data,batch_first=True)
        #print(z[0].size())
        z = torch.sum(z[0],1)/ i.unsqueeze(1).repeat(1,z[0].size(2)).float()
        z = self.classifier(z)
        #print(z.size())
        #print(sort_i)
        #input()
        
        return z
class Vgg16_feature(nn.Module):
    def __init__(self, hidden_dim, label_dim, dropout=0):
        super(Vgg16_feature, self).__init__()
        original_model = models.vgg16(pretrained=True)
        self.dropout= nn.Dropout(dropout)
        self.features = original_model.features
        self.classifier = nn.Sequential(
                nn.Linear( 35840,hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear( hidden_dim,hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear( hidden_dim,hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear( hidden_dim,label_dim))
    def forward(self, x, i):
        #print(x.size())
        #print(i)
        packed_data= nn.utils.rnn.pack_padded_sequence(x, i, batch_first=True)
        #print(packed_data.data[0])
        #print(i)
        #print(x.size())
        #print(packed_data.data.size())
        z = self.features(packed_data.data)
        z = z.view(z.size(0), -1)
        #print(packed_data.batch_sizes)
        #print(z.data[0])
        packed_data=nn.utils.rnn.PackedSequence(z, packed_data.batch_sizes)
        #print(packed_data.batch_sizes)
        #print(packed_data.data[:3])
        #input()
        z = nn.utils.rnn.pad_packed_sequence(packed_data,batch_first=True)
        #print(z[0].size())
        z = torch.sum(z[0],1)/ i.unsqueeze(1).repeat(1,z[0].size(2)).float()
        #print(z.size())
        #print(sort_i)
        #input()
        z = self.classifier(z)
        #print(torch.index_select(sort_i, 0, sort_index_reverse))
        #input()
        
        return z
class Vgg16_feature_rnn(nn.Module):
    def __init__(self, hidden_dim, layer_n, label_dim, dropout=0):
        super(Vgg16_feature_rnn, self).__init__()
        original_model = models.vgg16(pretrained=True)
        self.hidden_dim = hidden_dim
        self.layer_n = layer_n
        self.hidden= self.initHidden(hidden_dim)

        self.dropout= nn.Dropout(dropout)
        self.features = original_model.features
        self.classifier1= original_model.classifier
        self.rnn= nn.GRU(35840, hidden_dim,num_layers= layer_n,batch_first=True, dropout=dropout)
        self.classifier2= nn.Sequential(
                nn.Linear( hidden_dim,hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear( hidden_dim,hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear( hidden_dim,hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear( hidden_dim,label_dim))
    def forward(self, x, i):
        #print(x.size())
        #print(i)
        packed_data= nn.utils.rnn.pack_padded_sequence(x, i, batch_first=True)
        #print(packed_data.data[0])
        #print(i)
        #print(x.size())
        #print(packed_data.data.size())
        z = self.features(packed_data.data)
        z = z.view(z.size(0), -1)
        #print(packed_data.batch_sizes)
        #print(z.data[0])
        packed_data=nn.utils.rnn.PackedSequence(z, packed_data.batch_sizes)
        packed_data, _=self.rnn(packed_data, self.hidden_layer(len(x)))
        #print(packed_data.batch_sizes)
        #print(packed_data.data[:3])
        #input()
        z = nn.utils.rnn.pad_packed_sequence(packed_data,batch_first=True)
        #print(z[0].size())
        z = torch.sum(z[0],1)/ i.unsqueeze(1).repeat(1,z[0].size(2)).float()
        #print(z.size())
        #print(sort_i)
        #input()
        z = self.classifier2(z)
        #print(torch.index_select(sort_i, 0, sort_index_reverse))
        #input()
        
        return z
    def hidden_layer(self,n):
        return  self.hidden.repeat(1,n,1)
    def initHidden(self, hidden_size):
        return Variable(torch.zeros(self.layer_n,1, hidden_size).cuda(),requires_grad=True)
    def save(self, path):
        torch.save(self,path)
class Vgg16_feature_rnn_by_frame(nn.Module):
    def __init__(self, hidden_dim, layer_n, label_dim, dropout, input_path):
        super(Vgg16_feature_rnn, self).__init__()
        original_model = torch.load(input_path)
        self.hidden_dim = hidden_dim
        self.layer_n = layer_n
        self.hidden= self.initHidden(hidden_dim)

        self.dropout= nn.Dropout(dropout)
        self.features = original_model.features
        self.dimention_reduction= original_model.dimention_reduction 
        self.rnn= original_model.rnn
        self.classifier = original_model.classifier
    def forward(self, x, hidden):
        x = self.features(x.unsqueeze(0))
        x = x.view(x.size(0), -1)
        x = self.dimention_reduction(x)
        x, hidden=self.rnn(x.unsqueeze(0), hidden)

        x = self.classifier(x.squeeze(0))
        
        return x, hidden
    def hidden_layer(self,n):
        return  self.hidden.repeat(1,n,1)
    def initHidden(self, hidden_size):
        return Variable(torch.zeros(self.layer_n,1, hidden_size).cuda(),requires_grad=True)
    def save(self, path):
        torch.save(self,path)

class ImageDataLoader():
    def __init__(self, image, label, batch_size, shuffle, max_len= 16):
        self.image = image
        self.label = label
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_len = max_len
        self.transform =  torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    def __iter__(self):
        self.index = list(range(len(self.label)))
        if self.shuffle: random.shuffle(self.index)
        self.start_index=0
        self.end_index=min(len(self.label),self.start_index+self.batch_size)
        return self
    def __next__(self):
        if self.start_index >= len(self.label):
            raise StopIteration
        x,i,y=[], [], []
        for j in range(self.start_index,self.end_index):
            image=[ self.transform(torch.FloatTensor(i).permute(2,0,1)/255).unsqueeze(0) for i in self.image[self.index[j]][:self.max_len]]
            x.append(torch.cat(image,0))
            i.append(min(len(self.image[self.index[j]]),self.max_len))
            y.append(self.label[self.index[j]])
        sort_index= torch.LongTensor(sorted(range(len(i)), key=lambda k: i[k], reverse=True))
        sort_x=nn.utils.rnn.pad_sequence( [x[i] for i in sort_index],batch_first=True)
        sort_i= torch.index_select(torch.LongTensor(i), 0, sort_index)
        sort_y= torch.index_select(torch.LongTensor(y), 0, sort_index)
        self.start_index+=self.batch_size
        self.end_index=min(len(self.label),self.start_index+self.batch_size)
        #print(sort_x.size())
        #print(sort_i.size())
        #print(sort_y.size())
        #input()
        return sort_x,sort_i,sort_y
    def __len__(self):
        return len(self.label)
class MovieDataLoader():
    def __init__(self, image_path, label_path, batch_size, shuffle, max_len= 16):
        self.image = None
        self.image_path = image_path
        self.label = None
        self.label_path = label_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_len = max_len
        self.transform =  torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    def __iter__(self):
        x=[]
        file_list = [file for file in os.listdir(self.image_path) if file.endswith('.jpg')]
        file_list.sort()
        for i in file_list:
            x.append(np.array(Image.open('{}/{}'.format(self.image_path,i)),dtype=np.uint8).resize((224, 224)))
            print('\rreading image form {}...{}'.format(self.image_path,len(x)),end='')
        self.image=np.array(x)
        with open(self.label_path, 'r') as f:
            self.label= np.array(list(f)).astype(np.uint8)

        self.index = list(range(len(self.label)))
        if self.shuffle: random.shuffle(self.index)
        self.start_index=0
        self.end_index=min(len(self.label),self.start_index+self.batch_size)
        return self
    def __next__(self):
        if self.start_index >= len(self.label):
            del self.image, self.label
            self.image, self.label= None, None
            raise StopIteration
        x,y=[], []
        for j in range(self.start_index,self.end_index):
            x.append( self.transform(torch.FloatTensor(self.image[self.index[j]]).permute(2,0,1)/255).unsqueeze(0))
            y.append( self.label[self.index[j]])
        x= torch.cat(x,0)
        y= torch.cat(y,0)
        self.start_index+=self.batch_size
        self.end_index=min(len(self.label),self.start_index+self.batch_size)
        #print(sort_x.size())
        #print(sort_i.size())
        #print(sort_y.size())
        #input()
        return x,y
    def __len__(self):
        return len(self.label)
