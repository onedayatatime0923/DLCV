
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import torchvision
from torchvision import models
from tensorboardX import SummaryWriter 
import collections, os, skimage.transform, csv, time, math, random
from skvideo import io
from skimage.transform import resize
from PIL import Image
import matplotlib.pyplot as plt
assert torch and F and skimage and plt and resize and DataLoader

class DataManager():
    def __init__(self, path=None):
        self.feature_extractor = None
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
                frame = skimage.transform.rescale(frame, rescale_factor, mode='constant', preserve_range=True).astype(np.uint8)
                frames.append(frame)
            else:
                continue

        return np.array(frames).astype(np.uint8)
    def set_feature_extractor(self):
        self.feature_extractor = Vgg16_feature_extractor().cuda()
    def get_data(self, video_path, tag_path, save_path, batch_size=32, shuffle= True):
        if os.path.isfile(save_path[0]) and os.path.isfile(save_path[1]):
            feature_size= np.load(save_path[0])[0].shape[1]
            print('data has already been preprocessed!!!')
            print('feature size: {}'.format(feature_size))
            return feature_size

        file_dict=(self.getVideoList(tag_path))
        x, y=[], []
        self.set_feature_extractor()
        for i in range(len(file_dict['Video_index'])):
            image=self.readShortVideo(video_path, file_dict['Video_category'][i],file_dict['Video_name'][i])
            feature= []
            for im in image:
                variable_image= Variable(torch.FloatTensor(im).unsqueeze(0).permute(0,3,1,2).cuda())
                feature.append(self.feature_extractor(variable_image).detach().squeeze(0).cpu().numpy())
            x.append(np.array(feature))
            y.append(int(file_dict['Action_labels'][i]))
            print('\rreading image from {}...{}'.format(video_path,i),end='')
        x= np.array(x)
        y= np.array(y)
        np.save(save_path[0],x)
        np.save(save_path[1],y)
        feature_size= x[0].shape[1]
        print('\rreading image from {}...finished'.format(video_path))
        print('data has been preprocessed!!!')
        print('feature size: {}'.format(feature_size))
        return feature_size
    def train_classifier(self, model, dataloader, epoch, lr=1E-5, print_every= 10):
        start= time.time()
        model.train()
        
        optimizer = torch.optim.Adam(model.parameters(),lr=lr)
        criterion= nn.CrossEntropyLoss()
        total_loss= 0
        batch_loss= 0
        total_correct= 0
        batch_correct= 0
        
        data_size= len(dataloader.dataset)
        for b, (x, y) in enumerate(dataloader):
            batch_index=b+1
            x, y= Variable(x).cuda(), Variable(y).squeeze(1).cuda()
            output= model(x)
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
                            epoch , batch_index*len(x), data_size, 100. * batch_index*len(x)/ data_size,
                            batch_loss/ print_every, 100.* batch_correct/ print_every,
                            self.timeSince(start, batch_index*len(x)/ data_size)),end='')
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
    def val_classifier(self,model,dataloader, epoch, print_every= 10):
        start= time.time()
        model.eval()
        
        criterion= nn.CrossEntropyLoss()
        total_loss= 0
        batch_loss= 0
        total_correct= 0
        batch_correct= 0
        
        data_size= len(dataloader.dataset)
        for b, (x, y) in enumerate(dataloader):
            batch_index=b+1
            x, y= Variable(x).cuda(), Variable(y).squeeze(1).cuda()
            output= model(x)
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
                            epoch , batch_index*len(x), data_size, 100. * batch_index*len(x)/ data_size,
                            batch_loss/ print_every, 100.* batch_correct/ print_every,
                            self.timeSince(start, batch_index*len(x)/ data_size)),end='')
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
    def train_rnn(self, model, dataloader, epoch, lr=1E-5, print_every= 10):
        start= time.time()
        model.train()
        
        optimizer = torch.optim.Adam(model.parameters(),lr=lr)
        criterion= nn.CrossEntropyLoss()
        total_loss= 0
        batch_loss= 0
        total_correct= 0
        batch_correct= 0
        
        data_size= len(dataloader)
        for b, (x, i, y) in enumerate(dataloader):
            batch_index=b+1
            x, i, y= Variable(x).cuda(), Variable(i).cuda(), Variable(y).cuda()
            #print(x.size())
            #print(i)
            #print(y)
            output= model(x,i)
            loss = criterion(output,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss=loss.detach()
            print(loss)
            input()
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
                            epoch , batch_index*len(x), data_size, 100. * batch_index*len(x)/ data_size,
                            batch_loss/ print_every, 100.* batch_correct/ print_every,
                            self.timeSince(start, batch_index*len(x)/ data_size)),end='')
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
    def val_rnn(self,model,dataloader, epoch, print_every= 10):
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
            x, i, y= Variable(x).cuda(), Variable(i).cuda(), Variable(y).cuda()
            output= model(x)
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
                            epoch , batch_index*len(x), data_size, 100. * batch_index*len(x)/ data_size,
                            batch_loss/ print_every, 100.* batch_correct/ print_every,
                            self.timeSince(start, batch_index*len(x)/ data_size)),end='')
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

class ResNet50_feature_extractor(nn.Module):
    def __init__(self, hidden_dim, label_dim, dropout=0.1):
        super(ResNet50_feature_extractor, self).__init__()
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
    def forward(self, x, i):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)

        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.dropout(x)
        x = self.layer4(x)
        x = self.dropout(x)

        x = self.avgpool(x)
        x = x.view(x.sixe(0), -1)
        
        return x
class Vgg16_feature_extractor(nn.Module):
    def __init__(self):
        super(Vgg16_feature_extractor, self).__init__()
        self.features = models.vgg16(pretrained=True).features
    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0),-1)

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, label_dim, dropout=0.5):
        super(Classifier, self).__init__()
        self.dimention_reduction = nn.Sequential(
                nn.Linear( input_dim,hidden_dim),
                nn.SELU())
                #nn.ReLU(inplace=True))
        self.classifier = nn.Sequential(
                nn.Linear( hidden_dim,hidden_dim),
                nn.SELU(),
                #nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear( hidden_dim,hidden_dim),
                nn.SELU(),
                #nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear( hidden_dim,label_dim))
    def forward(self, x):
        x = self.dimention_reduction(x)
        x = self.classifier(x)
        return x
    def save(self, path):
        torch.save(self,path)
class Rnn_Classifier(nn.Module):
    def __init__(self, hidden_dim, layer_n, dropout, classifier_path):
        super(Rnn_Classifier, self).__init__()
        self.layer_n = layer_n
        self.hidden= self.initHidden(hidden_dim)

        self.dimention_reduction = torch.load(classifier_path).dimention_reduction
        self.rnn= nn.GRU( hidden_dim, hidden_dim,num_layers= layer_n,batch_first=True, dropout=dropout)
        self.classifier = torch.load(classifier_path).classifier
    def forward(self, x, i):
        packed_data= nn.utils.rnn.pack_padded_sequence(x, i, batch_first=True)

        z = self.dimention_reduction(packed_data.data)

        packed_data = nn.utils.rnn.PackedSequence(z,packed_data.batch_sizes)

        packed_data, _=self.rnn(packed_data, self.hidden_layer(len(x)))

        # todo: get the last step data of packed_data
        z = nn.utils.rnn.pad_packed_sequence(packed_data,batch_first=True)
        index= z[1].cuda().unsqueeze(1).unsqueeze(2).repeat(1,1,z[0].size(2))

        z= torch.gather(z[0],1,index-1).squeeze(1)

        z = self.classifier(z)
        
        return z
    def hidden_layer(self,n):
        return  self.hidden.repeat(1,n,1)
    def initHidden(self, hidden_size):
        return Variable(torch.zeros(self.layer_n,1, hidden_size).cuda())
    def save(self, path):
        torch.save(self,path)
class Vgg16_feature_rnn_by_frame(nn.Module):
    def __init__(self, hidden_dim, layer_n, label_dim, dropout, input_path):
        super(Vgg16_feature_rnn_by_frame, self).__init__()
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

class ImageDataset(Dataset):
    def __init__(self, image_path, label_path):
        self.image = np.load(image_path)
        self.label = np.load(label_path)
    def __getitem__(self, i):
        x=torch.mean(torch.FloatTensor(self.image[i]).cuda(),0)
        y=torch.LongTensor([self.label[i]])
        return x,y
    def __len__(self):
        return len(self.image)
class ImageDataLoader():
    def __init__(self, image_path, label_path, batch_size, shuffle, max_len= 16):
        self.image = np.load(image_path)
        self.label = np.load(label_path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_len = max_len
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
            x.append(torch.FloatTensor(self.image[self.index[j]][:self.max_len]))
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
