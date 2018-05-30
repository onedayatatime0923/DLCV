
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torchvision import models
from tensorboardX import SummaryWriter 
import collections, os, skimage.transform, csv, time, math
from skvideo import io
assert torch and Dataset and DataLoader and F

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
                frame = skimage.transform.rescale(frame, rescale_factor, mode='constant', preserve_range=True).astype(np.uint8)
                frames.append(frame)
            else:
                continue

        return np.array(frames).astype(np.uint8)
    def get_data(self, video_path, tag_path, save_path= None, batch_size=32, shuffle= True):
        if save_path != None and os.path.isfile(save_path[0]) and os.path.isfile(save_path[1]):
                x= np.load(save_path[0])
                y= np.load(save_path[1])
                return DataLoader(ImageDataset(x,y), batch_size=batch_size, shuffle=shuffle)
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
        return DataLoader(ImageDataset(x,y), batch_size=batch_size, shuffle=shuffle)
    def train(self, model, dataloader, epoch):
        start= time.time()
        model.train()
        
        optimizer = torch.optim.Adam(model.parameters())
        criterion= nn.CrossEntropyLoss()
        total_loss= 0
        total_correct= 0
        
        data_size= len(dataloader.dataset)
        for b, (x, i, y) in enumerate(dataloader):
            batch_index=b+1
            x, i, y= Variable(x).cuda(), Variable(i).cuda() , Variable(y).cuda().squeeze(1)
            output= model(x,i)
            loss = criterion(output,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # loss
            total_loss+= float(loss)* len(x)
            # accu
            pred = output.data.argmax(1) # get the index of the max log-probability
            correct = pred.eq(y.data).long().cpu().sum()
            total_correct += correct

            print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)] | Loss: {:.6f} | Accu: {:.6f}% | Time: {}  '.format(
                        epoch , batch_index*len(x), data_size, 100. * batch_index*len(i)/ data_size,
                        float(loss), 100.*correct/len(x),
                        self.timeSince(start, batch_index*len(i)/ data_size)),end='')
        print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)] | Loss: {:.6f} | Accu: {:.6f}% | Time: {}  '.format(
                    epoch , data_size, data_size, 100.,
                    float(total_loss)/ data_size, 100.*total_correct/ data_size,
                    self.timeSince(start, 1)))
        if self.writer != None:
            self.writer.add_scalar('Train Loss', float(total_loss)/ data_size, epoch)
            self.writer.add_scalar('Train Accu',  100.*total_correct/ data_size, epoch)
        return float(total_loss)/ data_size, 100. * total_correct/ data_size
    def val(self,model,dataloader, epoch):
        start= time.time()
        model.eval()
        
        criterion= nn.CrossEntropyLoss()
        total_loss= 0
        total_correct= 0
        
        data_size= len(dataloader.dataset)
        for b, (x, i, y) in enumerate(dataloader):
            batch_index=b+1
            x, i, y= Variable(x).cuda(), Variable(i).cuda() , Variable(y).cuda().squeeze(1)
            output= model(x,i)
            loss = criterion(output,y)
            # loss
            total_loss+= float(loss)* len(x)
            # accu
            pred = output.data.argmax(1) # get the index of the max log-probability
            correct = pred.eq(y.data).long().cpu().sum()
            total_correct += correct
            print('\rVal Epoch: {} | [{}/{} ({:.0f}%)] | Loss: {:.6f} | Accu: {:.6f}% | Time: {}  '.format(
                        epoch , batch_index*len(x), data_size, 100. * batch_index*len(i)/ data_size,
                        float(loss), 100.*correct/len(x),
                        self.timeSince(start, batch_index*len(i)/ data_size)),end='')
        print('\rVal Epoch: {} | [{}/{} ({:.0f}%)] | Loss: {:.6f} | Accu: {:.6f}% | Time: {}  '.format(
                    epoch , data_size, data_size, 100.,
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
    def __init__(self, hidden_dim, label_dim, dropout=0.5):
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
                nn.Linear( hidden_dim,label_dim))
    def forward(self, x, i):
        sort_index= torch.cuda.LongTensor(sorted(range(len(i)), key=lambda k: i[k], reverse=True))
        sort_x= torch.index_select(x, 0, sort_index)
        sort_i= torch.index_select(i, 0, sort_index)
        packed_data= nn.utils.rnn.pack_padded_sequence(sort_x, sort_i, batch_first=True)
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
        z = torch.sum(z[0],1)/ sort_i.unsqueeze(1).repeat(1,z[0].size(2)).float()
        z = self.classifier(z)
        #print(z.size())
        #input()
        #print(sort_i)
        #input()
        
        return z

class ImageDataset(Dataset):
    def __init__(self, image, label, max_len= 15):
        self.image = image
        self.label = label
        self.max_len = max_len #max([len(x) for x in image])
    def __getitem__(self, i):
        image= torch.from_numpy(self.image[i]).permute(0,3,1,2).float()
        if len(image)< self.max_len:
            x = torch.cat([image,torch.zeros(self.max_len- image.size(0), *image.size()[1:])],0)
        else:
            x = image[:self.max_len]
        y= torch.LongTensor([self.label[i]])
        return x, min(len(image), self.max_len), y
    def __len__(self):
        return len(self.image)
