
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
import sys
from skvideo import io
from scipy import misc
from PIL import Image
import matplotlib.pyplot as plt
assert torch and F and skimage and plt and DataLoader and sys

class DataManager():
    def __init__(self, path=None):
        self.feature_extractor = None
        self.transform= torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
    def get_data(self, video_path, tag_path, save_path= None):
        if save_path!= None:
            if os.path.isfile(save_path[0]) and os.path.isfile(save_path[1]):
                feature_size= np.load(save_path[0])[0].shape[1]
                print('data has already been preprocessed!!!')
                print('feature size: {}'.format(feature_size))
                return [np.load(save_path[0]),np.load(save_path[1])]

        file_dict=(self.getVideoList(tag_path))
        x, y=[], []
        self.set_feature_extractor()
        for i in range(len(file_dict['Video_index'])):
            image=self.readShortVideo(video_path, file_dict['Video_category'][i],file_dict['Video_name'][i])
            feature= []
            for im in image:
                variable_image= Variable(self.transform(torch.FloatTensor(misc.imresize(im,(224,224))).permute(2,0,1)/255).unsqueeze(0).cuda())
                feature.append(self.feature_extractor(variable_image).detach().squeeze(0).cpu().numpy())
            x.append(np.array(feature))
            y.append(int(file_dict['Action_labels'][i]))
            print('\rreading image from {}...{}'.format(video_path,i),end='')
        x= np.array(x)
        y= np.array(y)
        if save_path!= None:
            np.save(save_path[0],x)
            np.save(save_path[1],y)
        feature_size= x[0].shape[1]
        print('\rreading image from {}...finished'.format(video_path))
        print('data has been preprocessed!!!')
        print('feature size: {}'.format(feature_size))
        return [x,y]
    def get_movie(self, video_path, tag_path=None, save_path= None, cut=sys.maxsize):
        if save_path!= None:
            if os.path.isfile(save_path[0]) and os.path.isfile(save_path[1]):
                print('data has already been preprocessed!!!')
                return [np.load(save_path[0]),np.load(save_path[1])]

        self.set_feature_extractor()
        moviedir= os.listdir(video_path)
        x=[]
        for m in moviedir:
            file_list = [file for file in os.listdir('{}/{}'.format(video_path,m)) if file.endswith('.jpg')]
            file_list.sort()
            images=[]
            for f in file_list:
                im= np.array(Image.open('{}/{}/{}'.format(video_path,m,f)).resize((224,224)))
                variable_image= Variable(self.transform(torch.FloatTensor(im).permute(2,0,1)/255).unsqueeze(0).cuda())
                images.append(self.feature_extractor(variable_image).detach().squeeze(0).cpu().numpy())
                if len(images) >= cut:
                    x.append(np.array(images))
                    images=[]
                print('\rreading image from {}/{}...{}'.format(video_path,m,len(x)),end='')
            x.append(np.array(images))
        print('\rreading image from {}...finished'.format(video_path))
        x= np.array(x)

        if tag_path is not None:
            y=[]
            for m in moviedir:
                print('\rreading tag from {}/{}.txt...'.format(tag_path,m),end='')
                with open('{}/{}.txt'.format(tag_path,m)) as f:
                    data= [ i.strip() for i in f.readlines()]
                    data= [np.array(data[i:i + cut]).astype(np.uint8) for i in range(0, len(data), cut)]
                    y.extend(data)
            y= np.array(y)

        if save_path!= None:
            np.save(save_path[0],x)
            if tag_path is not None:
                np.save(save_path[1],y)
        feature_size= x[0].shape[1]
        print('data has been preprocessed!!!')
        print('feature size: {}'.format(feature_size))
        if tag_path is not None:
            return [x, y]
        else:
            return x, moviedir
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
                    epoch , data_size, data_size, 100.,
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
                    epoch , data_size, data_size, 100.,
                    float(total_loss)/ data_size, 100.*total_correct/ data_size,
                    self.timeSince(start, 1)))
        if self.writer != None:
            self.writer.add_scalar('Val Loss', float(total_loss)/ data_size, epoch)
            self.writer.add_scalar('Val Accu',  100.*total_correct/ data_size, epoch)
        return float(total_loss)/ data_size, 100. * total_correct/ data_size
    def test_classifier(self,model,dataloader, epoch, print_every= 10):
        start= time.time()
        model.eval()
        
        criterion= nn.CrossEntropyLoss()
        total_loss= 0
        batch_loss= 0
        total_correct= 0
        batch_correct= 0
        
        data_size= len(dataloader.dataset)
        result=[]
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
            # result
            result.extend(output.data.argmax(1).unsqueeze(0))
            if batch_index% print_every== 0:
                print('\rTest Epoch: {} | [{}/{} ({:.0f}%)] | Loss: {:.6f} | Accu: {:2f}% | Time: {}  '.format(
                            epoch , batch_index*len(x), data_size, 100. * batch_index*len(x)/ data_size,
                            batch_loss/ print_every, 100.* batch_correct/ print_every,
                            self.timeSince(start, batch_index*len(x)/ data_size)),end='')
                batch_loss= 0
                batch_correct= 0
        print('\rTest Epoch: {} | [{}/{} ({:.0f}%)] | Loss: {:.6f} | Accu: {:2f}% | Time: {}  '.format(
                    epoch , data_size, data_size, 100.,
                    float(total_loss)/ data_size, 100.*total_correct/ data_size,
                    self.timeSince(start, 1)))
        result= torch.cat( result, 0).cpu()
        return  result
    def train_rnn(self, model, dataloader, epoch, lr, print_every= 10):
        start= time.time()
        model.train()
        
        #optimizer = torch.optim.Adam(list(model.parameters())+[model.hidden],lr=lr)
        optimizer = torch.optim.Adam(model.parameters(),lr=lr)
        criterion= nn.CrossEntropyLoss()
        total_loss= 0
        batch_loss= 0
        total_correct= 0
        batch_correct= 0
        
        data_size= len(dataloader)
        for b, (x, i, y, _) in enumerate(dataloader):
            batch_index=b+1
            x, i, y= Variable(x).cuda(), Variable(i).cuda(), Variable(y).cuda()
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
            correct = int(pred.eq(y.data).long().cpu().sum())

            batch_correct += correct/ len(x)
            total_correct += correct
            if batch_index% print_every== 0:
                print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)] | Loss: {:.6f} | Accu: {:.2f}% | Time: {}  '.format(
                            epoch , batch_index*len(x), data_size, 100. * batch_index*len(x)/ data_size,
                            batch_loss/ print_every, 100.* batch_correct/ print_every,
                            self.timeSince(start, batch_index*len(x)/ data_size)),end='')
                batch_loss= 0
                batch_correct= 0
        print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)] | Loss: {:.6f} | Accu: {:.2f}% | Time: {}  '.format(
                    epoch , data_size, data_size, 100.,
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
        for b, (x, i, y, _) in enumerate(dataloader):
            batch_index=b+1
            x, i, y= Variable(x).cuda(), Variable(i).cuda(), Variable(y).cuda()
            output= model(x,i)
            loss = criterion(output,y)
            # loss
            batch_loss+= float(loss)
            total_loss+= float(loss)* len(x)
            # accu
            pred = output.data.argmax(1) # get the index of the max log-probability
            correct = int(pred.eq(y.data).long().cpu().sum())
            batch_correct += correct/ len(x)
            total_correct += correct
            if batch_index% print_every== 0:
                print('\rVal Epoch: {} | [{}/{} ({:.0f}%)] | Loss: {:.6f} | Accu: {:.2f}% | Time: {}  '.format(
                            epoch , batch_index*len(x), data_size, 100. * batch_index*len(x)/ data_size,
                            batch_loss/ print_every, 100.* batch_correct/ print_every,
                            self.timeSince(start, batch_index*len(x)/ data_size)),end='')
                batch_loss= 0
                batch_correct= 0
        print('\rVal Epoch: {} | [{}/{} ({:.0f}%)] | Loss: {:.6f} | Accu: {:.2f}% | Time: {}  '.format(
                    epoch , data_size, data_size, 100.,
                    float(total_loss)/ data_size, 100.*total_correct/ data_size,
                    self.timeSince(start, 1)))
        if self.writer != None:
            self.writer.add_scalar('Val Loss', float(total_loss)/ data_size, epoch)
            self.writer.add_scalar('Val Accu',  100.*total_correct/ data_size, epoch)
        return float(total_loss)/ data_size, 100. * total_correct/ data_size
    def test_rnn(self,model,dataloader, epoch, print_every= 10):
        start= time.time()
        model.eval()
        
        criterion= nn.CrossEntropyLoss()
        total_loss= 0
        batch_loss= 0
        total_correct= 0
        batch_correct= 0
        
        data_size= len(dataloader)
        result= []
        for b, (x, i, y, sort_index) in enumerate(dataloader):
            batch_index=b+1
            x, i, y= Variable(x).cuda(), Variable(i).cuda(), Variable(y).cuda()
            output= model(x,i)
            loss = criterion(output,y)
            # loss
            batch_loss+= float(loss)
            total_loss+= float(loss)* len(x)
            # accu
            pred = output.data.argmax(1) # get the index of the max log-probability
            correct = int(pred.eq(y.data).long().cpu().sum())
            batch_correct += correct/ len(x)
            total_correct += correct
            # result
            result.extend(dataloader.reverse(output.data.argmax(1),sort_index).unsqueeze(0))
            if batch_index% print_every== 0:
                print('\rTest Epoch: {} | [{}/{} ({:.0f}%)] | Loss: {:.6f} | Accu: {:.2f}% | Time: {}  '.format(
                            epoch , batch_index*len(x), data_size, 100. * batch_index*len(x)/ data_size,
                            batch_loss/ print_every, 100.* batch_correct/ print_every,
                            self.timeSince(start, batch_index*len(x)/ data_size)),end='')
                batch_loss= 0
                batch_correct= 0
        print('\rTest Epoch: {} | [{}/{} ({:.0f}%)] | Loss: {:.6f} | Accu: {:.2f}% | Time: {}  '.format(
                    epoch , data_size, data_size, 100.,
                    float(total_loss)/ data_size, 100.*total_correct/ data_size,
                    self.timeSince(start, 1)))
            
        result= torch.cat( result, 0).cpu()
        return  result
    def train_movie(self, model, dataloader, epoch, lr, print_every= 10):
        start= time.time()
        model.train()
        
        #optimizer = torch.optim.Adam(list(model.parameters())+[model.hidden],lr=lr)
        optimizer = torch.optim.Adam(model.parameters(),lr=lr)
        total_loss= 0
        batch_loss= 0
        total_correct= 0
        batch_correct= 0
        total_count= 0
        batch_count= 0
        
        data_size= len(dataloader)
        for b, (x, i, y, _) in enumerate(dataloader):
            batch_index=b+1
            x, i, y= Variable(x).cuda(), Variable(i).cuda(), Variable(y).cuda()
            output= model(x,i)
            loss = self.pack_CCE(output,y,i)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # loss
            batch_loss+= float(loss)
            total_loss+= float(loss)* len(x)
            # accu
            correct, count = self.pack_accu(output, y, i)

            batch_correct += correct
            total_correct += correct
            batch_count += count
            total_count += count
            if batch_index% print_every== 0:
                print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)] | Loss: {:.6f} | Accu: {:.2f}% | Time: {}  '.format(
                            epoch , batch_index*len(x), data_size, 100. * batch_index*len(x)/ data_size,
                            batch_loss/ print_every, 100.* batch_correct/ batch_count,
                            self.timeSince(start, batch_index*len(x)/ data_size)),end='')
                batch_loss= 0
                batch_correct= 0
                batch_count= 0
        print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)] | Loss: {:.6f} | Accu: {:.2f}% | Time: {}  '.format(
                    epoch , data_size, data_size, 100.,
                    float(total_loss)/ data_size, 100.*total_correct/ total_count,
                    self.timeSince(start, 1)))
        if self.writer != None:
            self.writer.add_scalar('Train Loss', float(total_loss)/ data_size, epoch)
            self.writer.add_scalar('Train Accu',  100.*total_correct/ data_size, epoch)
        return float(total_loss)/ data_size, 100. * total_correct/ data_size
    def val_movie(self,model,dataloader, epoch, print_every= 10):
        start= time.time()
        model.eval()
        
        total_loss= 0
        batch_loss= 0
        total_correct= 0
        batch_correct= 0
        total_count= 0
        batch_count= 0
        
        data_size= len(dataloader)
        for b, (x, i, y, _) in enumerate(dataloader):
            batch_index=b+1
            x, i, y= Variable(x).cuda(), Variable(i).cuda(), Variable(y).cuda()
            output= model(x,i)
            loss = self.pack_CCE(output,y,i)
            # loss
            batch_loss+= float(loss)
            total_loss+= float(loss)* len(x)
            # accu
            correct, count = self.pack_accu(output, y, i)

            batch_correct += correct
            total_correct += correct
            batch_count += count
            total_count += count
            if batch_index% print_every== 0:
                print('\rVal Epoch: {} | [{}/{} ({:.0f}%)] | Loss: {:.6f} | Accu: {:.2f}% | Time: {}  '.format(
                            epoch , batch_index*len(x), data_size, 100. * batch_index*len(x)/ data_size,
                            batch_loss/ print_every, 100.* batch_correct/ batch_count,
                            self.timeSince(start, batch_index*len(x)/ data_size)),end='')
                batch_loss= 0
                batch_correct= 0
                batch_count = 0
        print('\rVal Epoch: {} | [{}/{} ({:.0f}%)] | Loss: {:.6f} | Accu: {:.2f}% | Time: {}  '.format(
                    epoch , data_size, data_size, 100.,
                    float(total_loss)/ data_size, 100.*total_correct/ batch_count,
                    self.timeSince(start, 1)))
        if self.writer != None:
            self.writer.add_scalar('Val Loss', float(total_loss)/ data_size, epoch)
            self.writer.add_scalar('Val Accu',  100.*total_correct/ data_size, epoch)
        return float(total_loss)/ data_size, 100. * total_correct/ data_size
    def test_movie(self,model,dataloader, epoch, print_every= 10):
        start= time.time()
        model.eval()
        
        total_loss= 0
        batch_loss= 0
        total_correct= 0
        batch_correct= 0
        total_count= 0
        batch_count= 0
        
        data_size= len(dataloader)
        result= []
        index= []
        for b, (x, i, y, sort_index) in enumerate(dataloader):
            batch_index=b+1
            x, i, y= Variable(x).cuda(), Variable(i).cuda(), Variable(y).cuda()
            output= model(x,i)
            loss = self.pack_CCE(output,y,i)
            # loss
            batch_loss+= float(loss)
            total_loss+= float(loss)* len(x)
            # accu
            correct, count = self.pack_accu(output, y, i)

            batch_correct += correct
            total_correct += correct
            batch_count += count
            total_count += count
            if batch_index% print_every== 0:
                print('\rVal Epoch: {} | [{}/{} ({:.0f}%)] | Loss: {:.6f} | Accu: {:.2f}% | Time: {}  '.format(
                            epoch , batch_index*len(x), data_size, 100. * batch_index*len(x)/ data_size,
                            batch_loss/ print_every, 100.* batch_correct/ batch_count,
                            self.timeSince(start, batch_index*len(x)/ data_size)),end='')
                batch_loss= 0
                batch_correct= 0
                batch_count = 0

            result.extend(dataloader.reverse(output.data.argmax(2),sort_index).unsqueeze(0))
            index.extend(dataloader.reverse(i,sort_index).unsqueeze(0))
        print('\rVal Epoch: {} | [{}/{} ({:.0f}%)] | Loss: {:.6f} | Accu: {:.2f}% | Time: {}  '.format(
                    epoch , data_size, data_size, 100.,
                    float(total_loss)/ data_size, 100.*total_correct/ batch_count,
                    self.timeSince(start, 1)))
        result= torch.cat( result, 0).cpu()
        index= torch.cat( index, 0).cpu()
        return  result, index
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
    def plot(self, im):
        data= im
        plt.imshow(data)
        plt.show()
    def pack_CCE(self, x, y, i):
        packed_x= nn.utils.rnn.pack_padded_sequence(x, i, batch_first=True)
        packed_y= nn.utils.rnn.pack_padded_sequence(y, i, batch_first=True)
        result = F.cross_entropy(packed_x.data,packed_y.data)
        return result
    def pack_accu(self, x, y, i):
        packed_x= nn.utils.rnn.pack_padded_sequence(x, i, batch_first=True)
        packed_y= nn.utils.rnn.pack_padded_sequence(y, i, batch_first=True)
        pred = packed_x.data.argmax(1)
        correct = int(pred.eq(packed_y.data).long().cpu().sum())
        count = len(pred)
        return correct, count
    def write(self, data, path):
        output= '\n'.join(list(data.numpy().astype(str)))
        with open(path, 'w') as f:
            f.write(output)
    def write_movie(self, data, index, path_dir, path_list):
        for i in range(len(data)):
            self.write(data[i][:int(index[i])],'{}/{}.txt'.format(path_dir,path_list[i]))
        

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
        self.classifier = nn.Sequential(
                nn.Linear( input_dim,hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                #nn.SELU(),
                nn.Linear( hidden_dim,hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                #nn.SELU(),
                nn.Dropout(dropout),
                nn.Linear( hidden_dim,hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                #nn.SELU(),
                nn.Dropout(dropout),
                nn.Linear( hidden_dim,label_dim))
    def forward(self, x):
        x = self.classifier(x)
        return x
    def save(self, path):
        torch.save(self,path)
class Rnn_Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_n, label_dim, dropout ):
        super(Rnn_Classifier, self).__init__()
        self.layer_n = layer_n
        self.hidden= self.initHidden(hidden_dim)

        self.rnn= nn.GRU( input_dim, hidden_dim,num_layers= layer_n,batch_first=True, dropout=dropout)
        #self.rnn= nn.LSTM( input_dim, hidden_dim,num_layers= layer_n,batch_first=True, dropout=dropout)
        self.bn= nn.BatchNorm1d(hidden_dim* layer_n)
        self.classifier = nn.Sequential(
                nn.Linear( hidden_dim* layer_n,hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                #nn.SELU(),
                nn.Linear( hidden_dim,hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                #nn.SELU(),
                nn.Dropout(dropout),
                nn.Linear( hidden_dim,hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                #nn.SELU(),
                nn.Dropout(dropout),
                nn.Linear( hidden_dim,label_dim))
    def forward(self, x, i):
        packed_data= nn.utils.rnn.pack_padded_sequence(x, i, batch_first=True)

        packed_data, hidden=self.rnn(packed_data, self.hidden_layer(len(x)))
        #z = nn.utils.rnn.pad_packed_sequence(packed_data,batch_first=True)

        z = hidden.permute(1,0,2).contiguous().view(hidden.size(1),-1)
        #z=torch.mean(torch.transpose(hidden,0,1).contiguous(),1)


        #index= i.unsqueeze(1).unsqueeze(2).repeat(1,1,z[0].size(2))
        #z= torch.gather(z[0],1,index-1).squeeze(1)

        #z=torch.sum(z[0],1)/ i.float().unsqueeze(1).repeat(1,z[0].size(2))


        z = self.bn(z)
        z = self.classifier(z)
        
        return z
    def hidden_layer(self,n):
        return  self.hidden.repeat(1,n,1)
    def initHidden(self, hidden_size):
        return Variable(torch.zeros(self.layer_n,1, hidden_size).cuda(),requires_grad=True)
    def save(self, path):
        torch.save(self,path)
class Rnn_Classifier_Movie(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_n, label_dim, dropout ):
        super(Rnn_Classifier_Movie, self).__init__()
        self.layer_n = layer_n
        self.hidden= self.initHidden(hidden_dim)

        self.rnn= nn.GRU( input_dim, hidden_dim,num_layers= layer_n,batch_first=True, dropout=dropout)
        #self.rnn= nn.LSTM( input_dim, hidden_dim,num_layers= layer_n,batch_first=True, dropout=dropout)
        self.bn= nn.BatchNorm1d(hidden_dim)
        self.classifier = nn.Sequential(
                nn.Linear( hidden_dim ,hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                #nn.SELU(),
                nn.Linear( hidden_dim,hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                #nn.SELU(),
                nn.Dropout(dropout),
                nn.Linear( hidden_dim,hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                #nn.SELU(),
                nn.Dropout(dropout),
                nn.Linear( hidden_dim,label_dim))
    def forward(self, x, i):
        packed_data= nn.utils.rnn.pack_padded_sequence(x, i, batch_first=True)

        packed_data, hidden=self.rnn(packed_data, self.hidden_layer(len(x)))

        z = self.bn(packed_data.data)
        z = self.classifier(z)

        packed_data= nn.utils.rnn.PackedSequence( z, packed_data.batch_sizes)

        z = nn.utils.rnn.pad_packed_sequence(packed_data,batch_first=True)

        #z = hidden.permute(1,0,2).contiguous().view(hidden.size(1),-1)
        #z=torch.mean(torch.transpose(hidden,0,1).contiguous(),1)


        #index= i.unsqueeze(1).unsqueeze(2).repeat(1,1,z[0].size(2))
        #z= torch.gather(z[0],1,index-1).squeeze(1)

        #z=torch.sum(z[0],1)/ i.float().unsqueeze(1).repeat(1,z[0].size(2))


        return z[0]
    def hidden_layer(self,n):
        return  self.hidden.repeat(1,n,1)
    def initHidden(self, hidden_size):
        return Variable(torch.zeros(self.layer_n,1, hidden_size).cuda(),requires_grad=True)
    def save(self, path):
        torch.save(self,path)

class ImageDataset(Dataset):
    def __init__(self, image=None, label=None):
        self.image = image
        self.label = label
    def __getitem__(self, i):
        x=torch.mean(torch.FloatTensor(self.image[i]).cuda(),0)
        y=torch.LongTensor([self.label[i]])
        return x,y
    def __len__(self):
        return len(self.image)
class ImageDataLoader():
    def __init__(self, image, label, batch_size, shuffle, max_len= 512):
        self.image = image
        self.label = label
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
        #print(sort_i)
        #print(sort_y)
        #input()
        return sort_x,sort_i,sort_y, sort_index
    def __len__(self):
        return len(self.label)
    def reverse(self, x, i):
        sort_index= torch.cuda.LongTensor(sorted(range(len(i)), key=lambda k: i[k]))
        sort_x= torch.index_select(x, 0, sort_index)
        return sort_x
class MovieDataLoader():
    def __init__(self, image, label, batch_size, shuffle, max_len=10000 ):
        self.image = image
        self.label = label
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
            if self.label is None:
                y.append(torch.LongTensor([0 for i in range(min(len(self.image[self.index[j]]),self.max_len))]))
            else:
                y.append(torch.LongTensor(self.label[self.index[j]][:self.max_len]))
        sort_index= torch.LongTensor(sorted(range(len(i)), key=lambda k: i[k], reverse=True))
        sort_x=nn.utils.rnn.pad_sequence( [x[i] for i in sort_index],batch_first=True)
        sort_i= torch.index_select(torch.LongTensor(i), 0, sort_index)
        sort_y=nn.utils.rnn.pad_sequence( [y[i] for i in sort_index],batch_first=True)
        self.start_index+=self.batch_size
        self.end_index=min(len(self.label),self.start_index+self.batch_size)
        #print(sort_x.size())
        #print(sort_i)
        #print(sort_y.size())
        #input()
        return sort_x,sort_i,sort_y, sort_index
    def __len__(self):
        return len(self.label)
    def reverse(self, x, i):
        sort_index= torch.cuda.LongTensor(sorted(range(len(i)), key=lambda k: i[k]))
        sort_x= torch.index_select(x, 0, sort_index)
        return sort_x
