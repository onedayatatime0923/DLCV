
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import time, os, Image
import numpy as np
assert torch and nn and Variable and F and Dataset and DataLoader
assert time and np

class DataManager():
    def __init__(self):
        pass
    def get_data(self,name,path,batch_size, shuffle):
        x=[]
        file_list = [file for file in os.listdir(path) if file.endswith('.jpg')]
        file_list.sort()
        for i in range(len(file_list)):
            x.append(np.array(Image.open('{}/{}'.format(path,file_list[i])),dtype=np.uint8).transpose((2,0,1)))
            print('\rreading data...{}'.format(i),end='')
        print('\nreading data...finish')
        x=np.array(x)
        self.data[name]=DataLoader(ImageDataset(x), batch_size=batch_size, shuffle=shuffle)
        return x.shape[1:]
    def train(self,name, encoder, decoder, optimizer, epoch, print_every=5):
        start= time.time()
        encoder.train()
        decoder.train()
        
        encoder_optimizer= optimizer[0]
        decoder_optimizer= optimizer[1]
        
        criterion = nn.CrossEntropyLoss()
        total_loss=0
        batch_loss=0
        
        data_size= len(self.data[name].dataset)
        for i, (x, y) in enumerate(self.data[name]):
            batch_index=i+1
            batch_x, batch_y= Variable(x).cuda(), Variable(y).cuda().long()
            mean_x, logvar_x= encoder(batch_x)
            output= decoder(mean_x, logvar_x)
            #output = output.view(-1,output.size()[2])
            reconstruction_loss = criterion(output,batch_y)
            kl_divergence_loss= torch.sum((-1/2)*( 1+ logvar_x- mean_x**2 - torch.exp(logvar_x)))/(len(x))
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            reconstruction_loss.backward()
            kl_divergence_loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            batch_loss+= float(reconstruction_loss)+ float(kl_divergence_loss) # sum up batch loss
            total_loss+= float(reconstruction_loss)+ float(kl_divergence_loss) # sum up total loss
            if batch_index% print_every == 0:
                print_loss= batch_loss / print_every
                batch_loss= 0
                print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)] |  Loss: {:.6f} | Time: {}  '.format(
                                epoch , batch_index*len(batch_x), data_size,
                                100. * batch_index*len(batch_x)/ data_size, print_loss,
                                self.timeSince(start, batch_index*len(batch_x)/ data_size)),end='')
        print('\nTime: {} | Total loss: {:.4f}'.format(self.timeSince(start,1),total_loss/batch_index))
        print('-'*60)
    def val(self,name, encoder, decoder, optimizer, epoch, print_every=5):
        start= time.time()
        encoder.eval()
        decoder.eval()

        criterion = nn.CrossEntropyLoss()
        total_loss=0
        predict=[]
        
        data_size= len(self.data[name].dataset)
        for i, (x, y) in enumerate(self.data[name]):
            batch_index=i+1
            batch_x, batch_y= Variable(x).cuda(), Variable(y).cuda().long()
            mean_x, logvar_x= encoder(batch_x)
            output= decoder(mean_x, logvar_x, mode='test')
            reconstruction_loss = criterion(output,batch_y)
            kl_divergence_loss= torch.sum((-1/2)*( 1+ logvar_x- mean_x**2 - torch.exp(logvar_x)))/(len(x))
            # loss
            total_loss+= float(reconstruction_loss)+ float(kl_divergence_loss) # sum up total loss
            # predict
            predict.extend(output.unsqueeze(0))
            if batch_index % print_every == 0:
                print('\rVal: [{}/{} ({:.0f}%)] | Time: {}'.format(
                         batch_index * len(x), data_size,
                        100. * batch_index / data_size,
                        self.timeSince(start, batch_index*len(batch_x)/ data_size)),end='')

        total_loss/=  batch_index
        predict=torch.cat(predict,0)
        print('\nVal set: Average loss: {:.4f} | Time: {}  '.format(total_loss,
                        self.timeSince(start, batch_index*len(batch_x)/ data_size)),end='')
        print('-'*80)
class Encoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(Encoder, self).__init__()
        self.output_size=output_size
        self.conv1 = nn.Sequential(                 # input shape (1, 28, 28)
            nn.Conv2d(input_size[0], 128, 3, 1, 1),              # output shape (16, 28, 28)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),            # output shape (16,  7,  7)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),         
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),            
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d( 64, 32, 3, 1, 1),         
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),            
        )
        self.den1= nn.Sequential(
            nn.Linear( 32*(input_size[1]//8)*(input_size[2]//8),  output_size*2),
            nn.ReLU(),
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x= self.den1(x)
        return x[:,:self.output_size], x[:,self.output_size:]
class Decoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(Encoder, self).__init__()
        self.output_size=output_size
        self.den1= nn.Sequential(
            nn.Linear(input_size, (output_size[1]//8)* (output_size[2]//8) *32),
            nn.ReLU(),
        )
        self.conv1 = nn.Sequential(                 # input shape (1, 28, 28)
            nn.Conv2d( 32, 64, 3, 1, 1),              # output shape (16, 28, 28)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),            # output shape (16,  7,  7)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,128, 3, 1, 1),         
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),            
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128,output_size[0], 3, 1, 1),         
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),            
        )
    def forward(self, mean_x, logvar_x):
        epsilon_x= torch.normal(mean=torch.zeros_like(mean_x))
        x = mean_x+ epsilon_x*(torch.exp((1/2)*logvar_x))
        x= self.den1(x)
        x = x.view(32,(self.output_size[1]//8), (self.output_size[2]//8))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x 
class ImageDataset(Dataset):
    def __init__(self, data):
        self.data=data
    def __getitem__(self, i):
        x=torch.FloatTensor(self.data[i][:])/255
        y=torch.FloatTensor(self.data[i][:])/255
        return x,y
    def __len__(self):
        return len(self.data)
