
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import torchvision
from PIL import Image
import numpy as np
import time, os, math
from tensorboardX import SummaryWriter 
assert torch and nn and Variable and F and Dataset and DataLoader
assert time and np

torch.manual_seed(1213)

class DataManager():
    def __init__(self,latent_dim=0, discriminator_update_num=0, generator_update_num=0):
        self.data={}
        self.latent_dim= latent_dim
        self.discriminator_update_num= discriminator_update_num
        self.generator_update_num= generator_update_num
    def tb_setting(self, path):
        for f in os.listdir(path): os.remove('{}/{}'.format(path,f))
        self.writer = SummaryWriter(path)
    def tb_graph(self, model, input_shape):
        if isinstance(input_shape, tuple):
            dummy_input= Variable( torch.rand(1, *input_shape).cuda())
        elif isinstance(input_shape, int):
            dummy_input= Variable( torch.rand(1, input_shape).cuda())
        else: raise ValueError('Wrong input_shape')
        self.writer.add_graph(nn.Sequential(*model), (dummy_input, ))
    def get_data(self,name,path,mode,batch_size, shuffle):
        x=[]
        for p in path:
            file_list = [file for file in os.listdir(p) if file.endswith('.png')]
            file_list.sort()
            for i in file_list:
                x.append(np.array(Image.open('{}/{}'.format(p,i)),dtype=np.uint8).transpose((2,0,1)))
                print('\rreading {} data...{}'.format(name,len(x)),end='')
        print('\rreading {} data...finish'.format(name))
        x=np.array(x)
        self.data[name]=DataLoader(ImageDataset(x,mode),batch_size=batch_size, shuffle=shuffle)
        return x.shape[1:]
    def train_gan(self,name, generator, discriminator, optimizer, epoch, print_every=1):
        start= time.time()
        generator.train()
        discriminator.train()
        
        generator_optimizer= optimizer[0]
        discriminator_optimizer= optimizer[1]
        
        criterion= torch.nn.BCELoss()
        total_loss= [0,0]
        batch_loss= [0,0]
        
        data_size= len(self.data[name].dataset)
        for i, y in enumerate(self.data[name]):
            batch_index=i+1
            batch_y = Variable(y).cuda()
            # update discriminator
            for j in range(self.discriminator_update_num):
                batch_x = Variable(torch.randn(len(y),self.latent_dim).cuda())
                #loss_gen= torch.mean( -torch.log(1-discriminator(generator(batch_x))))
                #loss_dis= torch.mean( -torch.log(discriminator(batch_y)))
                loss_fake= criterion(discriminator(generator(batch_x)),Variable(torch.zeros(len(y),1)).cuda())
                loss_real= criterion(discriminator(batch_y),Variable(torch.ones(len(y),1)).cuda())
                loss= (loss_fake + loss_real) /2
                '''
                if epoch== 3:
                    self.write(generator(batch_x[: 3]).cpu().data,'./data/gan','gan')
                    print('fake')
                    input()
                    self.write(batch_y[: 3].cpu().data,'./data/gan','gan')
                    print('real')
                    input()
                    print(discriminator(batch_Dx[:10]))
                    print(batch_Dy[:10])
                    print(discriminator(batch_Dx[-10:]))
                    print(batch_Dy[-10:])
                    print(discriminator(batch_Dx).size())
                    print(float(loss))
                    input()
                '''
                discriminator_optimizer.zero_grad()
                loss.backward()
                discriminator_optimizer.step()
                batch_loss[0]+= float(loss)
                #print(float(loss))

            # update generator
            for j in range(self.generator_update_num):
                batch_x = Variable(torch.randn(len(y),self.latent_dim).cuda())
                loss= criterion(discriminator(generator(batch_x)),Variable(torch.ones(len(y),1)).cuda())
                '''
                if epoch== 3:
                    print(discriminator(batch_Dx[:10]))
                    print(batch_Dy[:10])
                    print(discriminator(batch_Dx[-10:]))
                    print(batch_Dy[-10:])
                    print(discriminator(batch_Dx).size())
                    print(float(loss))
                    input()
                '''
                #loss=  torch.mean(-torch.log(discriminator(generator(batch_x))))
                generator_optimizer.zero_grad()
                loss.backward()
                generator_optimizer.step()
                batch_loss[1]+= float(loss)
                #print(float(loss))

            if batch_index% print_every == 0:
                total_loss[0]+= batch_loss[0]/ (self.discriminator_update_num ) if (self.discriminator_update_num!=0) else 0
                total_loss[1]+= batch_loss[1]/ (self.generator_update_num ) if (self.generator_update_num!=0) else 0
                print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)] | G Loss: {:.6f} | D Loss: {:.6f} | Time: {}  '.format(
                                epoch , batch_index*len(batch_x), data_size, 
                                100. * batch_index*len(batch_x)/ data_size,
                                batch_loss[1]/ (self.generator_update_num *print_every) if (self.generator_update_num!=0) else 0,
                                batch_loss[0]/ (self.discriminator_update_num *print_every)if (self.discriminator_update_num!=0) else 0,
                                self.timeSince(start, batch_index*len(batch_x)/ data_size)),end='')
                batch_loss= [0,0]
        print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)] | Total G Loss: {:.6f} | Total D Loss: {:.6f} | Time: {}  '.format(
                        epoch , data_size, data_size, 100. ,
                        float(total_loss[1])/batch_index,float(total_loss[0])/batch_index,
                        self.timeSince(start, 1)))
        self.writer.add_scalar('discriminator loss', float(total_loss[0])/ batch_index, epoch)
        self.writer.add_scalar('generator loss', float(total_loss[1])/ batch_index, epoch)
        print('-'*80)
    def val_gan(self, generator, discriminator, epoch, n=20, path=None):
        generator.eval()
        discriminator.eval()
        
        batch_x = Variable(torch.randn(n,self.latent_dim).cuda())
        predict= generator(batch_x).cpu().data
        self.write(predict,path,'gan')

        self.writer.add_image('sample image result', torchvision.utils.make_grid(predict), epoch)
    def train_vae(self,name, encoder, decoder, optimizer, epoch, kl_coefficient=5E-5, print_every=5):
        start= time.time()
        encoder.train()
        decoder.train()
        
        encoder_optimizer= optimizer[0]
        decoder_optimizer= optimizer[1]
        
        criterion = nn.MSELoss()
        total_loss=torch.zeros(2)
        batch_loss=torch.zeros(2)
        
        data_size= len(self.data[name].dataset)
        for i, x in enumerate(self.data[name]):
            batch_index=i+1
            batch_x= Variable(x).cuda()
            mean_x, logvar_x= encoder(batch_x)
            output= decoder(mean_x, logvar_x,mode='train')
            #output = output.view(-1,output.size()[2])
            reconstruction_loss = criterion(output,batch_x)
            kl_divergence_loss= torch.sum((-1/2)*( 1+ logvar_x- mean_x**2 - torch.exp(logvar_x)))/(len(x))
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            (reconstruction_loss+ kl_coefficient*kl_divergence_loss).backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            batch_loss+= torch.FloatTensor([float(reconstruction_loss), float(kl_divergence_loss)]) # sum up batch loss
            total_loss+= torch.FloatTensor([float(reconstruction_loss), float(kl_divergence_loss)]) # sum up total loss
            if batch_index% print_every == 0:
                print_loss= batch_loss / print_every
                batch_loss=torch.zeros(2)
                print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)] |  Reconstruction Loss: {:.6f} | KL Divergance Loss: {:.6f} | Time: {}  '.format(
                                epoch , batch_index*len(batch_x), data_size, 100. * batch_index*len(batch_x)/ data_size, 
                                float(print_loss[0]),float(print_loss[1]),
                                self.timeSince(start, batch_index*len(batch_x)/ data_size)),end='')
        print('\nTime: {} | Total  Reconstruction Loss: {:.6f} | Total KL Divergance Loss: {:.6f}  '.format(self.timeSince(start,1),
                    float(total_loss[0])/batch_index, float(total_loss[1])/batch_index))
        print('-'*80)
    def val_vae(self,name, encoder, decoder, optimizer, epoch, print_every=5, sample_n= 50, record=0, path=None):
        start= time.time()
        encoder.eval()
        decoder.eval()

        criterion = nn.MSELoss()
        total_loss=torch.zeros(2)
        predict=[]
        
        data_size= len(self.data[name].dataset)
        for i, x in enumerate(self.data[name]):
            batch_index=i+1
            batch_x= Variable(x).cuda()
            mean_x, logvar_x= encoder(batch_x)
            output= decoder(mean_x, logvar_x, mode='test')
            reconstruction_loss = criterion(output,batch_x)
            kl_divergence_loss= torch.sum((-1/2)*( 1+ logvar_x- mean_x**2 - torch.exp(logvar_x)))/(len(x))
            # loss
            total_loss+= torch.FloatTensor([float(reconstruction_loss), float(kl_divergence_loss)]) # sum up total loss
            # predict
            predict.extend(output.cpu().data.unsqueeze(0))
            if batch_index % print_every == 0:
                print('\rVal: [{}/{} ({:.0f}%)] | Time: {}'.format(
                         batch_index * len(x), data_size,
                        100. * batch_index* len(x) / data_size,
                        self.timeSince(start, batch_index*len(x)/ data_size)),end='')

        total_loss/=  batch_index
        predict=torch.cat(predict,0)

        sample_x = Variable(torch.normal(torch.zeros(sample_n,self.latent_dim))).cuda()
        predict= torch.cat((decoder(sample_x,mode='test').cpu().data,predict),0)
        self.write(predict,path)
        print('\nVal set: Average Reconstruction Loss: {:.6f} | Average KL Divergance Loss: {:.6f} | Time: {}  '.format(float(total_loss[0]),float(total_loss[1]),
                        self.timeSince(start, batch_index*len(batch_x)/ data_size)))
        print('-'*80)
        if float(total_loss[0])< record and path != None:
            self.write(predict,path)
            record=float(total_loss[0])
        return record
    def write(self,data,path,mode):
        data=data.numpy()
        #print(output.shape)
        for i in range(data.shape[0]):
            if mode== 'vae':
                im=data[i].transpose((1,2,0))*255
            elif mode== 'gan':
                im=(data[i].transpose((1,2,0))*128)+127
            else: raise ValueError('Wrong mode')
            im=im.astype(np.uint8)
            image = Image.fromarray(im,'RGB')
            image.save('{}/{:0>4}.png'.format(path,i))
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
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.output_size=output_size
        self.conv1 = nn.Sequential(                 # input shape (1, 28, 28)
            nn.Conv2d(input_size[0], hidden_size, 3, 1, 1),              # output shape (16, 28, 28)
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),            # output shape (16,  7,  7)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_size,hidden_size, 3, 1, 1),         
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),            
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d( hidden_size,hidden_size, 3, 1, 1),         
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),            
        )
        self.den1= nn.Sequential(
            nn.Linear( hidden_size*(input_size[1]//8)*(input_size[2]//8),  output_size*2),
            nn.BatchNorm1d(output_size*2),
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x= self.den1(x)
        return x[:,:self.output_size], x[:,self.output_size:]
class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.output_size=output_size
        self.den1= nn.Sequential(
            nn.Linear(input_size, (output_size[1]//8)* (output_size[2]//8) *hidden_size),
            nn.BatchNorm1d((output_size[1]//8)* (output_size[2]//8) *hidden_size),
            nn.ReLU(),
        )
        self.conv1 = nn.Sequential(                 # input shape (1, 28, 28)
            nn.ConvTranspose2d(hidden_size,hidden_size, 4, stride=2, padding=1),# output shape (16, 28, 28)
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_size, hidden_size ,4, stride=2, padding=1),# output shape (16, 28, 28)
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(hidden_size, output_size[0],4, stride=2, padding=1),# output shape (16, 28, 28)
            nn.BatchNorm2d(output_size[0]),
            nn.Sigmoid(),
        )
    def forward(self, mean_x, logvar_x=0 , mode='test'):
        if mode== 'train':
            epsilon_x= torch.normal(torch.zeros_like(mean_x))
            x= mean_x + epsilon_x*torch.exp((1/2)*logvar_x)
        elif mode== 'test':
            x= mean_x
        else: raise ValueError('Wrong mode')
        x= self.den1(x)
        x = x.view(x.size(0), -1,(self.output_size[1]//8), (self.output_size[2]//8))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x 
class Generator(nn.Module):
    def __init__(self, input_size ):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(input_size, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3 , 4, 2, 1, bias=False),
            nn.BatchNorm2d(3),
            nn.Tanh())
    def forward(self, x):
        x = x.unsqueeze(2).unsqueeze(3)
        x = self.main(x)
        return x 
    def make_layers(self, input_channel, cfg,  batch_norm=False):
        #cfg = [(64,2), (64,2)]
        layers = []
        in_channels = input_channel
        extend=1
        for v in cfg[:-1]:
            conv2d = nn.ConvTranspose2d( in_channels, v[0], kernel_size=2+v[1], stride=v[1], padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v[0]), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v[0]
            extend*=v[1]
        conv2d = nn.ConvTranspose2d( in_channels, cfg[-1][0], kernel_size=2+cfg[-1][1], stride=cfg[-1][1], padding=1)
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(cfg[-1][0]), nn.Tanh()]
        else:
            layers += [conv2d, nn.Tanh()]
        extend*=cfg[-1][1]
        return nn.Sequential(*layers), extend
    def optimizer(self, lr=0.001):
        return torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999))
class Discriminator(nn.Module):
    def __init__(self ):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0),-1)
        return x
    def make_layers(self, input_channel, cfg,  batch_norm=False):
        #cfg = [(64,2), (64,2)]
        layers = []
        in_channels = input_channel
        compress=1
        for v in cfg[:-1]:
            conv2d = nn.Conv2d( in_channels, v[0], kernel_size=2+v[1], stride=v[1], padding=1,bias=False)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v[0]),nn.LeakyReLU(0.2)]
            else:
                layers += [conv2d, nn.LeakyReLU(0.2)]
            in_channels = v[0]
            compress*=v[1]
        conv2d = nn.Conv2d( in_channels, cfg[-1][0], kernel_size=2+cfg[-1][1], stride=cfg[-1][1], padding=1,bias=False)
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(cfg[-1][0]),nn.Sigmoid()]
        else:
            layers += [conv2d, nn.Sigmoid()]
        compress*=cfg[-1][1]
        return nn.Sequential(*layers), compress
    def optimizer(self, lr=0.001):
        return torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999))
class ImageDataset(Dataset):
    def __init__(self, data, mode):
        self.data=data
        self.mode= mode
    def __getitem__(self, i):
        if self.mode=='vae':
            x=torch.FloatTensor(self.data[i][:])/255
        elif self.mode=='gan':
            x=(torch.FloatTensor(self.data[i][:])-127)/128
        else: raise ValueError('Wrong mode.')
        return x
    def __len__(self):
        return len(self.data)
