

import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Conv2DTranspose, Activation, Dropout, ZeroPadding2D, UpSampling2D, Add
from keras.layers import BatchNormalization
from keras.utils import to_categorical
from keras import backend as K
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os, random, time, math
assert keras and Image and np and to_categorical and Dense and Flatten and K and Dropout and ZeroPadding2D and plt
assert BatchNormalization and Add and UpSampling2D and DataLoader

class Datamanager():
    def __init__(self):
        self.data={}
        self._label2pix =[(0,255,255),(255,255,0),(255,0,255),(0,255,0),(0,0,255),(255,255,255),(0,0,0)]
    def label2pix(self,data):
        '''
        tranform data to categorical
        '''
        masks = np.empty((512, 512,3),dtype=np.uint8)
        mask  = np.empty((512, 512),dtype=np.uint8)
        mask [ data == 0] = 3  # (Cyan: 011) Urban land 
        mask [ data == 1] = 6  # (Yellow: 110) Agriculture land 
        mask [ data == 2] = 5  # (Purple: 101) Rangeland 
        mask [ data == 3] = 2  # (Green: 010) Forest land 
        mask [ data == 4] = 1  # (Blue: 001) Water 
        mask [ data == 5] = 7  # (White: 111) Barren land 
        mask [ data == 6] = 0  # (Black: 000) Unknown 
        masks[:,:,0] =(mask//4)%2
        masks[:,:,1] =(mask//2)%2
        masks[:,:,2] = mask%2
        masks=masks*255

        return masks.astype(np.uint8)
    def pix2label(self,data):
        '''
        tranform data to categorical
        '''
        masks = np.empty((512, 512),dtype=np.uint8)

        mask = (data >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        masks[ mask == 3] = 0  # (Cyan: 011) Urban land 
        masks[ mask == 6] = 1  # (Yellow: 110) Agriculture land 
        masks[ mask == 5] = 2  # (Purple: 101) Rangeland 
        masks[ mask == 2] = 3  # (Green: 010) Forest land 
        masks[ mask == 1] = 4  # (Blue: 001) Water 
        masks[ mask == 7] = 5  # (White: 111) Barren land 
        masks[ mask == 0] = 6  # (Black: 000) Unknown 
        masks[ mask == 4] = 6  # (Black: 000) Unknown 

        return masks.astype(np.uint8)
    def get_data(self, mode, name, path, train_file='./data/train_x.npy', label_file= './data/train_y.npy',validation_split=0):
        x=[]
        y=[]
        if mode=='test':
            file_list = [file for file in os.listdir(path) if file.endswith('.jpg')]
            file_list.sort()
            for i in range(len(file_list)):
                x.append(np.array(Image.open('{}/{}'.format(path,file_list[i]))))
                print('\rreading data...{}'.format(i),end='')
            print('\nreading data...finish')
            x =np.array(x).astype(np.uint8)
            self.data[name]=x/255
        elif mode=='train':
            if os.path.isfile(train_file) and os.path.isfile(label_file):
                x=np.load(train_file)
                y=np.load(label_file)
            else:
                x_list = [file for file in os.listdir(path) if file.endswith('.jpg')]
                y_list = [file for file in os.listdir(path) if file.endswith('.png')]
                x_list.sort()
                y_list.sort()
                for i in range(len(x_list)):
                    x.append(np.array(Image.open('{}/{}'.format(path,x_list[i]))).astype(np.uint8))
                    y.append(self.pix2label(np.array(Image.open('{}/{}'.format(path,y_list[i]))).astype(np.uint8)))
                    print('\rreading data...{}'.format(i),end='')
                print('\nreading data...finish')
                x, y =np.array(x).astype(np.uint8), np.array(y).astype(np.uint8)
                np.save(train_file,x)
                np.save(label_file,y)
            self.data[name]=[x,y]
        else: raise ValueError('Wrong mode.')
    def generate(self, data, batch_size, shuffle):
        [x,y]=data
        index= list(range(len(x)))
        while True:
            if shuffle: random.shuffle(index)
            for i in range(0,len(index)-batch_size,batch_size):
                batch_x=x[i:i+batch_size]/255
                batch_y=to_categorical(y[i:i+batch_size],7)
                yield batch_x,batch_y
            batch_x=x[(len(index)//batch_size)*batch_size:]/255
            batch_y=to_categorical(y[(len(index)//batch_size)*batch_size:],7)
            yield batch_x,batch_y
    def dataloader(self,data,batch_size, shuffle=True):
        if isinstance(data,list):
            return DataLoader(VideoDataset(data[0],data[1]), batch_size=batch_size, shuffle=shuffle)
        else: 
            return DataLoader(VideoDataset(data), batch_size=batch_size, shuffle=shuffle)
    def model(self,input_shape, path=None, dropout_rate=0.3):
        # Block 1
        img_input = Input(shape=input_shape)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        o_block_4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(o_block_4)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        # Classification block
        x = Conv2D(4096, (3, 3), padding='same', name='conv1')(x)
        x = BatchNormalization()(x)
        x = Activation(activation='selu')(x)
        x = Dropout(dropout_rate)(x)
        x = Conv2D(4096, (1, 1), padding='same', name='conv2')(x)
        x = BatchNormalization()(x)
        x = Activation(activation='selu')(x)
        x = Conv2D(512, (1, 1), padding='same', name='conv3')(x)
        x = BatchNormalization()(x)
        x = Activation(activation='selu')(x)
        x = Dropout(dropout_rate)(x)
        x = Conv2DTranspose(512, (4, 4), padding='same',strides=2, name='convtranspose1')(x)
        x = Add()([x,o_block_4])
        x = Conv2DTranspose(  7, (32,32), padding='same',strides=16, name='convtranspose2')(x)
        x = BatchNormalization()(x)
        x = Activation('softmax')(x)

        # Create model.
        model = Model(img_input, x, name='vgg16')
        if path != None: model.load_weights(path,by_name=True)
        #for l in model.layers[:19]:
            #l.trainable=False
        return model
    def fcn32(self,path, dropout):
        """VGG 16-layer model (configuration "D") with batch normalization
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        model = FCN32(torch.load(path),dropout)
        return model
    def train(self,model,trainloader,optimizer, criterion, epoch, print_every=5):
        start= time.time()
        model.train()

        total_loss=0
        batch_loss=0
        
        for i, (x, y) in enumerate(trainloader):
            batch_index=i+1
            batch_x, batch_y= Variable(x,requires_grad=True).cuda(), Variable(y).cuda().long()
            output = model(batch_x)
            #output = output.view(-1,output.size()[2])
            loss = criterion(output,batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss+= loss.data[0]* len(batch_x) # sum up batch loss
            total_loss+= loss.data[0]* len(batch_x) # sum up total loss
            if batch_index% print_every == 0:
                print_loss= batch_loss / print_every
                batch_loss= 0
                print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)] |  Loss: {:.6f} | Time: {}  '.format(
                                epoch , batch_index*len(batch_x), len(trainloader.dataset),
                                100. * batch_index*len(batch_x)/ len(trainloader.dataset), print_loss,
                                self.timeSince(start, batch_index*len(batch_x)/ len(trainloader.dataset))),end='')
        print('\nTime: {} | Total loss: {:.4f}'.format(self.timeSince(start,1),total_loss/batch_index))
        print('-'*60)
    def val(self,model,valloader,optimizer, criterion, print_every=5):
        model.eval()
        test_loss = 0
        correct = 0
        predict=[]
        target=[]
        for batch_index ,(x, y) in enumerate(valloader):
            x, y = Variable(x, volatile=False).cuda(), Variable(y,volatile=False).cuda()
            output = model(x)
            loss = criterion(output,y)
            # loss
            test_loss += loss.data[0]* len(x)
            # accu
            pred = output.data.max(1,keepdim=True)[1].squeeze(1) # get the index of the max log-probability
            correct += pred.eq(y.data).long().cpu().sum()
            # mean iou
            predict.extend(pred.unsqueeze(0))
            target.extend(y.unsqueeze(0))
            if batch_index % print_every == 0:
                print('\rVal: [{}/{} ({:.0f}%)]'.format(
                         batch_index * len(x), len(valloader.dataset),
                        100. * batch_index / len(valloader)),end='')

        test_loss /= len(valloader.dataset)
        predict=torch.cat(predict,0)
        target=torch.cat(target,0)
        miou= self.mean_iou_score(predict,target)
        print('\nVal set: Average loss: {:.4f} | Accuracy: {}/{} ({:.0f}%) | Mean IOU: {:.4f}  '.format(
                test_loss, correct, len(valloader.dataset)*valloader.dataset.label.size()[1]*valloader.dataset.label.size()[2],
                100. * correct / (len(valloader.dataset)*valloader.dataset.label.size()[1]*valloader.dataset.label.size()[2]), miou))
        print('-'*60)
    def test(self,model,trainloader):
        model.eval()
        pred=[]
        for x in trainloader:
            pred.extend(torch.max(model(Variable(x).cuda()),1)[1].cpu().data.unsqueeze(0))
        pred=torch.cat(pred,0)
        return pred.numpy()
    def write(self,data,path):
        output=data
        #print(output.shape)
        for i in range(output.shape[0]):
            im=self.label2pix(output[i])
            image = Image.fromarray(im,'RGB')
            image.save('{}/{:0>4}_mask.png'.format(path,i))
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
    def count_parameters(self,model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    def mean_iou_score(self,pred, labels):
        '''
        Compute mean IoU score over 6 classes
        '''
        mean_iou = 0
        for i in range(6):
            tp_fp = np.sum(pred == i)
            tp_fn = np.sum(labels == i)
            tp = np.sum((pred == i) * (labels == i))
            iou = tp / (tp_fp + tp_fn - tp)
            mean_iou += iou / 6
            #print('class #%d : %1.5f'%(i, iou))
        #print('\nmean_iou: %f\n' % mean_iou)
        return np.array(mean_iou)
class FCN32(nn.Module):
    def __init__(self, pretrained_dict,dropout=0.1):
        super(FCN32, self).__init__()
        self.features = self.make_layers( batch_norm=False)
        self.transposeconv= nn.Sequential(
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.SELU(True),
            nn.Dropout(dropout),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.SELU(True),
            nn.Dropout(dropout),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.SELU(True),
            nn.Dropout(dropout),
            nn.ConvTranspose2d(512,7,kernel_size=64,padding=16, stride=32),
        )
        model_dict = self.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        self.load_state_dict(model_dict)
        # 4. set untrainable
        for p in self.features.parameters():
                p.requires_grad=True
        for p in self.transposeconv.parameters():
                p.requires_grad=True
        # 5. initialize weight
        self._initialize_weights(self.transposeconv)
    def make_layers(self, batch_norm=False):
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.features(x)
        x = self.transposeconv(x)
        return x
    def _initialize_weights(self,model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
class VideoDataset(Dataset):
    def __init__(self, data, label=None):
        self.data=data
        if label is not None: self.label=torch.Tensor(label).long()
        else: self.label= None
    def __getitem__(self, i):
        x=torch.Tensor(self.data[i]).permute(2,0,1)/255
        if self.label is not None:
            y=torch.Tensor(self.label[i]).long()
            return x,y
        else:return x
    def __len__(self):
        return len(self.data)
