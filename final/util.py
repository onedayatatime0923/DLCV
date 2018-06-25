
import os, time, math, random
from skimage import io
import numpy as np
from scipy import misc
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from torchvision import models
from torch.utils.data import Dataset,DataLoader
from tensorboardX import SummaryWriter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA, TruncatedSVD
assert Variable and F and DataLoader


class DataManager():
    def __init__(self, tensorboard_dir=None, character_file=None):
        self.character= Character(character_file)
        if tensorboard_dir== None: self.writer=None
        else: self.tb_setting(tensorboard_dir)
    def tb_setting(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        for f in os.listdir(directory): 
            os.remove('{}/{}'.format(directory,f))
        self.writer = SummaryWriter(directory)
    def readfile(self, dir_path, label_path, save_path ):
        if os.path.isfile(save_path[0]) and os.path.isfile(save_path[1]):
            x = np.load(save_path[0])
            y = np.load(save_path[1])
            return x, y
        file_list = [filename for filename in os.listdir(dir_path)]
        file_list.sort()
        x, y = [], [] 
        with open(label_path) as f:
                labels = [line.strip().split(' ')[1] for line in f]
        for idx, file in enumerate(file_list):
                img = io.imread(os.path.join(dir_path, file))
                #img = skimage.transform.resize(img,(224,224), preserve_range=True)
                x.append(img)
                y.append(self.character.addCharacter(labels[idx]))
                print('\rreading image from {}...{}'.format(dir_path,len(x)),end='')
        print('\rreading image from {}...finished'.format(dir_path))
        x = np.array(x)
        y = np.array(y)
        np.save(save_path[0],x)
        np.save(save_path[1],y)
        return x, y
    def train_classifier(self, model, dataloader, epoch, optimizer, print_every= 2):
        start= time.time()
        model.train()
        
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
            correct = int(pred.eq(y.data).long().cpu().sum())
            batch_correct += correct/ len(x)
            total_correct += correct
            if batch_index% print_every== 0:
                print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)] | Loss: {:.6f} | Accu: {:.4f}% | Time: {}  '.format(
                            epoch , batch_index*len(x), data_size, 100. * batch_index*len(x)/ data_size,
                            batch_loss/ print_every, 100.* batch_correct/ print_every,
                            self.timeSince(start, batch_index*len(x)/ data_size)),end='')
                batch_loss= 0
                batch_correct= 0
        print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)] | Loss: {:.6f} | Accu: {:.4f}% | Time: {}  '.format(
                    epoch , data_size, data_size, 100.,
                    float(total_loss)/ data_size, 100.*total_correct/ data_size,
                    self.timeSince(start, 1)))
        if self.writer != None:
            self.writer.add_scalar('Train Loss', float(total_loss)/ data_size, epoch)
            self.writer.add_scalar('Train Accu',  100.*total_correct/ data_size, epoch)
        return float(total_loss)/ data_size, 100. * total_correct/ data_size
    def val_classifier(self,model,dataloader, epoch, print_every= 2):
        start= time.time()
        model.eval()
        
        criterion= nn.CrossEntropyLoss()
        total_loss= 0
        batch_loss= 0
        total_correct= 0
        batch_correct= 0
        
        data_size= len(dataloader.dataset)
        for b, (x, y) in enumerate(dataloader):
            with torch.no_grad():
                batch_index=b+1
                x, y= Variable(x).cuda(), Variable(y).squeeze(1).cuda()
                output= model(x)
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
                    print('\rVal Epoch: {} | [{}/{} ({:.0f}%)] | Loss: {:.6f} | Accu: {:.4f}% | Time: {}  '.format(
                            epoch , batch_index*len(x), data_size, 100. * batch_index*len(x)/ data_size,
                            batch_loss/ print_every, 100.* batch_correct/ print_every,
                            self.timeSince(start, batch_index*len(x)/ data_size)),end='')
                    batch_loss= 0
                    batch_correct= 0
        print('\rVal Epoch: {} | [{}/{} ({:.0f}%)] | Loss: {:.6f} | Accu: {:.4f}% | Time: {}  '.format(
                    epoch , data_size, data_size, 100.,
                    float(total_loss)/ data_size, 100.*total_correct/ data_size,
                    self.timeSince(start, 1)))
        if self.writer != None:
            self.writer.add_scalar('Val Loss', float(total_loss)/ data_size, epoch)
            self.writer.add_scalar('Val Accu',  100.*total_correct/ data_size, epoch)
        return float(total_loss)/ data_size, 100. * total_correct/ data_size
    def train_AE(self, model, dataloader, epoch, optimizer, print_every= 10):
        start= time.time()
        model.train()
        
        criterion= nn.MSELoss()
        total_loss= 0
        batch_loss= 0
        
        data_size= len(dataloader.dataset)
        for b, x  in enumerate(dataloader):
            batch_index=b+1
            x = Variable(x).cuda()
            output= model(x)
            loss = criterion(output,x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # loss
            batch_loss+= float(loss)
            total_loss+= float(loss)* len(x)
            if batch_index% print_every== 0:
                print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)] | Loss: {:.6f} | Time: {}  '.format(
                            epoch , batch_index*len(x), data_size, 100. * batch_index*len(x)/ data_size,
                            batch_loss/ print_every,
                            self.timeSince(start, batch_index*len(x)/ data_size)),end='')
                batch_loss= 0
        print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)] | Loss: {:.6f} | Time: {}  '.format(
                    epoch , data_size, data_size, 100.,
                    float(total_loss)/ data_size,
                    self.timeSince(start, 1)))
        if self.writer != None:
            self.writer.add_scalar('Train Loss', float(total_loss)/ data_size, epoch)
        return float(total_loss)/ data_size
    def val_AE(self,model,dataloader, epoch, print_every= 10):
        start= time.time()
        model.eval()
        
        criterion= nn.MSELoss()
        total_loss= 0
        batch_loss= 0
        
        data_size= len(dataloader.dataset)
        for b, x in enumerate(dataloader):
            batch_index=b+1
            x = Variable(x).cuda()
            output= model(x)
            loss = criterion(output,x)
            # loss
            batch_loss+= float(loss)
            total_loss+= float(loss)* len(x)
            if batch_index% print_every== 0:
                print('\rVal Epoch: {} | [{}/{} ({:.0f}%)] | Loss: {:.6f} | Time: {}  '.format(
                            epoch , batch_index*len(x), data_size, 100. * batch_index*len(x)/ data_size,
                            batch_loss/ print_every,
                            self.timeSince(start, batch_index*len(x)/ data_size)),end='')
                batch_loss= 0
        print('\rVal Epoch: {} | [{}/{} ({:.0f}%)] | Loss: {:.6f} | Time: {}  '.format(
                    epoch , data_size, data_size, 100.,
                    float(total_loss)/ data_size,
                    self.timeSince(start, 1)))
        if self.writer != None:
            self.writer.add_scalar('Val Loss', float(total_loss)/ data_size, epoch)
        return float(total_loss)/ data_size
    def dimension_reduction_model(self, model, data, save_path):
        if os.path.isfile(save_path):
            x = np.load(save_path)
            return x
        transform= torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        feature = []
        for i in range(len(data)):
            #print(transform(torch.FloatTensor(data[i]).permute(2,0,1)/255).cuda().unsqueeze(0).size())
            x=model(transform(torch.FloatTensor(data[i]).permute(2,0,1)/255).cuda().unsqueeze(0)).cpu()
            feature.append(x.data)
            print('\rdimension reduction ...{}'.format(len(feature)),end='')
        print('\rreading image from ...finished')
        feature= torch.cat(feature,0).numpy()
        np.save(save_path, feature)
        return feature
    def pca_construct(self, x, n= 128):
        pca= PCA(n_components=n, svd_solver="randomized")
        pca.fit(x)
        return pca
    def pca_predict(self, pca, x):
        return pca.transform(x)
    def svd_construct(self, x, n=2):
        svd= TruncatedSVD(n_components=n)
        svd.fit(x)
        return svd
    def svd_predict(self, svd, x):
        return svd.transform(x)
    def knn_construct(self, x, y, n=3):
        neigh = KNeighborsClassifier(n_neighbors=n)
        neigh.fit(x, y) 
        return neigh
    def knn_predict(self, neigh, x):
        return neigh.predict(x)
    def naive_bayes_construct(self, x, y):
         nb = GaussianNB().fit(x, y)
         return nb
    def naive_bayes_predict(self, nb, x):
        return nb.predict(x)
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

class CNN(nn.Module):
    def __init__(self, dropout, pretrained=False):
        super(CNN, self).__init__()
        self.conv = models.vgg16_bn(pretrained=pretrained).features
        self.fc = nn.Sequential(
            nn.Linear(512 * 6 * 5, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 2360),
        )
        self.transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self._initialize_weights()
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x
    def save(self, path):
        torch.save(self,path)
    def _initialize_weights(self):
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

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, label_dim, dropout ):
        super(Classifier, self).__init__()
        self.conv = models.vgg16_bn(pretrained=False).features
        self.fc = nn.Sequential(
            nn.Linear(512 * (input_dim[1]//32) * (input_dim[2]//32), hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, label_dim),
        )
        self._initialize_weights()
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    def _initialize_weights(self):
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
    def save(self, path):
        torch.save(self,path)
class AutoEncoder(nn.Module):
    def __init__(self, dropout):
        super(AutoEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(8,16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True))

        self.transeposeconv = nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=1),
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(64, 32, kernel_size=(5,4), stride=2, padding=1),
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(16,  8, kernel_size=5, stride=2, padding=1),
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(  8,   3, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True))
    def forward(self, x):
        x = self.conv(x)
        x = self.transeposeconv(x)
        return x
    def save(self, path):
        torch.save(self,path)
class Encoder(nn.Module):
    def __init__(self, load_path):
        super(Encoder, self).__init__()
        self.conv= torch.load(load_path).conv
    def forward(self, x):
        #print(x.size())
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        return x
    def save(self, path):
        torch.save(self,path)

class Feature_extractor(nn.Module):
    def __init__(self, dropout, pretrained=True):
        super(CNN, self).__init__()
        self.conv = models.vgg16_bn(pretrained=pretrained).features
    def forward(self, x):
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x
    def save(self, path):
        torch.save(self,path)

class Character:
    def __init__(self, character_file= None):
        self.character2index= {}
        self.index2character= {}
        self.n_character = 0
        if character_file != None:
            self.load(character_file)
    def addCharacter(self, character):
        character=int(character)
        if character not in self.character2index:
            self.character2index[character] = self.n_character
            self.index2character[self.n_character] = character
            self.n_character += 1
        return self.character2index[character]
    def save(self, path):
        index_list= sorted( self.character2index, key= self.character2index.get)
        with open( path, 'w') as f:
            f.write('\n'.join([str(i) for i in index_list]))
    def load(self, path):
        self.character2index= {}
        self.index2character= {}
        self.color2index= {}
        self.color2count= {}
        self.index2color= {}
        with open(path,'r') as f:
            for line in f:
                character=line.replace('\n','')
                self.addCharacter(character)

class EasyDataset(Dataset):
    def __init__(self, image, label, flip = True, rotate = True, angle = 5):
        self.image = image
        self.label = label

        self.flip_n= int(flip)+1
        self.rotate= rotate
        self.transform= torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.angle= angle
    def __getitem__(self, i):
        index= i// self.flip_n 
        flip = bool( i % self.flip_n )

        if flip == True: x= np.flip(self.image[index],1).copy()
        else: x= self.image[index]
        if self.rotate: x= misc.imrotate(x, random.uniform(-self.angle, self.angle))
        x=self.transform(torch.FloatTensor(x).permute(2,0,1)/255)

        y=torch.LongTensor([self.label[index]])
        return x,y
    def __len__(self):
        return len(self.image)*self.flip_n
class ImageDataset(Dataset):
    def __init__(self, image=None, label=None):
        self.image = image
        self.label = label
        self.transform= torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    def aim(self, target):
        self.target = target
        index=[]
        for i in range(len(self.label)):
            for t in range(len(target)):
                if self.label[i] in target[t]:
                    index.append([i,t])
        #print(index)
        #input()
        self.index = index
        return self
    def __getitem__(self, i):
        #x=self.transform(torch.FloatTensor(self.image[self.index[i][0]]).permute(2,0,1)/255)
        x=torch.FloatTensor(self.image[self.index[i][0]]).permute(2,0,1)/255
        y=torch.LongTensor([self.index[i][1]])
        return x,y
    def __len__(self):
        return len(self.index)
class AEDataset(Dataset):
    def __init__(self, image=None, label=None):
        self.image = image
        self.transform= torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    def __getitem__(self, i):
        x=self.transform(torch.FloatTensor(self.image[i]).permute(2,0,1)/255)
        return x
    def __len__(self):
        return len(self.image)
class ClassifierDataset(Dataset):
    def __init__(self, image=None, label=None):
        self.image = image
        self.label = label
    def __getitem__(self, i):
        x=torch.FloatTensor(self.image[i])
        y=torch.LongTensor([self.label[i]])
        return x,y
    def __len__(self):
        return len(self.image)
