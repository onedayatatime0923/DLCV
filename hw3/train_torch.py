
from util import Datamanager
import torch
import torch.nn as nn
assert torch and nn

TRAIN_SIZE= 2313
VAL_SIZE= 257
EPOCHS= 100
BATCH_SIZE=  8
DROPOUT=0.1

dm = Datamanager()


dm.get_data('train', 'train',  './data/train', TRAIN_SIZE,'./data/train_x.npy','./data/train_y.npy')
dm.get_data('train', 'val',  './data/validation', VAL_SIZE,'./data/val_x.npy','./data/val_y.npy')
print('train_x shape: {}'.format(dm.data['train'][0].shape))
print('train_y shape: {}'.format(dm.data['train'][1].shape))
print('val_x shape: {}'.format(dm.data['val'][0].shape))
print('val_y shape: {}'.format(dm.data['val'][1].shape))
trainloader= dm.dataloader(dm.data['train'],BATCH_SIZE,shuffle=True)
valloader= dm.dataloader(dm.data['val'],BATCH_SIZE,shuffle=True)

model=dm.fcn32('./data/vgg16.pth',DROPOUT).cuda()
torch.save(model,'./data/model.pt')
print(model)
print('Total Parameter: {}'.format(dm.count_parameters(model)))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())   # optimize all parameters
for epoch in range(1,EPOCHS+1):
    dm.train(model,trainloader,optimizer, criterion, epoch, print_every=5)
    dm.val(model,valloader,optimizer, criterion, print_every=5)

torch.save(model,'./data/model.pt')
