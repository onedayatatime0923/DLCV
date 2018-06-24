
import torch
from torch.utils.data import DataLoader
from util import DataManager, AutoEncoder, AEDataset
import argparse


parser = argparse.ArgumentParser(description='DLCV HW5')
#parser.add_argument('-p','--problem', dest='problem',type=int,required=True)
args = parser.parse_args()

TENSORBOARD_DIR= './runs/train'


dm= DataManager(tensorboard_dir= TENSORBOARD_DIR)

EPOCH = 50
BATCH_SIZE = 128
LABEL_DIM = 11
DROPOUT = 0.5
LEARNING_RATE = 1E-3
PRETRAIN = True
OUTPUT_PATH = './model/pretrained.pt'
OUTPUT_CHARACTER = 'data/character.txt'

train_path=['./data/trainx.npy','./data/trainy.npy']
val_path=['./data/valx.npy','./data/valy.npy']
val_data=dm.readfile('./dataset/val', './dataset/val_id.txt', save_path=val_path)
train_data=dm.readfile('./dataset/train/', './dataset/train_id.txt', save_path=train_path)
#dm.character.save(OUTPUT_CHARACTER)

model= AutoEncoder(PRETRAIN).cuda()

optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)

train_dataloader= DataLoader(AEDataset(train_data[0]),batch_size= BATCH_SIZE, shuffle= True)
val_dataloader= DataLoader(AEDataset(val_data[0]),batch_size= BATCH_SIZE, shuffle= False)

accu_record=100
for epoch in range(1,EPOCH+1):
    dm.train_AE( model, train_dataloader, epoch, optimizer)
    record=dm.val_AE( model, val_dataloader, epoch)
    if record< accu_record:
        model.save(OUTPUT_PATH)
        accu_record= record
        print('model saved!!!')
    print('='*80)
