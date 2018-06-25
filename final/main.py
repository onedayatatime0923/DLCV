

import torch
from torch.utils.data import DataLoader
from util import DataManager, CNN, EasyDataset
import argparse
assert torch and DataLoader and CNN and EasyDataset


parser = argparse.ArgumentParser(description='DLCV Final')
#parser.add_argument('-p','--problem', dest='problem',type=int,required=True)
args = parser.parse_args()

TENSORBOARD_DIR= './runs/train'

dm= DataManager(tensorboard_dir= TENSORBOARD_DIR)

EPOCH = 50
BATCH_SIZE = 32
LEARNING_RATE = 1E-4
DROPOUT = 0.5
PRETRAIN = False
OUTPUT_PATH = './model/model.pt'
OUTPUT_CHARACTER = 'data/character.txt'

train_path=['./data/trainx.npy','./data/trainy.npy']
val_path=['./data/valx.npy','./data/valy.npy']
val_data=dm.readfile('./dataset/val', './dataset/val_id.txt', save_path=val_path)
train_data=dm.readfile('./dataset/train/', './dataset/train_id.txt', save_path=train_path)
#dm.character.save(OUTPUT_CHARACTER)

model= CNN(DROPOUT, PRETRAIN).cuda()

optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)

train_dataloader= DataLoader(EasyDataset(*train_data),batch_size= BATCH_SIZE, shuffle= True, num_workers = 0)
val_dataloader= DataLoader(EasyDataset(*val_data, flip = False, rotate = False),batch_size= BATCH_SIZE, shuffle= False, num_workers = 0)

accu_record=0
for epoch in range(1,EPOCH+1):
    dm.train_classifier( model, train_dataloader, epoch, optimizer)
    record=dm.val_classifier( model, val_dataloader, epoch)
    if record[1]> accu_record:
        #model.save(OUTPUT_PATH)
        accu_record= record[1]
    print('='*80)
