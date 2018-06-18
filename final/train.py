
import torch
from torch.utils.data import DataLoader
from util import DataManager, CNN, ImageDataset, Classifier
import argparse
assert torch and DataLoader and CNN and ImageDataset


parser = argparse.ArgumentParser(description='DLCV Final')
#parser.add_argument('-p','--problem', dest='problem',type=int,required=True)
args = parser.parse_args()

TENSORBOARD_DIR= './runs/train'


dm= DataManager(tensorboard_dir= TENSORBOARD_DIR)

EPOCH = 50
BATCH_SIZE =  16
HIDDEN_DIM = 4096
DROPOUT = 0.5
LEARNING_RATE = 1E-3
OUTPUT_PATH = './model/model.pt'
OUTPUT_CHARACTER = 'data/character.txt'
TARGET = [[i] for i in range(50)]

val_path=['./data/valx.npy','./data/valy.npy']
train_path=['./data/trainx.npy','./data/trainy.npy']
val_data=dm.readfile('./dataset/val', './dataset/val_id.txt', save_path=val_path)
train_data=dm.readfile('./dataset/train/', './dataset/train_id.txt', save_path=train_path)
print('train_x shape:{}'.format(train_data[0].shape))
print('train_y shape:{}'.format(train_data[1].shape))
print('val_x shape:{}'.format(val_data[0].shape))
print('val_y shape:{}'.format(val_data[1].shape))

input_dim = (train_data[0][0].shape[2],*train_data[0][0].shape[:2])
label_dim = len(TARGET)
print('input_dim: {}'.format(input_dim))
print('label_dim: {}'.format(label_dim))

model= Classifier(input_dim, HIDDEN_DIM, label_dim, DROPOUT).cuda()

train_dataset = ImageDataset(*train_data).aim(TARGET)
train_dataloader= DataLoader(train_dataset, batch_size= BATCH_SIZE, shuffle= True)
val_dataset = ImageDataset(*val_data).aim(TARGET)
val_dataloader= DataLoader(val_dataset, batch_size= BATCH_SIZE, shuffle= False)

optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)


accu_record=0
for epoch in range(1,EPOCH+1):
    dm.train_classifier( model, train_dataloader, epoch, optimizer)
    record=dm.val_classifier( model, val_dataloader, epoch)
    if record[1]> accu_record:
        model.save(OUTPUT_PATH)
        accu_record= record[1]
        print('model saved')
    print('='*80)
