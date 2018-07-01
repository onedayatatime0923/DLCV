

import torch
from torch.utils.data import DataLoader
import argparse
from util import DataManager, CNN_squeezenet, CNN_vgg16, EasyDataset
assert torch and DataLoader and CNN_squeezenet and CNN_vgg16 and EasyDataset


parser = argparse.ArgumentParser(description='DLCV Final')
#parser.add_argument('-p','--problem', dest='problem',type=int,required=True)
args = parser.parse_args()


dm= DataManager()

BATCH_SIZE = 32
INPUT_MODEL = './model/resnet50_recover.pt'
INPUT_CHARACTER = 'data/character.txt'
OUTPUT_PATH = './resnet_output.csv'

dm.character.load(INPUT_CHARACTER)
val_path=['./data/valx.npy','./data/valy.npy']
test_path='./data/test.npy'
val_data=dm.readfile('./dataset/val', './dataset/val_id.txt', save_path=val_path)
test_data=dm.readtestfile('./dataset/test/', save_path= test_path)

'''
model= resnet50()
model= dm.load(INPUT_MODEL, model).cuda()
'''
model= torch.load(INPUT_MODEL).cuda()
print('Model parameters: {}'.format(dm.count_parameters(model)))


val_dataloader= DataLoader(EasyDataset(*val_data),batch_size= BATCH_SIZE, shuffle= False, num_workers = 0)
test_dataloader= DataLoader(EasyDataset(test_data),batch_size= BATCH_SIZE, shuffle= False, num_workers = 8)

accu_record=0
dm.val_classifier( model, val_dataloader, 0)
test = dm.test_classifier( model, test_dataloader)

dm.write(test,OUTPUT_PATH)
