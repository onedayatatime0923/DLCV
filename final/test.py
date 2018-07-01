

import torch
from torch.utils.data import DataLoader
import argparse
from util import DataManager, CNN_squeezenet, CNN_vgg16, EasyDataset
from resnet import resnet50
assert torch and DataLoader and CNN_squeezenet and CNN_vgg16 and EasyDataset


parser = argparse.ArgumentParser(description='DLCV Final')
#parser.add_argument('-p','--problem', dest='problem',type=int,required=True)
args = parser.parse_args()


dm= DataManager()

EPOCH = 50
BATCH_SIZE = 32
LEARNING_RATE = 1E-4
DROPOUT = 0.5
PRETRAIN = False
INPUT_MODEL = './model/resnet_compress.pt'
INPUT_CHARACTER = 'data/character.txt'
OUTPUT_PATH = './output.csv'

dm.character.load(INPUT_CHARACTER)
test_path='./data/test.npy'
test_data=dm.readtestfile('./dataset/test/', save_path= test_path)

model= resnet50()
model= dm.load(INPUT_MODEL, model).cuda()
print('Model parameters: {}'.format(dm.count_parameters(model)))

optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)

test_dataloader= DataLoader(EasyDataset(test_data),batch_size= BATCH_SIZE, shuffle= False, num_workers = 8)

accu_record=0
test = dm.test_classifier( model, test_dataloader)

dm.write(test,OUTPUT_PATH)
