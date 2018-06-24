
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
from util import DataManager, Encoder, Classifier, ClassifierDataset
assert torch and DataLoader and Classifier and ClassifierDataset and np and Encoder


parser = argparse.ArgumentParser(description='DLCV HW5')
#parser.add_argument('-p','--problem', dest='problem',type=int,required=True)
args = parser.parse_args()

TENSORBOARD_DIR= './runs/train'

dm= DataManager(tensorboard_dir= TENSORBOARD_DIR)

EPOCH = 500
BATCH_SIZE = 1024
LABEL_DIM = 11
DROPOUT = 0.5
LEARNING_RATE = 1E-3
PRETRAIN_PATH = './model/pretrained.pt'
OUTPUT_CHARACTER = 'data/character.txt'

train_path=['./data/trainx.npy','./data/trainy.npy']
val_path=['./data/valx.npy','./data/valy.npy']
train_feature_path = './data/train_feature.npy'
val_feature_path = './data/val_feature.npy'

'''
val_data=dm.readfile('./dataset/val', './dataset/val_id.txt', save_path=val_path)
train_data=dm.readfile('./dataset/train/', './dataset/train_id.txt', save_path=train_path)

model= Encoder(PRETRAIN_PATH).cuda()

train_feature= dm.dimension_reduction_model(model, train_data[0], train_feature_path)
val_feature= dm.dimension_reduction_model(model, val_data[0], val_feature_path)

train_x, train_y = train_feature, train_data[1]
val_x, val_y = val_feature, val_data[1]
del train_data
del val_data
'''
train_x, train_y = np.load('./data/train_feature.npy'), np.load('./data/trainy.npy')
val_x, val_y = np.load('./data/val_feature.npy'), np.load('./data/valy.npy')

print(train_x.shape)
print(train_y.shape)
print(val_x.shape)
print(val_y.shape)
################################################################
#                      nn                                      #
################################################################
'''
train_dataloader = DataLoader(ClassifierDataset(train_x, train_y), batch_size = BATCH_SIZE, shuffle = True)
val_dataloader = DataLoader(ClassifierDataset(val_x, val_y), batch_size = BATCH_SIZE, shuffle = False)


model = Classifier(dropout= 0.5).cuda()

optimizer = torch.optim.Adam(model.fc.parameters(),lr=LEARNING_RATE)


accu_record=0
for epoch in range(1,EPOCH+1):
    dm.train_classifier( model, train_dataloader, epoch, optimizer)
    record=dm.val_classifier( model, val_dataloader, epoch)
    if record[1]> accu_record:
        #model.save(OUTPUT_PATH)
        accu_record= record[1]
    print('='*80)
'''
################################################################
#                      naive bayes                             #
################################################################
'''
nb = dm.naive_bayes_construct(train_x, train_y)
pred= dm.naive_bayes_predict(nb, val_x)

'''
################################################################
#                      knn                                     #
################################################################


knn = dm.knn_construct(train_x, train_y)
pred = dm.knn_predict(knn, val_x)
print(pred)
print(val_y)

print("Accu: {}".format(1-(np.count_nonzero(pred-val_y)/ len(val_y))))
