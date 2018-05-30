
from util import DataManager, ResNet50_feature


EPOCH =200
BATCH_SIZE = 2
HIDDEN_SIZE = 4096
LABEL_DIM = 11
TENSORBOARD_DIR= './runs/train'

dm= DataManager(TENSORBOARD_DIR)
model= ResNet50_feature(HIDDEN_SIZE, LABEL_DIM).cuda()
train_dataloader= dm.get_data('./data/TrimmedVideos/video/train', './data/TrimmedVideos/label/gt_train.csv', save_path=['./data/trainx.npy','./data/trainy.npy'], batch_size= BATCH_SIZE, shuffle= True)
val_dataloader= dm.get_data('./data/TrimmedVideos/video/valid', './data/TrimmedVideos/label/gt_valid.csv', save_path=['./data/valx.npy','./data/valy.npy'], batch_size= BATCH_SIZE, shuffle= True)

for epoch in range(1,EPOCH+1):
    dm.train( model, train_dataloader, epoch)
    dm.val( model, val_dataloader, epoch)
    print('-'*80)
