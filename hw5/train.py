
from util import DataManager, ResNet50_feature


EPOCH = 32
BATCH_SIZE = 32
HIDDEN_SIZE = 2048
LABEL_DIM = 11
TENSORBOARD_DIR= './runs/train'

dm= DataManager(TENSORBOARD_DIR)
model= ResNet50_feature(HIDDEN_SIZE, LABEL_DIM).cuda()
dataloader= dm.get_data('./data/TrimmedVideos/video/train', './data/TrimmedVideos/label/gt_train.csv', save_path=['./data/trainx.npy','./data/trainy.npy'], batch_size= BATCH_SIZE, shuffle= True)

for epoch in range(EPOCH):
    dm.train( model, dataloader, epoch)
