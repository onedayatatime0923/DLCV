
from util import DataManager, ResNet50_feature, Vgg16_feature_rnn
assert ResNet50_feature


EPOCH =200
BATCH_SIZE = 2
HIDDEN_SIZE = 1024
LAYER_N = 3
LABEL_DIM = 11
DROPOUT = 0.2
LEARNING_RATE = 1E-5
OUTPUT_PATH = './model/vgg16_feature_rnn.pt'
TENSORBOARD_DIR= './runs/train'


dm= DataManager(TENSORBOARD_DIR)
model= Vgg16_feature_rnn(HIDDEN_SIZE, LAYER_N, LABEL_DIM, DROPOUT).cuda()
model.save(OUTPUT_PATH)
train_dataloader= dm.get_data('./data/TrimmedVideos/video/train', './data/TrimmedVideos/label/gt_train.csv', save_path=['./data/trainx.npy','./data/trainy.npy'], batch_size= BATCH_SIZE, shuffle= True)
val_dataloader= dm.get_data('./data/TrimmedVideos/video/valid', './data/TrimmedVideos/label/gt_valid.csv', save_path=['./data/valx.npy','./data/valy.npy'], batch_size= BATCH_SIZE, shuffle= True)

for epoch in range(1,EPOCH+1):
    dm.train( model, train_dataloader, epoch, LEARNING_RATE)
    dm.val( model, val_dataloader, epoch)
    print('-'*80)
    model.save(OUTPUT_PATH)
