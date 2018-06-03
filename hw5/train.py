
from torch.utils.data import DataLoader
import argparse
from util import DataManager, ImageDataset, ImageDataLoader, Classifier, Rnn_Classifier

parser = argparse.ArgumentParser(description='DLCV HW5')
parser.add_argument('-p','--problem', dest='problem',type=int,required=True)
args = parser.parse_args()
TENSORBOARD_DIR= './runs/train'
dm= DataManager(TENSORBOARD_DIR)

################################################################
#                      problem 1                               #
################################################################
if args.problem==1:
    EPOCH =200
    BATCH_SIZE = 512
    TRAIN_FEATURE = 35840
    HIDDEN_DIM = 2048
    LABEL_DIM = 11
    DROPOUT = 0.2
    LEARNING_RATE = 1E-5
    OUTPUT_PATH = './model/classifier.pt'

    train_path=['./dataset/trainx.npy','./dataset/trainy.npy']
    val_path=['./dataset/valx.npy','./dataset/valy.npy']
    train_feature_dim= dm.get_data('./data/TrimmedVideos/video/train', './data/TrimmedVideos/label/gt_train.csv', save_path=train_path, batch_size= BATCH_SIZE, shuffle= True)
    val_feature_dim= dm.get_data('./data/TrimmedVideos/video/valid', './data/TrimmedVideos/label/gt_valid.csv', save_path=val_path, batch_size= BATCH_SIZE, shuffle= True)
    assert train_feature_dim == val_feature_dim
    model= Classifier(TRAIN_FEATURE, HIDDEN_DIM, LABEL_DIM, DROPOUT).cuda()
    model.save(OUTPUT_PATH)

    train_dataloader= DataLoader(ImageDataset(*train_path),batch_size= BATCH_SIZE, shuffle= True)
    val_dataloader= DataLoader(ImageDataset(*val_path),batch_size= BATCH_SIZE, shuffle= True)

    for epoch in range(1,EPOCH+1):
        dm.train_classifier( model, train_dataloader, epoch, LEARNING_RATE)
        dm.val_classifier( model, val_dataloader, epoch)
        print('-'*80)
    model.save(OUTPUT_PATH)
################################################################
#                      problem 2                               #
################################################################
elif args.problem==2:
    EPOCH =200
    BATCH_SIZE = 1
    TRAIN_FEATURE = 35840
    HIDDEN_DIM = 2048
    LAYER_N = 3
    LABEL_DIM = 11
    DROPOUT = 0.2
    LEARNING_RATE = 1E-5
    INPUT_PATH = './model/classifier.pt'
    OUTPUT_PATH = './model/rnn_classifier.pt'

    train_path=['./dataset/trainx.npy','./dataset/trainy.npy']
    val_path=['./dataset/valx.npy','./dataset/valy.npy']
    #train_feature_dim= dm.get_data('./data/TrimmedVideos/video/train', './data/TrimmedVideos/label/gt_train.csv', save_path=train_path, batch_size= BATCH_SIZE, shuffle= True)
    #val_feature_dim= dm.get_data('./data/TrimmedVideos/video/valid', './data/TrimmedVideos/label/gt_valid.csv', save_path=val_path, batch_size= BATCH_SIZE, shuffle= True)
    #assert train_feature_dim == val_feature_dim
    model= Rnn_Classifier(HIDDEN_DIM, LABEL_DIM, DROPOUT, INPUT_PATH).cuda()
    model.save(OUTPUT_PATH)

    train_dataloader= ImageDataLoader(train_path[0], train_path[1],batch_size= BATCH_SIZE, shuffle= True)
    val_dataloader= ImageDataLoader(val_path[0],val_path[1],batch_size= BATCH_SIZE, shuffle= True)

    for epoch in range(1,EPOCH+1):
        dm.train_rnn( model, train_dataloader, epoch, LEARNING_RATE)
        dm.val_rnn( model, val_dataloader, epoch)
        print('-'*80)
    model.save(OUTPUT_PATH)
################################################################
#                      problem 2                               #
################################################################
LAYER_N = 3
