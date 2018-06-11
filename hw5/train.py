
from torch.utils.data import DataLoader
import argparse
from util import DataManager, ImageDataset, ImageDataLoader, MovieDataLoader,  Classifier, Rnn_Classifier, Rnn_Classifier_Movie

parser = argparse.ArgumentParser(description='DLCV HW5')
parser.add_argument('-p','--problem', dest='problem',type=int,required=True)
args = parser.parse_args()
TENSORBOARD_DIR= './runs/train_problem{}'.format(args.problem)
dm= DataManager(TENSORBOARD_DIR)

################################################################
#                      problem 1                               #
################################################################
if args.problem==1:
    EPOCH = 50
    BATCH_SIZE = 128
    TRAIN_FEATURE = 25088
    HIDDEN_DIM = 1024
    LABEL_DIM = 11
    DROPOUT = 0.5
    LEARNING_RATE = 1E-4
    OUTPUT_PATH = './model/classifier.pt'
    TEST_PATH = './test/problem1.txt'

    train_path=['./dataset/trainx.npy','./dataset/trainy.npy']
    val_path=['./dataset/valx.npy','./dataset/valy.npy']
    train_data=dm.get_data('./data/TrimmedVideos/video/train', './data/TrimmedVideos/label/gt_train.csv', save_path=train_path)
    val_data=dm.get_data('./data/TrimmedVideos/video/valid', './data/TrimmedVideos/label/gt_valid.csv', save_path=val_path)
    model= Classifier(TRAIN_FEATURE, HIDDEN_DIM, LABEL_DIM, DROPOUT).cuda()
    model.save(OUTPUT_PATH)

    train_dataloader= DataLoader(ImageDataset(*train_data),batch_size= BATCH_SIZE, shuffle= True)
    val_dataloader= DataLoader(ImageDataset(*val_data),batch_size= BATCH_SIZE, shuffle= False)

    accu_record=0
    for epoch in range(1,EPOCH+1):
        dm.train_classifier( model, train_dataloader, epoch, LEARNING_RATE)
        record=dm.val_classifier( model, val_dataloader, epoch)
        if record[1]> accu_record:
            model.save(OUTPUT_PATH)
            accu_record= record[1]
            dm.write(dm.test_classifier(model, val_dataloader, epoch),TEST_PATH)
        print('-'*80)
################################################################
#                      problem 2                               #
################################################################
elif args.problem==2:
    EPOCH =50
    BATCH_SIZE = 32
    TRAIN_FEATURE = 25088
    HIDDEN_DIM = 1024
    LAYER_N = 3
    LABEL_DIM = 11
    DROPOUT = 0.5
    LEARNING_RATE = 1E-4
    OUTPUT_PATH = './model/rnn_classifier.pt'
    TEST_PATH = './test/problem2.txt'

    train_path=['./dataset/trainx.npy','./dataset/trainy.npy']
    val_path=['./dataset/valx.npy','./dataset/valy.npy']
    train_data=dm.get_data('./data/TrimmedVideos/video/train', './data/TrimmedVideos/label/gt_train.csv', save_path=train_path)
    val_data =dm.get_data('./data/TrimmedVideos/video/valid', './data/TrimmedVideos/label/gt_valid.csv', save_path=val_path)
    model= Rnn_Classifier(TRAIN_FEATURE, HIDDEN_DIM, LAYER_N, LABEL_DIM,  DROPOUT ).cuda()
    model.save(OUTPUT_PATH)

    train_dataloader= ImageDataLoader(train_data[0], train_data[1],batch_size= BATCH_SIZE, shuffle= True)
    val_dataloader= ImageDataLoader(val_data[0],val_data[1],batch_size= BATCH_SIZE, shuffle= False)

    accu_record=0
    for epoch in range(1,EPOCH+1):
        dm.train_rnn( model, train_dataloader, epoch, LEARNING_RATE)
        record=dm.val_rnn( model, val_dataloader, epoch)
        if record[1]> accu_record:
            model.save(OUTPUT_PATH)
            accu_record= record[1]
            dm.write(dm.test_rnn(model, val_dataloader, epoch),TEST_PATH)
        print('-'*80)
################################################################
#                      problem 3                               #
################################################################
elif args.problem==3:
    EPOCH =50
    BATCH_SIZE =  8
    TRAIN_FEATURE = 25088
    HIDDEN_DIM = 1024
    LAYER_N = 3
    LABEL_DIM = 11
    DROPOUT = 0.5
    LEARNING_RATE = 1E-4
    OUTPUT_PATH = './model/rnn_movie.pt'
    TEST_DIR = './test/'


    train_path=['./dataset/movie_trainx.npy','./dataset/movie_trainy.npy']
    val_path=['./dataset/movie_valx.npy','./dataset/movie_valy.npy']
    train_data, _= dm.get_movie('./data/FullLengthVideos/videos/train', './data/FullLengthVideos/labels/train', save_path=train_path, cut= 350)
    val_data, test_path= dm.get_movie('./data/FullLengthVideos/videos/valid', './data/FullLengthVideos/labels/valid', save_path=val_path)
    model= Rnn_Classifier_Movie(TRAIN_FEATURE, HIDDEN_DIM, LAYER_N, LABEL_DIM,  DROPOUT ).cuda()
    model.save(OUTPUT_PATH)

    train_dataloader= MovieDataLoader(train_data[0], train_data[1],batch_size= BATCH_SIZE, shuffle= True)
    val_dataloader= MovieDataLoader(val_data[0],val_data[1],batch_size= BATCH_SIZE, shuffle= False)

    accu_record=0
    for epoch in range(1,EPOCH+1):
        dm.train_movie( model, train_dataloader, epoch, LEARNING_RATE)
        record=dm.val_movie( model, val_dataloader, epoch)
        if record[1]> accu_record:
            model.save(OUTPUT_PATH)
            accu_record= record[1]
            dm.write_movie(*dm.test_movie(model, val_dataloader, epoch),TEST_DIR, test_path)
        print('-'*80)
