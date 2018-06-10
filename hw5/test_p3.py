

import torch
import argparse
from util import DataManager, MovieDataLoader

parser = argparse.ArgumentParser(description='DLCV HW5')
parser.add_argument('-v','--video_dir', dest='video_dir',required=True)
parser.add_argument('-t','--tag', dest='tag',required=True)
parser.add_argument('-o','--output', dest='output',required=True)
args = parser.parse_args()
dm= DataManager()

EPOCH =50
BATCH_SIZE =  8
TRAIN_FEATURE = 25088
HIDDEN_DIM = 1024
LAYER_N = 3
LABEL_DIM = 11
DROPOUT = 0.5
LEARNING_RATE = 1E-4
INPUT_PATH = './model/rnn_movie.pt'
TEST_DIR = './test/'


val_data, moviedir = dm.get_movie('./data/FullLengthVideos/videos/valid', './data/FullLengthVideos/labels/valid', save_path=None)
test_path = ['{}.txt'.format(i) for i in moviedir]

model= torch.load(INPUT_PATH)

val_dataloader= MovieDataLoader(val_data,None,batch_size= BATCH_SIZE, shuffle= False)

dm.write_movie(*dm.test_movie(model, val_dataloader, 0),TEST_DIR, test_path)
