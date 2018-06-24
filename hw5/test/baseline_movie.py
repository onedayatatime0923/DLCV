
import numpy as np
import argparse
parser = argparse.ArgumentParser(description="Baseline")
parser.add_argument('-p','--predict', type=str, default='./')
parser.add_argument('-g','--ground', type=str, default='../data/FullLengthVideos/labels/valid')
args = parser.parse_args()

TEST_FILE = ['OP05-R07-Pizza', 'OP06-R05-Cheeseburger', 'OP01-R03-BaconAndEggs', 'OP02-R04-ContinentalBreakfast', 'OP03-R02-TurkeySandwich']
correct=0
count=0
for test_f in TEST_FILE:
    with open('{}/{}.txt'.format(args.predict,test_f), 'r') as f:
        pred=np.array([i.strip() for i in f.readlines()]).astype(np.uint8)
    with open('{}/{}.txt'.format(args.ground,test_f), 'r') as f:
        ground=np.array([i.strip() for i in f.readlines()]).astype(np.uint8)
    correct+= len(ground)-np.count_nonzero(ground- pred)
    count+= len(ground)
print('Accuracy: {:.4f}%'.format(100. *correct/count))
