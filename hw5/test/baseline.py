
import numpy as np
import argparse
parser = argparse.ArgumentParser(description="Baseline")
parser.add_argument('-p','--predict', type=str)
parser.add_argument('-g','--ground', type=str, default='../dataset/valy.npy')
args = parser.parse_args()

ground= np.load(args.ground)
with open(args.predict, 'r') as f:
    pred=np.array([i.strip() for i in f.readlines()]).astype(np.uint8)
correct= len(ground)-np.count_nonzero(ground- pred)
print('Accuracy: {:.4f}%'.format(100. *correct/len(ground)))
