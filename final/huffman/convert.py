from huffman import HuffmanCoding
import sys
import torch
import pickle
import numpy as np
assert sys and torch and np
'''
fetched = []
state_dict = torch.load('state_dict.pt')
keys = list(state_dict.keys())
layer_name = []

for i in range(len(keys)):
    key = keys[i]
    if key[9:13] == 'conv' or key[0:4] == 'conv' or key == 'fc.weight':
        print (key)
        layer_weight = state_dict[key].view(-1).numpy()
        print (layer_weight.shape)
        fetched.append(layer_weight)
        layer_name.append(key)
'''
'''
cal = np.zeros(shape=(1,256))
for i in range(len(fetched)):
    for j in range(len(fetched[i])):
        cal[0][int(fetched[i][j])] += 1
for i in range(256):
    print(str(i)+':'+str(cal[0][i]))
'''
'''
with open('para.txt','w') as f:
    for i in range(len(fetched)):
        for j in range(len(fetched[i])):
            f.write(str(fetched[i][j]))
            if j<len(fetched[i])-1 : f.write(' ')
        f.write('\n') 

'''
path = "para.txt"

h = HuffmanCoding(path)

output_path = h.compress()
print("Compressed file path: " + output_path)

with open('decoder.pkl', 'wb') as output:
    pickle.dump(h, output, pickle.HIGHEST_PROTOCOL)


with open('decoder.pkl', 'rb') as input:
    h = pickle.load(input)

output_path = 'para.bin'
decom_path = h.decompress(output_path)
print("Decompressed file path: " + decom_path)
