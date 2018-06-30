
import time, math
import numpy as np
import torch
from sklearn.cluster import KMeans
import pickle


class DataManager():
    def timeSince(self,since, percent):
        now = time.time()
        s = now - since
        es = s / (percent)
        rs = es - s
        return '%s (- %s)' % (self.asMinutes(s), self.asMinutes(rs))
    def asMinutes(self,s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)
    def save(self, model, path, cluster= 256):
        model = model.cpu().eval()
        state_dict = model.state_dict()
        weight = {}
        kmeans= KMeans(n_clusters=cluster, random_state=0, n_jobs=4)

        for key in state_dict:
            print(key)
            print(state_dict[key].numel())
            layer = key.split('.')
            #if state_dict[key].numel()<= cluster:
            if layer[0] == 'conv':
                weight[key] = state_dict[key].numpy()
            else:
                size = state_dict[key].size()
                params = state_dict[key].view(-1,1).numpy()
                kmeans.fit(params)
                quantized_table = kmeans.cluster_centers_.reshape((-1,)).astype(np.uint8)
                quantized_weight = kmeans.labels_.reshape(size).astype(np.uint8)
                weight[key] = (quantized_table, quantized_weight)

        with open(path, 'wb') as f:
                pickle.dump(weight, f, protocol=pickle.HIGHEST_PROTOCOL)
    def load(self, path, model):
        with open(path, 'rb') as f:
            weight= pickle.load(f)
        state_dict = {}
        for key in state_dict:
            if isinstance(state_dict[key], np.ndarray):
                state_dict[key] = torch.from_numpy(weight[key][0])
            else:
                quantized_table = state_dict[key][0]
                quantized_weight = state_dict[key][1]
                state_dict[key] = torch.from_numpy(quantized_weight[quantized_table])
        model.load_state_dict(state_dict)



if __name__ == '__main__':
    dm = DataManager()
    model = torch.load('./model/model.pt')
    print(model)
    dm.save(model, './')
