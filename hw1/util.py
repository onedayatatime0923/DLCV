
from skimage import io
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
assert io and np

class   DataManager:
    def __init__(self):
        self.data={}
        self.pca=0
    def load_image(self,path,name):
        res=[]
        for i in range(1,41):
            for j in range(1,11):
                res.append(io.imread('{}/{}_{}.png'.format(path,str(i),str(j))).flatten())
        res=np.array(res)
        self.data[name]=res
        train=[]
        test=[]
        for j in range(len(res)):
            if 0<=(j%10)<=5:
                train.append(res[j])
            else:
                test.append(res[j])
        return np.array(train),np.array(test)
    def pca_construct(self,data,n):
        pca = PCA(n_components=n)
        pca.fit(data)
        self.pca=pca
    def pca_transform(self,data):
        return self.pca.transform(data)
    def pca_reconstruct(self,data):
        return self.pca.inverse_transform(data)
    def KNN(self,k,n,mode):
        correct=0
        if mode=='val':
            correct+=self.KNN_nfold(k,n,0,1,0,5)
            correct+=self.KNN_nfold(k,n,2,3,0,5)
            correct+=self.KNN_nfold(k,n,4,5,0,5)
            return correct/240
        elif mode=='test':
            correct+=self.KNN_nfold(k,n,6,9,0,9)
            return correct/160
        else: return ValueError("Wrong mode.")
    def KNN_nfold(self,k,n,val_start,val_end,train_start,train_end):
        train_x,train_y=[],[]
        val_x,val_y=[],[]
        for j in range(len(self.data['image'])):
            if val_start<=j%10<=val_end:
                val_x.append(self.data['image'][j])
                val_y.append(j//10+1)
            elif train_start<=j%10<=train_end:
                train_x.append(self.data['image'][j])
                train_y.append(j//10+1)
        train_x,train_y=np.array(train_x),np.array(train_y)
        val_x,val_y=np.array(val_x),np.array(val_y)
        pca = PCA(n_components=n)
        pca.fit(train_x)
        train_x_tran=pca.transform(train_x)
        val_x_tran=pca.transform(val_x)
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(train_x_tran, train_y) 
        correct=len(val_y)-np.count_nonzero((neigh.predict(val_x_tran))-val_y)
        return correct
    def plot(self,data,output_file,title=None):
        x=(data).reshape(56,46)
        plt.imshow(x,cmap='gray')
        if title != None:
            plt.title(title)
        plt.savefig(output_file)
