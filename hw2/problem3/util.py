
#from PIL import Image
from skimage import  color
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.signal   import correlate2d
from scipy.misc     import imread
from scipy import io 
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import cv2
assert KMeans and np and color and correlate2d and io and imread and OneHotEncoder and KNeighborsClassifier

class Datamanager:
    def __init__(self):
        self.data={}
    def read_image(self,name,path,mode):
        if mode=='rgb':
            self.data[name]=imread( path )
        elif mode=='lab':
            rgb = imread( path )
            self.data[name]= color.rgb2lab(rgb)
        elif mode=='L':
            self.data[name]= imread( path ,mode='L')
        else: raise ValueError('Wrong mode.')
    def read_mat(self,name,path):
        res=io.loadmat(path)
        self.data[name]=res['F']
    def correlate(self,x,y):
        return correlate2d(x,y,mode='same',boundary='symm')
    def pca_construct(self,data,n):
        pca = PCA(n_components=n)
        pca.fit(data)
        self.pca=pca
    def pca_transform(self,data):
        return self.pca.transform(data)
    def kmeans_construct(self,data,n=50,max_iter=5000):
        self.kmeans= KMeans(n_clusters=n, random_state=0,max_iter=max_iter).fit(data)
    def KNN_construct(self,x,y,n):
        neigh = KNeighborsClassifier(n_neighbors=n)
        neigh.fit(x, y) 
        self.knn=neigh
    def KNN_predict(self,x):
        return self.knn.predict(x)
    def surf_detect(self,data):
        surf = cv2.xfeatures2d.SURF_create()
        self.surf= surf
        kp, des = surf.detectAndCompute(data,None)
        return des
    def surf_plot(self,image,title,path):
        surf=self.surf
        kp= surf.detect(image,None)[:30]
        img2 = cv2.drawKeypoints(image,kp,None,(255,0,0),4)
        self.plot_image(img2,title,path)
    def plot_image(self,data,title,path):
        data=data.astype(np.uint8)
        plt.figure()
        plt.imshow(data)
        plt.title(title)
        #plt.show()
        plt.savefig(path)
    def plot_bar(self,data,path):
        plt.figure(figsize=(100,60))
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        for i in range(len(data)):
            plt.subplot(5,3,i+1)
            x=np.arange(data[i][3].shape[0])
            plt.bar(x, data[i][3], facecolor='#9999ff', edgecolor='white')
            plt.title('Class_{}_{}_{}'.format(data[i][0],data[i][1],data[i][2]),fontdict={'fontsize':50})
        #plt.show()
        plt.savefig(path)
    def embedding(self,points,mode):
        assign_=self.kmeans_assign(points,mode=mode[0])
        #print(assign_.shape)
        emb=self.pooling(assign_,mode=mode[1])
        #print(emb.shape)
        return emb
    def kmeans_assign(self,data,mode):
        if mode=='hard':
            c=self.kmeans.predict(data).reshape((-1,1))
            enc = OneHotEncoder(50,sparse=False).fit(c)
            res=enc.transform(c)
            return res
        elif mode=='soft':
            c=self.kmeans.transform(data)
            return c
        else: raise ValueError('Wrong mode.')
    def pooling(self,data,mode):
        if mode=='sum':
            res=np.sum(data,0)
            res/=np.linalg.norm(res)
            return res
        elif mode=='max':
            return np.max(data,0)
        else: raise ValueError('Wrong mode.')
