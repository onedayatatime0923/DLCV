
#from PIL import Image
from skimage import  color
import numpy as np
from sklearn.cluster import KMeans
from scipy.signal   import correlate2d
from scipy.misc     import imread
from scipy import io 
import matplotlib.pyplot as plt
assert KMeans and np and color and correlate2d and io and imread

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
    def kmeans(self,name,data,n=10,max_iter=1000):
        self.data[name]= KMeans(n_clusters=n, random_state=0,max_iter=max_iter).fit(data)
    def plot(self,data,title,path):
        data=data.astype(np.uint8)
        plt.title(title)
        plt.imshow(data)
        #plt.show()
        plt.savefig(path)
