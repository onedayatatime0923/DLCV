
#from PIL import Image
from skimage import io, color
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
assert KMeans and np and color

class Datamanager:
    def __init__(self):
        self.data={}
    def read_image(self,name,path,mode):
        rgb = io.imread( path )
        if mode=='rgb':
            self.image=rgb
        elif mode=='lab':
            self.image= color.rgb2lab(rgb)
        else: raise ValueError('Wrong mode.')

    def kmeans_cluster(self,data,n=10,max_iter=1000):
        self.kmeans = KMeans(n_clusters=n, random_state=0,max_iter=max_iter).fit(data)
    def plot(self,data,path):
        plt.imshow(data)
        plt.savefig(path)
        #plt.show()

