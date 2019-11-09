import matplotlib.path as mplPath
import numpy as np
import scipy.io
import glob
from dist2 import *
from scipy import misc
import matplotlib.pyplot as plt
from displaySIFTPatches import displaySIFTPatches
from selectRegion import roipoly
from getPatchFromSIFTParameters import getPatchFromSIFTParameters
from skimage.color import rgb2gray
import matplotlib.cm as cm
import pylab as pl
import pdb
from imageio import imread
import pickle
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq,kmeans2
import math

start = 0
end = 50
step=5
is_clustered=True  #For the first train, is_clustered=False
hist_clusered=True
k=1500
ID=85 #96 for savebelt (failure)  #70 for the wall #100 for window #56 for poster #131 for shirt #85 for dress



def reginQueries():
    #get kmeans
    with open('cluster_question3.pkl', "rb") as f:
        kmeans = pickle.load(f)
    #get histogram and histogram mapping
    with open("histogram.pkl", 'rb') as f:
        histogram = pickle.load(f)
    histogram = np.array(histogram)
    print(histogram.shape)
    with open("histogram_mapping.pkl", 'rb') as f:
        histogram_mappping = pickle.load(f)
    print(len(histogram_mappping))
    #get selected region
    framesdir = 'frames/'
    siftdir = 'sift/'
    histogram_choosed=[]
    hist_choosed=np.zeros(k)
    fnames = glob.glob(siftdir + '*.mat')
    fnames = [i[-27:] for i in fnames]
    fname = siftdir + fnames[ID]
    mat = scipy.io.loadmat(fname, verify_compressed_data_integrity=False)
    imname = framesdir + fnames[ID][:-4]
    im = imread(imname)
    pl.imshow(im)
    MyROI = roipoly(roicolor='r')
    Ind = MyROI.getIdx(im, mat['positions'])
    # get parameters
    describtors = mat['descriptors'][Ind, :]
    if len(describtors) > 0:
        predict_result = kmeans.predict(describtors)
        for label in predict_result:
            hist_choosed[label] += 1
        histogram_choosed.append(hist_choosed)
    histogram_choosed = np.array(histogram_choosed)
    return histogram,histogram_mappping,histogram_choosed


def comput_distance_hist_hist(hist1,hist2):
    numerator=0
    for i in range (k):
        numerator+=hist1[:,i]*hist2[i]
    ss1 = math.sqrt(np.sum(hist1 ** 2))
    ss2 = math.sqrt(np.sum(hist2 ** 2))
    denominator=ss1*ss2
    return numerator/denominator

def compute_distance_choosehist_allhist(hist_choosed,histogram):
    value=[]
    i=0
    for hist in histogram:
        #print("now computing ：",i)
        value.append(comput_distance_hist_hist(hist_choosed,hist))
        i+=1
    #value=np.array(value)
    return value

def find_img(ID,histogram,histogram_mapping,hist_choosed):
    res=[]
    hist_we_choose=hist_choosed
    #print("compute the distance between ", ID, "and others")
    distance=compute_distance_choosehist_allhist(hist_we_choose,histogram)
    for i in range (7):
        best=np.argmax(distance)
        distance[best]=0
        if i>0:
            res.append(best)
            print(" the ", i , " bset image is ", histogram_mapping[best])
    plt.figure()
    tmp=1
    for i in res:
        im=imread(histogram_mapping[i])
        ax=plt.subplot(2,3,tmp)
        ax.set_title("Best image")
        ax.imshow(im)
        tmp+=1
    plt.show()

if __name__ == "__main__":
    #ID=85 #70 #55
    print("compute histogram")
    histogram,histogram_img,histogram_choosed=reginQueries()
    print("histogram_choosed.shape=",histogram_choosed.shape)
    print("histogram.shape=", histogram.shape)
    find_img(ID,histogram,histogram_img,histogram_choosed)









