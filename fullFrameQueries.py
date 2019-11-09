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



def fullFrameQuries():
    # specific frame dir and siftdir
    framesdir = 'frames/'
    siftdir = 'sift/'
    # Get a list of all the .mat file in that directory.
    # there is one .mat file per image.
    fnames = glob.glob(siftdir + '*.mat')
    fnames = [i[-27:] for i in fnames]
    fnames = fnames[0::step]
    fnames = fnames[start:end]
    #fnames=fnames[start:end]

    print ('reading %d total files...' %(len(fnames)))
    N = 100 #to visualize a sparser set of the features
    all_describtor = []
    mapping={}
    #all_position = []
    #all_orient = []
    #all_scale = []
    descriptors_count =0
    for i in range(len(fnames)):
        print('reading frame %d of %d' % (i, len(fnames)))
        # load that file
        fname = siftdir + fnames[i]
        print(fname)
        mat = scipy.io.loadmat(fname,verify_compressed_data_integrity=False )
        #get parameters
        describtors=mat['descriptors']
        for j,des in enumerate(describtors):
            all_describtor.append(des)
            mapping[descriptors_count]=(fnames[i],j)
            descriptors_count+=1

    all_describtor=np.array(all_describtor)


    #k-means
    k=1500
    #whiten(all_describtor)
    #center,label=kmeans2(all_describtor,k)

    #save the dataset
    if not is_clustered:
        print("Start Kmeans")
        kmeans=KMeans(n_clusters=k).fit(all_describtor) #Compute cluster centers and predict cluster index for each sample.
    # to save it
        with open('cluster_question3.pkl',"wb") as f:
            pickle.dump(kmeans, f)
    else:
        #with open('kmeans.pkl',"rb") as f:
        #    kmeans=pickle.load(f)
        with open('cluster_question3.pkl',"rb") as f:
            kmeans=pickle.load(f)


    #compute histogram for every image .
    fnames = glob.glob(siftdir + '*.mat')
    fnames = [i[-27:] for i in fnames]

    #print('reading %d total files...' % (len(fnames)))
    histogram=[]
    histogram_mappping=[]
    if not hist_clusered:
        for i in range(len(fnames)):
            print('reading frame %d of %d' % (i, len(fnames)))
            # load that file
            hist= np.zeros(k)
            fname = siftdir + fnames[i]
            mat = scipy.io.loadmat(fname, verify_compressed_data_integrity=False)
            imname = framesdir + fnames[i][:-4]
            # get parameters
            describtors = mat['descriptors']
            if len(describtors)>0:
                predict_result=kmeans.predict(describtors)
                for label in predict_result:
                    hist[label]+=1
                histogram.append(hist)
                histogram_mappping.append(imname)

        with open("histogram.pkl", 'wb') as f:
            pickle.dump(histogram, f)
        with open("histogram_mapping.pkl", 'wb') as f:
            pickle.dump(histogram_mappping, f)

    else:
        with open("histogram.pkl", 'rb') as f:
            histogram=pickle.load(f)
        histogram=np.array(histogram)
        print(histogram.shape)
        with open("histogram_mapping.pkl", 'rb') as f:
            histogram_mappping=pickle.load(f)
        print(len(histogram_mappping))
        '''
        for des in range (describtors.shape[0]): #for every des in descriptors
            for label in range (center_number): #find the closet label
                distance=np.linalg.norm(describtors[des]-cluster_center[label])
                if distance<min_distance:
                    min_distance=distance
                    min_distance_label=label
                histogram[i,min_distance_label]+=1
        '''
    return histogram, histogram_mappping
def comput_distance_hist_hist(hist1,hist2):
    numerator=0
    for i in range (k):
        numerator+=hist1[i]*hist2[i]
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

def find_img(ID,histogram,histogram_mapping):
    res=[]
    hist_we_choose=histogram[ID]
    mapping_we_choose=histogram_mapping[ID]
    #plt.figure()
    #plt.title("orginal image ")
    #plt.savefig("img_choose_for_Q3_1.png",mapping_we_choose)
    print("compute the distance between ", ID, "and others")
    distance=compute_distance_choosehist_allhist(hist_we_choose,histogram)
    print(" the original image is ", histogram_mapping[ID])
    for i in range (6):
        best=np.argmax(distance)
        distance[best]=0
        if i>0:
            res.append(best)
            print(" the ", i , " bset image is ", histogram_mapping[best])
    plt.figure()
    im=imread(mapping_we_choose)
    ax=plt.subplot(2,3,1)
    ax.set_title("original")
    ax.imshow(im)
    tmp=2
    for i in res:
        im=imread(histogram_mapping[i])
        ax=plt.subplot(2,3,tmp)
        ax.set_title("Best image")
        ax.imshow(im)
        tmp+=1
    plt.show()


if __name__ == "__main__":
    ID=70 #85 #70 #55
    print("compute histogram")
    histogram,histogram_img=fullFrameQuries()
    print("Find img")
    find_img(ID,histogram,histogram_img)










