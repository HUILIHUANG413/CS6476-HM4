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
import math

start = 0
end = 40
is_clustered=True  #For the first train, is_clustered=False
cluster_to_choose=102  #61 #or 131
k=1500 #cluster number

def compute_distance(hist1,hist2):
    ########
    dis1=math.sqrt(np.sum(hist1 ** 2))
    dis2=math.sqrt(np.sum(hist2 ** 2))
    sum=0
    for i in range (k):
        sum+=hist1[i]*hist2[i]
    return sum/(dis1*dis2)


def visualizeVocabulary():

    # specific frame dir and siftdir
    framesdir = 'frames/'
    siftdir = 'sift/'
    # Get a list of all the .mat file in that directory.
    # there is one .mat file per image.
    fnames = glob.glob(siftdir + '*.mat')
    fnames = [i[-27:] for i in fnames]
    fnames=fnames[start:end]

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
        mat = scipy.io.loadmat(fname)
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
        cluster=KMeans(n_clusters=k).fit_predict(all_describtor) #Compute cluster centers and predict cluster index for each sample.
    # to save it
        with open('cluster.pkl', 'wb') as f:
            pickle.dump(cluster, f)
        print('Clustering finished')
    else:
        with open('cluster.pkl', 'rb') as f:
            cluster = pickle.load(f)


    plt.figure()
    n=1
    for i in range (len(cluster)):
        if cluster[i] == cluster_to_choose:
            fname,index = mapping[i]
            mat=scipy.io.loadmat(siftdir+fname)
            imname = framesdir + fname[:-4]
            im = imread(imname)
            print(index)
            #index=np.int(index)
            a=mat['positions'][index,:]
            b=mat['scales'][index]
            c=mat['orients'][index]
            img = getPatchFromSIFTParameters(a,b,c, rgb2gray(im))
            #img = getPatchFromSIFTParameters(mat['positions'][index, :], mat['scales'][index],
            #                                       mat['orients'][index], rgb2gray(im))
            pl.subplot(5,5,n)
            pl.imshow(img,cmap='gray')
            n+=1
            if n>25:
                break


    plt.show()

if __name__ == "__main__":
    visualizeVocabulary()










