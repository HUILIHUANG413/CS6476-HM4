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
ratio=0.75 #threshold you choose to pick patch
def rawDescriptorMatches(fileName):
    mat = scipy.io.loadmat(fileName)
    #numfeats=mat['im1'].shape[0]
    #numdimention=mat[im1].shape[1]
    #print(mat)
    im1 = mat['im1']

    fig=plt.figure()
    #ax=fig.add_subplot(111)
    #ax.imshow(im1)
    # now show how to select a subset of the features using polygon drawing
    print('nuse the mouse to draw a polygon, right click to end it')
    plt.imshow(im1)
    MyROI = roipoly(roicolor='r')
    # Ind contains the indices of the SIFT features whose centers fall
    Ind = MyROI.getIdx(im1, mat['positions1'])
    descriptor_1 = mat['descriptors1'][Ind]
    print(descriptor_1.shape)
    descriptor_2=mat['descriptors2']


    distance_list=np.empty((0,2))
    for i in descriptor_1:
        sift_descriptor_vector = np.array(i, ndmin=2)
        distance=dist2(sift_descriptor_vector,descriptor_2)
        distance_list = np.vstack((distance_list, [np.amin(distance), np.argmin(distance)]))
    print("shape of distance_list",distance_list.shape)
    distance_mean=np.mean(distance_list[:,0],axis=0)
    distance_list_mask=distance_list[:,0]< (distance_mean*ratio)
    print("shape of distance_mask", distance_list_mask.shape)
    distance_list=distance_list[distance_list_mask,:]
    print("shape of after mask", distance_list.shape)
    distance_array=np.array(distance_list,dtype=np.uint)

    im2 = mat['im2']
    fig=plt.figure()
    bx=fig.add_subplot(111)
    #coners = displaySIFTPatches(mat['positions2'][min_distance], mat['scales2'][min_distance], mat['orients2'][min_distance])
    coners = displaySIFTPatches(mat['positions2'][distance_array[:,1],:], mat['scales2'][distance_array[:,1],:],
                                mat['orients2'][distance_array[:,1],:])
    #coners = displaySIFTPatches(mat['positions2'][tlist[:, 1], :], mat['scales2'][tlist[:, 1], :],
    #                            mat['orients2'][tlist[:, 1], :])

    for j in range(len(coners)):
        bx.plot([coners[j][0][1], coners[j][1][1]], [coners[j][0][0], coners[j][1][0]], color='g', linestyle='-', linewidth=1)
        bx.plot([coners[j][1][1], coners[j][2][1]], [coners[j][1][0], coners[j][2][0]], color='g', linestyle='-', linewidth=1)
        bx.plot([coners[j][2][1], coners[j][3][1]], [coners[j][2][0], coners[j][3][0]], color='g', linestyle='-', linewidth=1)
        bx.plot([coners[j][3][1], coners[j][0][1]], [coners[j][3][0], coners[j][0][0]], color='g', linestyle='-', linewidth=1)
    bx.set_xlim(0, im2.shape[1])
    bx.set_ylim(0, im2.shape[0])  
    plt.gca().invert_yaxis()
    bx.imshow(im2)
    plt.show()


def main():
    rawDescriptorMatches("twoFrameData.mat")

if __name__ == "__main__":
    main()