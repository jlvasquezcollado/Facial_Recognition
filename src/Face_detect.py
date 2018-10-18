from __future__ import division, print_function
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import scipy
from skimage import feature
from PIL import Image
from scipy.misc import imresize
import matplotlib.cm as cm
from scipy.interpolate import interp2d
from sklearn.decomposition import PCA
from sklearn import svm 
import argparse
import pattern_recog_func as prf
import matplotlib.gridspec as gridspec
import cv2
import copy


# Start of Part A
# Note MUST CHANGE PICTURE DIRECTORIES TO MATCH YOUR COMPUTERS 
paths = ["/Users/jorge/Desktop/Luke", "/Users/jorge/Desktop/Janet", "/Users/jorge/Desktop/Gilbert"]
names = ["Luke", "Janet", "Gilbert"]

X = prf.load_images(paths, names)
name_dic = {1:" Luke", 2:" Janet", 3:" Gilbert"}

# creating target
y = np.concatenate((np.ones(40), np.ones(37) *2, np.ones(40) * 3))

Errors = 0
for i in range(X.shape[0]):
    Xtest = X[i]
    ytest = y[i]
    Xtrain = np.delete(X, i, axis = 0)
    ytrain = np.delete(y, i, axis = 0)
    md_pca, Xproj = prf.pca_X(Xtrain, n_comp = 50)
    XtestProj = md_pca.transform(Xtest.reshape(1,-1))
    md_clf = prf.svm_train(Xproj, ytrain)
    predic = md_clf.predict(XtestProj)
    if(predic[0] != ytest):
         Errors +=1


print("success rate: ", (1 - (Errors/X.shape[0])) * 100, "%")

# START OF PART C
#NOTE MUST CHANGE PICTURE DIRECTORY TO MATCH YOUR INDIVIDUAL DIRECTORY
GuessPicture = "/Users/jorge/Desktop/whoswho.JPG"
faces = prf.get_faces(GuessPicture)
predictions = prf.pca_svm_pred(faces, md_pca, md_clf)
print("PCA+SVM predition for person 1: {}".format(name_dic[predictions[0]]))
print("PCA+SVM predition for person 2: {}".format(name_dic[predictions[1]]))
print("PCA+SVM predition for person 3: {}".format(name_dic[predictions[2]]))

plt.imshow(cv2.imread(GuessPicture))
plt.axis("off")
plt.show()


















