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
import cv2


def interpol_im(im, dim1 = 8, dim2 = 8, plot_new_im = False, cmap = 'binary', axis_off = False):
	# get image as array
    if(plot_new_im == True):
        fig = plt.figure()
    img = im[:,:,0]
   
    # no need to crop image, no bar is expected 
    # uses interpolation now!!! Can also use imresize(img, 16*16) instead too
    x = np.arange(img.shape[1])
    y = np.arange(img.shape[0])
    f = interp2d(x,y,img)
    newx = np.linspace(0, im.shape[1], dim1)
    newy = np.linspace(0, im.shape[0], dim2)
    newImg = f(newx, newy)
    let_im = newImg
    let_im_flat = let_im.flatten()
    #if plot_let is true, only then will the letter be shown, NOT SURE WHY THIS IF STATEMENT NOT ENTIRELY WORKING
    if(plot_new_im != False):
        plt.imshow(let_im, cmap = cmap)
        plt.grid(axis_off)
        plt.axis('off')
#plt.show() un comment to see digit image after interpolation.
    

    #return 1,256 (16*16 array flattened) array.
    return let_im_flat

def pca_svm_pred(imfile, md_pca, md_clf, dim1 = 45, dim2 = 60):
    images = imfile
    images_interp = [interpol_im(img, dim1= dim1, dim2 = dim2) for img in images]
    img_interp_proj = md_pca.transform(images_interp)
    predic = md_clf.predict(img_interp_proj)
    return predic

def pca_X(X, n_comp = 10):
	md_pca = PCA(n_comp, whiten = True)
	md_pca.fit(X)
	X_proj = md_pca.transform(X)
	return md_pca, X_proj

def rescale_pixel(X, unseen, ind = 0):
    test = X[1]
    unseen_img = interpol_im(unseen, dim1 = 8, dim2 = 8, plot_new_im = True)
    unseen_img = np.array((unseen_img * 15), dtype = int)
    unseen_img = unseen_img.reshape(8,8)

    for i in range(8):
        for j in range(8):
            if unseen_img[i,j] == 0:
                unseen_img[i,j] = 15
            else:
                unseen_img[i,j] = 0
    return unseen_img.flatten()

def  svm_train(X, y, gamma = 0.001, C = 100):
     clf = svm.SVC(gamma= gamma, C = C)
     clf.fit(X, y)
     return clf

def load_images(paths, names):
    images = []
    cascPath = 'haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascPath)
    for j in range(len(paths)):
        path = paths[j]
        for i in range(40):
            file = (path +"/" + names[j] + "_" + "{}.png").format(i)
            img = cv2.imread(file)
            if(type(img) != None):
                faces = faceCascade.detectMultiScale(
                                     img,
                                     scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(30, 30),
                                     )
                for x,y,w,h in faces:
                    img = img[y:y+h, x:x+w]
                    images.append(img)

    interp_Images = [interpol_im(img, dim1 = 45, dim2 = 60, plot_new_im = False) for img in images]
    X = np.vstack(interp_Images)

    return X

def get_faces(path):
    cascPath = 'haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascPath)
    img = cv2.imread(path)
    faces = faceCascade.detectMultiScale(
                                     img,
                                     scaleFactor=1.2,
                                     minNeighbors=5,
                                     minSize=(300, 300),
                                     )
    cropped_faces = [img[y:y+h, x:x+w] for x,y,w,h in faces] 
    return cropped_faces 











