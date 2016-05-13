import sys
import os
import numpy as np
from subspace import pca, project, reconstruct
from util import normalize , asRowMatrix , read_images
from plot import subplot
import matplotlib.cm as cm

class PCA():   
    
    
    def __init__(self):

        # <001> read images
        path = "./Assignments/_03/pca/data"
        [self.__X, self.__y] = read_images(path)
        
    def run(self):

        # <002> perform a full pca
        num_components = 100
        [eigenvalues, eigenvectors, mu] = pca(asRowMatrix(self.__X), self.__y, num_components)

        # turn the first (at most ) 16 eigenvectors into grayscale
        # images ( note : eigenvectors are stored by column!)
        E = []
        for i in xrange(min(len(self.__X), 16)):
            e = eigenvectors[:,i].reshape(self.__X[0].shape)
            E.append(normalize(e ,0 ,255))
            
        # plot
        subplot(title ="Eigenfaces", images = E, rows = 4, cols = 4, sptitle = "Eigenface", 
                  colormap = cm.gray , filename = None)
        
        # <006> use different number of eigen faces to construct a face from dataset
        # How many components do you think are enough for realistic reconstruction of a face?
        # (select a face in the data set)
        
        E = []
        # FOR PRINTING IMAGES IN STEPS
        #steps = [20]
        #for i in range(16):
            #last = steps[-1]
            #proj = project(eigenvectors[:, 0:last], self.__X[0].reshape(1,-1), mu)
            #recon = reconstruct(eigenvectors[:, 0:last], proj, mu)
            
            #recon = recon.reshape(self.__X[0].shape)
            #E.append(normalize(recon, 0, 255))     
            #steps.append(last + 20)  # Add a new step
            
        #print steps
        
        # FOR PRINTING SINGLE IMAGE - change rows, cols from 4 to 1!!!
        steps = [num_components]
        proj = project(eigenvectors[:, 0:num_components], self.__X[0].reshape(1,-1), mu)
        recon = reconstruct(eigenvectors[:, 0:num_components], proj, mu)
                    
        recon = recon.reshape(self.__X[0].shape)
        E.append(normalize(recon, 0, 255))          
        
        # plot reconstructed face
        subplot(title = "Reconstruction", images = E, rows = 1, cols = 1, sptitle = "Eigenvectors", 
                sptitles = steps , colormap =cm.gray , filename = None)    