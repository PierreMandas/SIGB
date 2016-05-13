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
        self.__num_components = 0
        
    def run(self):

        # <002> perform a full pca
        [eigenvalues, eigenvectors, mu] = pca(asRowMatrix(self.__X), self.__y, self.__num_components)

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
            
        #print steps
        
        # FOR PRINTING SINGLE IMAGE - change rows, cols from 4 to 1!!!

        
        # plot reconstructed face
        #subplot(title = "Reconstruction", images = E, rows = 1, cols = 1, sptitle = "Eigenvectors", 
                #sptitles = steps , colormap =cm.gray , filename = None)
                
        E = self.__reconstruct(eigenvectors, mu, 1, 1, None)
                
    def __reconstruct(self, eigenvectors, mu, numrows=4, numcols=4, numsteps=20):
        '''
        Reconstruct the faces from eigenvectors. 
        Change numrows, numcols to 1 and numsteps to None to show single image
        '''
        E = []
        if numsteps is not None:
            # Multiple images
            steps = [numsteps]
            for i in range(16):
                last = steps[-1]
                proj = project(eigenvectors[:, 0:last], self.__X[0].reshape(1,-1), mu)
                recon = reconstruct(eigenvectors[:, 0:last], proj, mu)
        
                recon = recon.reshape(self.__X[0].shape)
                E.append(normalize(recon, 0, 255))     
                steps.append(last + 20)  # Add a new step              
        else:
            # Single image
            if self.__num_components == 0:
                self.__num_components = 400
            steps = [self.__num_components]
            proj = project(eigenvectors[:, 0:self.__num_components], self.__X[0].reshape(1,-1), mu)
            recon = reconstruct(eigenvectors[:, 0:self.__num_components], proj, mu)
        
            recon = recon.reshape(self.__X[0].shape)
            E.append(normalize(recon, 0, 255))     
            
        subplot(title = "Reconstruction", images = E, rows = numrows, cols = numcols, sptitle = "Eigenvectors", 
                sptitles = steps , colormap =cm.gray , filename = None)        
                 