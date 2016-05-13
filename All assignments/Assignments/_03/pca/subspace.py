import numpy as np
import sys

def pca (X, y, number_of_components = 0):
    [n,d] = X.shape
    if (number_of_components <= 0) or (number_of_components > n):
        number_of_components = n
    # <003> implement pca steps explained in the algorithm
    #mu = np.mean(X)
    #S = np.cov(X, mu)
    # center data
    mu = X.mean(axis=0)
    #X = X - mu    
    covariance = np.dot((X - mu), (X - mu).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvectors = np.dot((X - mu).T, eigenvectors)
    for i in xrange(n):
        eigenvectors[:, i] = eigenvectors[:, i]/np.linalg.norm(eigenvectors[:, i])
    
    # <004> sort eigenvectors descending by their eigenvalue
    
    #sortedE = np.sort(eigenvalues, axis=0)  # sort and create a new array
    #np.sort(a)
    #eigenvalues[::-1].sort()

    # Sort in descending order and return the original indexes of the sorted array
    indexes = np.argsort(eigenvalues)[::-1][:n]
    #print indexes
    #sys.exit(0)    
    
    eigenvalues = eigenvalues[indexes]
    eigenvectors = eigenvectors[:, indexes]

    # <005> select only number_of_components
    eigenvalues = eigenvalues[0:number_of_components]
    eigenvectors = eigenvectors[:, 0:number_of_components]

    return [eigenvalues, eigenvectors, mu]



def project(W, X, mu=None):
    if mu is None :
        return np.dot(X,W)
    return np.dot(X - mu , W)



def reconstruct(W, Y, mu=None):
    if mu is None :
        return np.dot(Y,W.T)
    return np.dot(Y, W.T) + mu