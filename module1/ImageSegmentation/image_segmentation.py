# image_segmentation.py
"""Volume 1: Image Segmentation.
R Scott Collings
Math 345 Sec 002
2 Nov 2018"""

import numpy as np
from scipy import linalg as spla
from scipy import sparse
from imageio import imread
from matplotlib import pyplot as plt
from scipy.sparse import linalg as sparsespla

# Problem 1
def laplacian(A):
    """Compute the Laplacian matrix of the graph G that has adjacency matrix A.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.

    Returns:
        L ((N,N) ndarray): The Laplacian matrix of G.
    """
    #put the sum of the columns in a diagonal array
    D = np.diag(np.sum(A,axis=0))
    return D - A


# Problem 2
def connectivity(A, tol=1e-8):
    """Compute the number of connected components in the graph G and its
    algebraic connectivity, given the adjacency matrix A of G.

    Parameters: 
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.
        tol (float): Eigenvalues that are less than this tolerance are
            considered zero.

    Returns:
        (int): The number of connected components in G.
        (float): the algebraic connectivity of G.
    """
    #get real part of eigen-values of the Laplacian
    eigVals = spla.eigvals(laplacian(A))
    eigVals = np.real(eigVals)
    print(eigVals)
    #get second smallest eigenvalue
    algCon = np.sort(abs(eigVals))[1]
    #count how many eigenVals are essentially 0
    return sum(abs(eigVals) < tol), algCon


# Helper function for problem 4.
def get_neighbors(index, radius, height, width):
    """Calculate the flattened indices of the pixels that are within the given
    distance of a central pixel, and their distances from the central pixel.

    Parameters:
        index (int): The index of a central pixel in a flattened image array
            with original shape (radius, height).
        radius (float): Radius of the neighborhood around the central pixel.
        height (int): The height of the original image in pixels.
        width (int): The width of the original image in pixels.

    Returns:
        (1-D ndarray): the indices of the pixels that are within the specified
            radius of the central pixel, with respect to the flattened image.
        (1-D ndarray): the euclidean distances from the neighborhood pixels to
            the central pixel.
    """
    # Calculate the original 2-D coordinates of the central pixel.
    row, col = index // width, index % width

    # Get a grid of possible candidates that are close to the central pixel.
    r = int(radius)
    x = np.arange(max(col - r, 0), min(col + r + 1, width))
    y = np.arange(max(row - r, 0), min(row + r + 1, height))
    X, Y = np.meshgrid(x, y)

    # Determine which candidates are within the given radius of the pixel.
    R = np.sqrt(((X - col)**2 + (Y - row)**2))
    mask = R < radius
    return (X[mask] + Y[mask]*width).astype(np.int), R[mask]


# Problems 3-6
class ImageSegmenter:
    """Class for storing and segmenting images."""

    # Problem 3
    def __init__(self, filename):
        """Read the image file. Store its brightness values as a flat array."""
        image = imread(filename)
        #find out if image is already scaled, create scaled image
        if image.max() > 1:
            self.scaled = image / 255
        else:
            self.scaled = image
        #find out if image is already grayscale, create brightness image, then flatten brightness image
        if image.ndim == 3:
            self.brightness = np.ravel(self.scaled.mean(axis=2))
        else:
            self.brightness = np.ravel(self.scaled)
    
    # Problem 3
    def show_original(self):
        """Display the original image."""
        #see if image is color or grayscale
        if self.scaled.ndim == 3:
            plt.imshow(self.scaled)
        else:
            plt.imshow(self.scaled, cmap="gray")
        plt.show()

    # Problem 4
    def adjacency(self, r=5., sigma_B2=.02, sigma_X2=3.):
        """Compute the Adjacency and Degree matrices for the image graph."""
        #initialize arrays of proper size
        A = sparse.lil_matrix((len(self.brightness),len(self.brightness)),dtype=float)
        D = np.zeros_like(self.brightness)
        m = self.scaled.shape[0]
        n = self.scaled.shape[1]
        for x in range(len(self.brightness)):
            #find pixels within r distance and their distance
            J,R = get_neighbors(x,r,m,n)
            #use shi and malik's algorithm to find weights between each pair of pixels within radius
            weights = np.exp([-(radius / sigma_X2) - (abs(self.brightness[x] - self.brightness[j]) / sigma_B2) for radius,j in zip(R,J)])
            #add weights to adjacency array
            A[x, J] = weights
            #get sum of weights in each column of adjacency array
            D[x] = np.sum(A[x])
        return sparse.csc_matrix(A),D

    # Problem 5
    def cut(self, A, D):
        """Compute the boolean mask that segments the image."""
        #return array which is used for finding eigen-vectors according to shi and malik algorithm
        L = sparse.csgraph.laplacian(A)
        degree = sparse.diags(1 / D ** .5)
        shimalik = degree @ L @ degree
        #find two eigenvectors of two lowest eigen-values
        eigs = sparsespla.eigsh(shimalik,which='SM',k=2)
        #using 2nd smallest value's vector, create a mask for the vector based on positive=true, negative=false
        segmenterEig = np.reshape(eigs[1][:,1],(self.scaled.shape[0],self.scaled.shape[1]))
        segmenterMask = segmenterEig > 0
        return segmenterMask


    # Problem 6
    def segment(self, r=5., sigma_B=.02, sigma_X=3.):
        """Display the original image and its segments."""
        A,D = self.adjacency(r,sigma_B,sigma_X)
        mask1 = self.cut(A,D)
        #invert the mask
        mask2 = np.invert(mask1)
        #see if image is color or grayscale
        if self.scaled.ndim == 3:
            #split the image with the mask then the inversion of the mask for the two image segments
            segment1 = np.dstack((mask1 * self.scaled[:,:,0],mask1 * self.scaled[:,:,1],mask1 * self.scaled[:,:,2]))
            segment2 = np.dstack((mask2 * self.scaled[:,:,0],mask2 * self.scaled[:,:,1],mask2 * self.scaled[:,:,2]))
            #plot color image, image segment 1 and image segment 2
            plt.subplot(131)
            plt.imshow(self.scaled)
            plt.axis("off")
            plt.subplot(132)
            plt.imshow(segment1)
            plt.axis("off")
            plt.subplot(133)
            plt.imshow(segment2)
            plt.axis("off")
        else:
            #plot grayscale image and segments
            plt.subplot(131)
            plt.imshow(self.scaled,cmap="gray")
            plt.axis("off")
            plt.subplot(132)
            plt.imshow(mask1 * self.scaled,cmap="gray")
            plt.axis("off")
            plt.subplot(133)
            plt.imshow(mask2 * self.scaled,cmap="gray")
            plt.axis("off")
        plt.show()



# if __name__ == '__main__':
#     ImageSegmenter("dream_gray.png").segment()
#     ImageSegmenter("dream.png").segment()
#     ImageSegmenter("monument_gray.png").segment()
#     ImageSegmenter("monument.png").segment()
