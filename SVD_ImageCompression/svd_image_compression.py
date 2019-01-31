# solutions.py
"""Volume 1: The SVD and Image Compression. Solutions File.
R Scott Collings
Math 345 Sec 002
6 Nov 2018"""

from scipy import linalg as la
import numpy as np
from matplotlib import pyplot as plt
from imageio import imread

# Problem 1
def compact_svd(A, tol=1e-6):
    """Compute the truncated SVD of A.

    Parameters:
        A ((m,n) ndarray): The matrix (of rank r) to factor.
        tol (float): The tolerance for excluding singular values.

    Returns:
        ((m,r) ndarray): The orthonormal matrix U in the SVD.
        ((r,) ndarray): The singular values of A as a 1-D array.
        ((r,n) ndarray): The orthonormal matrix V^H in the SVD.
    """
    A_H = A.conj().T
    #get the eigenvalues and eigenvectors of A^H A
    spectrum, V = la.eig(A_H @ A)
    singVals = spectrum ** .5
    #sort the singular values in descending order along with 
    #the corresponding eigenvectors 
    sortedIndices = np.flip(np.argsort(singVals))
    sortedSings = np.array([singVals[i] for i in sortedIndices])
    sortedEigVecs = np.array([V[:,i] for i in sortedIndices])
    #find the singular values that are within the tolerance and throw away 
    #the others along with their eigenvalues
    r = np.sum(sortedSings > tol)
    sigma1 = sortedSings[:r]
    V1 = sortedEigVecs[:r,:]
    #calculate U
    U1 = A @ (V1.T / sigma1)
    return U1, sigma1, V1.conj()


# Problem 2
def visualize_svd(A):
    """Plot the effect of the SVD of A as a sequence of linear transformations
    on the unit circle and the two standard basis vectors.
    """
    #check that array is 2x2 and convert to numpy array if needed
    if type(A) is not np.ndarray:
        A = np.array(A)
    if A.shape[0] != 2 or A.shape[1] != 2 or A.ndim != 2:
        raise ValueError("invalid array input")
    #initialize points and basis vectors and svd
    E = np.array([[1,0,0],[0,0,1]])
    points = np.linspace(0, 2 * np.pi, 200)
    x = np.cos(points)
    y = np.sin(points)
    U,D,V = la.svd(A)
    #plot original circle
    plt.subplot(221)
    plt.plot(E[0],E[1],'k-')
    plt.plot(x,y,'b-')
    plt.gca().set_aspect('equal')
    plt.gca().set_title('S')
    #plot circle with V^H applied to it
    plt.subplot(222)
    step1Basis = V @ E
    step1Points = V @ [x,y]
    plt.plot(step1Basis[0],step1Basis[1],'k-')
    plt.plot(step1Points[0],step1Points[1],'b-')
    plt.gca().set_title('(V^H)S')
    plt.gca().set_aspect('equal')
    #plot circle with Sigma V^H applied to it
    plt.subplot(223)
    step2Basis = np.diag(D) @ step1Basis
    step2Points = np.diag(D) @ step1Points
    plt.plot(step2Basis[0],step2Basis[1],'k-')
    plt.plot(step2Points[0],step2Points[1],'b-')
    plt.gca().set_title('Sigmal(V^H)S')
    plt.gca().set_aspect('equal')
    #plot circle with U Sigma V^H (or A) applied to it
    plt.subplot(224)
    step3Basis = U @ step2Basis
    step3Points = U @ step2Points
    plt.plot(step3Basis[0],step3Basis[1],'k-')
    plt.plot(step3Points[0],step3Points[1],'b-')
    plt.gca().set_title('U(Sigma)(V^H)S')
    plt.gca().set_aspect('equal')
    plt.show()



# Problem 3
def svd_approx(A, s):
    """Return the best rank s approximation to A with respect to the 2-norm
    and the Frobenius norm, along with the number of bytes needed to store
    the approximation via the truncated SVD.

    Parameters:
        A ((m,n), ndarray)
        s (int): The rank of the desired approximation.

    Returns:
        ((m,n), ndarray) The best rank s approximation of A.
        (int) The number of entries needed to store the truncated SVD.
    """
    #get svd and rank of A
    U,D,V = la.svd(A)
    r = D.shape[0]
    #error if desired rank is greater than rank of A
    if s > r:
        raise ValueError("specified rank is greater than matrix rank")
    if s < 0:
        raise ValueError("rank can't be less than 0")
    #find the first s cols of U, sing vals of D, and rows of V^H
    Us = U[:,:s]
    Ds = D[:s]
    Vs = V[:s,:]
    return Us @ np.diag(Ds) @ Vs, Us.size + Ds.size + Vs.size


# Problem 4
def lowest_rank_approx(A, err):
    """Return the lowest rank approximation of A with error less than 'err'
    with respect to the matrix 2-norm, along with the number of bytes needed
    to store the approximation via the truncated SVD.

    Parameters:
        A ((m, n) ndarray)
        err (float): Desired maximum error.

    Returns:
        A_s ((m,n) ndarray) The lowest rank approximation of A satisfying
            ||A - A_s||_2 < err.
        (int) The number of entries needed to store the truncated SVD.
    """
    #get svd and singular values less then error
    U,D,V = la.svd(A)
    i = np.where(D < err)
    #complain if no sing vals are less than error
    if i[0].shape[0] <= 0:
        raise ValueError("cannot approximate matrix with given tolerance")
    #use problem three to compute approximation with rank 1 less than 
    #first sing val smaller than allotted error
    return svd_approx(A,i[0][0])



# Problem 5
def compress_image(filename, s):
    """Plot the original image found at 'filename' and the rank s approximation
    of the image found at 'filename.' State in the figure title the difference
    in the number of entries used to store the original image and the
    approximation.

    Parameters:
        filename (str): Image file path.
        s (int): Rank of new image.
    """
    image = imread(filename)
    #scale the image
    scaled = image / 255
    #if the image is color
    if scaled.ndim == 3:
        #get the approximation for each color and number of entries then combine into one
        Rs,countR = svd_approx(scaled[:,:,0],s)
        Gs,countG = svd_approx(scaled[:,:,1],s)
        Bs,countB = svd_approx(scaled[:,:,2],s)
        imageS = np.dstack((np.clip(Rs,0,1),np.clip(Gs,0,1),np.clip(Bs,0,1)))
        #plot the original and approximate images
        plt.subplot(121)
        plt.imshow(scaled)
        plt.subplot(122)
        plt.imshow(imageS)
        plt.gcf().suptitle(str(scaled.size - (countR + countG + countB)) + ' fewer entries')
        plt.show()
    #if the image is grayscale
    else:
        #approximate the image and get values saved
        imageS,countS = svd_approx(scaled,s)
        #plot original and approximation
        plt.subplot(121)
        plt.imshow(scaled,cmap='gray')
        plt.subplot(122)
        plt.imshow(np.clip(imageS,0,1),cmap='gray')
        plt.gcf().suptitle(str(scaled.size - countS) + ' fewer entries')
        plt.show()
