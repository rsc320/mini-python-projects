# nearest_neighbor.py
"""Volume 2: Nearest Neighbor Search.
R Scott Collings
Math 321
29 Sept 2018
"""

import numpy as np
from math import sqrt
from scipy import spatial as sp
from scipy import stats
from matplotlib import pyplot as plt


# Problem 1
def exhaustive_search(X, z):
    """Solve the nearest neighbor search problem with an exhaustive search.

    Parameters:
        X ((m,k) ndarray): a training set of m k-dimensional points.
        z ((k, ) ndarray): a k-dimensional target point.

    Returns:
        ((k,) ndarray) the element (row) of X that is nearest to z.
        (float) The Euclidean distance from the nearest neighbor to z.
    """
    difference = X - z
    squares = np.sum(difference ** 2,axis=1) #get the distance squared
    index = np.argmin(squares) #find the smalleest distance away
    return X[index], round(sqrt(squares[index]),10)


# Problem 2: Write a KDTNode class.
class KDTNode:
    """A node for a k-dimensional binary tree

    Attributes:
        left: the pointer to the node of the left child
        right: the pointer to the node of the right child
        value (np.ndarray): the value of the k-dimensional node
        pivot: the dimension on which to compare children
    """
    def  __init__(self, x):
        """Initializes the KDTNode, other attributes assigned on insertion to tree
        """
        if type(x) != np.ndarray:
            raise TypeError("Must be np.ndarray") #only accepts an ndarray
        else:
            self.value = x
            self.left = None
            self.right = None
            self.pivot = None


# Problems 3 and 4
class KDT:
    """A k-dimensional binary tree for solving the nearest neighbor problem.

    Attributes:
        root (KDTNode): the root node of the tree. Like all other nodes in
            the tree, the root has a NumPy array of shape (k,) as its value.
        k (int): the dimension of the data in the tree.
    """
    def __init__(self):
        """Initialize the root and k attributes."""
        self.root = None #declares attributes of KDT tree
        self.k = None

    def find(self, data):
        """Return the node containing the data. If there is no such node in
        the tree, or if the tree is empty, raise a ValueError.
        """
        def _step(current):
            """Recursively step through the tree until finding the node
            containing the data. If there is no such node, raise a ValueError.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree")
            elif np.allclose(data, current.value):
                return current                      # Base case 2: data found!
            elif data[current.pivot] < current.value[current.pivot]:
                return _step(current.left)          # Recursively search left.
            else:
                return _step(current.right)         # Recursively search right.

        # Start the recursive search at the root of the tree.
        return _step(self.root)

    # Problem 3
    def insert(self, data):
        """Insert a new node containing the specified data.

        Parameters:
            data ((k,) ndarray): a k-dimensional point to insert into the tree.

        Raises:
            ValueError: if data does not have the same dimensions as other
                values in the tree.
        """
        def add_child(node, data):
            if not self.root:
                self.root = KDTNode(data)
                self.root.pivot = 0
                self.k = data.shape[0] #assign the dimension of the tree
            else:
                if data.shape[0] != self.k or np.ndim(data) != 1:
                    raise ValueError("must be " + str(self.k) + "-dimensional")
                elif np.allclose(data, node.value):
                    raise ValueError(str(data) + " is already in tree")
                elif data[node.pivot] < node.value[node.pivot]:
                    if node.left:
                        add_child(node.left, data)
                    else: #add as left child if no current left child
                        node.left = KDTNode(data)
                        node.left.pivot = (node.pivot + 1) % self.k #find new pivot
                else:
                    if node.right:
                        add_child(node.right, data)
                    else: #add as right child if no current right child
                        node.right = KDTNode(data)
                        node.right.pivot = (node.pivot + 1) % self.k

        add_child(self.root, data) #start at root for insert

    # Problem 4
    def query(self, z):
        """Find the value in the tree that is nearest to z.

        Parameters:
            z ((k,) ndarray): a k-dimensional target point.

        Returns:
            ((k,) ndarray) the value in the tree that is nearest to z.
            (float) The Euclidean distance from the nearest neighbor to z.
        """
        def kd_search(node, nearest, distance):
            if node == None:
                return nearest, distance
            x = node.value
            i = node.pivot
            newDistance = sqrt(np.sum((x - z) ** 2))
            if newDistance < distance:
                distance = newDistance
                nearest = node
            if z[i] < x[i]: #search to the left
                nearest, distance = kd_search(node.left,nearest,distance)
                if z[i] + distance >= x[i]: #search to the right if intersects separating plane
                    nearest, distance = kd_search(node.right,nearest,distance)
            else: #search to the right
                nearest, distance = kd_search(node.right,nearest,distance)
                if z[i] - distance <= x[i]: #search to the left is intersects separtaing plane
                    nearest, distance = kd_search(node.left,nearest,distance)
            return nearest,distance
        
        if not self.root:
            raise ValueError('No nodes in tree to compare against')
        elif z.shape[0] != self.k: #make sure target had same dimension as tree
            raise ValueError('The target is not in the same dimension as the tree')
        nearest, distance = kd_search(self.root,self.root,sqrt(np.sum(((self.root.value - z) ** 2))))
        return nearest.value, round(distance,10)


    def __str__(self):
        """String representation: a hierarchical list of nodes and their axes.

        Example:                           'KDT(k=2)
                    [5,5]                   [5 5]   pivot = 0
                    /   \                   [3 2]   pivot = 1
                [3,2]   [8,4]               [8 4]   pivot = 1
                    \       \               [2 6]   pivot = 0
                    [2,6]   [7,5]           [7 5]   pivot = 0'
        """
        if self.root is None:
            return "Empty KDT"
        nodes, strs = [self.root], [] #gets nodes in level order
        while nodes:
            current = nodes.pop(0)
            strs.append("{}\tpivot = {}".format(current.value, current.pivot))
            for child in [current.left, current.right]:
                if child:
                    nodes.append(child)
        return "KDT(k={})\n".format(self.k) + "\n".join(strs)


# Problem 5: Write a KNeighborsClassifier class.
class KNeighborsClassifier:
    """A k-dimensional binary tree for solving the nearest neighbor problem.

    Attributes:
        root (KDTNode): the root node of the tree. Like all other nodes in
            the tree, the root has a NumPy array of shape (k,) as its value.
        k (int): the dimension of the data in the tree.
    """
    
    def __init__(self,n_neighbors):
        """Defines how many neighbors to include in a K-neighbors vote

        Parameters:
            n_neighbors (int): number of neighbors to inclued in vote
        """
        self.k = n_neighbors #set number of neighbors that vote

    def fit(self,trainingSet,trainingLabels):
        """Initializes the training set and labels of a KDTree
        
        Parameters:
            trainingSet (np.ndarray): an mxn array of m KDTree values
            trainingLabel (np.ndarray): a mx1 array of labels for the training set
        """
        if type(trainingSet) != np.ndarray or type(trainingLabels) != np.ndarray:
            raise TypeError("Must be np.ndarray")
        if trainingSet.shape[0] != trainingLabels.shape[0]: #make sure there is a label for each object
            raise IndexError("Array sizes not compatible")
        self.tree = sp.KDTree(trainingSet) #create tree
        self.labels = trainingLabels
        

    def predict(self,query):
        """Find k-neighbors of the query and return the most common label

        Parameters:
            query (np.ndarray): a nx1 array to find the nearest neighbors too

        Returns:
            label: the most common label of the self.k closest neighbors
        """
        neighbors, indices = self.tree.query(query,self.k)
        if self.k == 1:
            indices = np.array([indices])
        labels = np.zeros_like(indices)
        i = 0
        for x in indices:
            labels[i] = self.labels[x] #get the labels of each closest neighbor
            i += 1
        label,count = stats.mode(labels,axis=None,nan_policy='omit') #mind the mode
        return label[0]


# Problem 6
def prob6(n_neighbors, filename="mnist_subset.npz"):
    """Extract the data from the given file. Load a KNeighborsClassifier with
    the training data and the corresponding labels. Use the classifier to
    predict labels for the test data. Return the classification accuracy, the
    percentage of predictions that match the test labels.

    Parameters:
        n_neighbors (int): the number of neighbors to use for classification.
        filename (str): the name of the data file. Should be an npz file with
            keys 'X_train', 'y_train', 'X_test', and 'y_test'.

    Returns:
        (float): the classification accuracy.
    """
    data = np.load(filename)
    X_train = np.array(data["X_train"]).astype(np.float)
    y_train = np.array(data["y_train"])
    X_test = np.array(data["X_test"]).astype(np.float)
    y_test = np.array(data["y_test"])
    kdt = KNeighborsClassifier(n_neighbors)
    kdt.fit(X_train,y_train)
    i = 0
    labels = np.zeros_like(y_test)
    for x in X_test:
        label = kdt.predict(x)
        if label == y_test[i]: #if predicted label is same as actual
            labels[i] = 1
        i += 1
    return float(np.sum(labels) / len(labels)) #count 1s and divide by possile 1s

if __name__ == "__main__":
    pass
