# markov_chains.py
"""Volume II: Markov Chains.
R Scott Collings
Math 321 Sec 002
22 Oct 2018
"""

import numpy as np
from math import sqrt


# Problem 1
def random_chain(n):
    """Create and return a transition matrix for a random Markov chain with
    'n' states. This should be stored as an nxn NumPy array.
    """
    transition = np.random.random((n,n))
    print(transition)
    #normalize the columns of the random array so they add to 1
    transition = transition / np.sum(transition,axis=0)
    return transition


# Problem 2
def forecast(days):
    """Forecast weather for 'days' days given that today is hot."""
    transition = np.array([[0.7, 0.6], [0.3, 0.4]])
    forecast = np.zeros(days).astype(int)
    # Sample from a binomial distribution to choose a new state 'days' times.
    for i in range(days):
        if i == 0:
            forecast[i] = np.random.binomial(1, transition[1, 0])
        else:
            forecast[i] = np.random.binomial(1, transition[1, forecast[i-1]])
    return forecast


# Problem 3
def four_state_forecast(days):
    """Run a simulation for the weather over the specified number of days,
    with mild as the starting state, using the four-state Markov chain.
    Return a list containing the day-by-day results, not including the
    starting day.

    Examples:
        >>> four_state_forecast(3)
        [0, 1, 3]
        >>> four_state_forecast(5)
        [2, 1, 2, 1, 1]
    """
    transition = np.array([[0.5,  0.3, 0.1, 0],
                            [0.3, 0.3, 0.3, 0.3],
                            [0.2, 0.3, 0.4, 0.5],
                            [0,   0.1, 0.2, 0.2]])
    forecast = np.zeros(days).astype(int)
    #sample from multinomial distribution 'days' times to change state
    for i in range(days):
        if i == 0:
            forecast[i] = (np.random.multinomial(1, transition[:,1])).tolist().index(1)
        else:
            forecast[i] = (np.random.multinomial(1, transition[:,forecast[i-1]])).tolist().index(1)
    return forecast


# Problem 4
def steady_state(A, tol=1e-12, N=40):
    """Compute the steady state of the transition matrix A.

    Inputs:
        A ((n,n) ndarray): A column-stochastic transition matrix.
        tol (float): The convergence tolerance.
        N (int): The maximum number of iterations to compute.

    Raises:
        ValueError: if the iteration does not converge within N steps.

    Returns:
        x ((n,) ndarray): The steady state distribution vector of A.
    """
    #create state distribution vector
    x_old = np.random.random(A.shape[0])
    x_old = x_old / np.sum(x_old)
    while N > 0:
        x_new = A @ x_old
        diff = x_new - x_old
        #norm of difference
        lengthDiff = np.sum(diff ** 2) ** .5
        if lengthDiff < tol:
            return x_new
        x_old = x_new
        N -= 1
    raise ValueError("A^k does not converge")



# Problems 5 and 6
class SentenceGenerator(object):
    """Markov chain creator for simulating bad English.

    Attributes:
        (what attributes do you need to keep track of?)

    Example:
        >>> yoda = SentenceGenerator("Yoda.txt")
        >>> print(yoda.babble())
        The dark side of loss is a path as one with you.
    """
    def __init__(self, filename):
        """Read the specified file and build a transition matrix from its
        contents. You may assume that the file has one complete sentence
        written on each line.
        """
        contents = []
        with open(filename,'r') as file:
            contents = file.readlines()
            for i in range(len(contents)):
                contents[i] = contents[i].strip('\n')
        #get all words and render as a set to get rid of duplicates then put back in list
        self.words = ['$tart'] + list(set((' '.join(contents)).split(' '))) + ['$top']
        self.wordCount = len(self.words)
        self.transitionMatrix = np.zeros((self.wordCount,self.wordCount))
        for sentence in contents:
            sentence = sentence.split(" ")
            for j in range(len(sentence) + 1):
                #$tart transitions to first word
                if j == 0:
                    y = 0
                #previous word transitions to this word
                else:
                    y = self.words.index(sentence[j-1])
                #last word transitions to $top
                if j == len(sentence):
                    x = -1
                else:
                    x = self.words.index(sentence[j])
                self.transitionMatrix[x,y] += 1
        #$top transitions to itself only
        self.transitionMatrix[-1,-1] += 1
        #normalize transition matrix
        self.transitionMatrix = self.transitionMatrix / np.sum(self.transitionMatrix,axis=0)



    def babble(self):
        """Begin at the start sate and use the strategy from
        four_state_forecast() to transition through the Markov chain.
        Keep track of the path through the chain and the corresponding words.
        When the stop state is reached, stop transitioning and terminate the
        sentence. Return the resulting sentence as a single string.
        """
        wordsUsed = []
        currentWord = "$tart"
        #sample from multinomial distribution 'days' times to change state
        j = 0
        while currentWord != "$top":
            #get the next word by probability using current word
            if currentWord == '$tart':
                wordsUsed.append((np.random.multinomial(1, self.transitionMatrix[:,0])).tolist().index(1))
            else:
                wordsUsed.append((np.random.multinomial(1, self.transitionMatrix[:,wordsUsed[j-1]])).tolist().index(1))
            currentWord = self.words[wordsUsed[j]]
            j += 1
        thisBabble = ""
        #add words into a sentence 'thisBabble'
        for word in wordsUsed:
            if self.words[word] != '$top':
                thisBabble += ' ' + self.words[word]
        return thisBabble.strip(' ')

if __name__ == "__main__":
    sg = SentenceGenerator("yoda.txt")
    sg.babble()
