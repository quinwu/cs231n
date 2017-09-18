import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    num_classes = W.shape[1]
    '''
    X.shape is (N,D)
    W.shape is (D,C)
    scores = X[i].dot(W) shape is (1,C)
    '''
    for i in xrange(num_train):
        scores = X[i].dot(W)
        exp_scores = np.exp(scores - np.max(scores))
        exp_sum = np.sum(exp_scores)
        P = exp_scores / exp_sum
        loss += -np.log(P[y[i]])

        # dW[:, y[i]] -= X[i, :]
        # dW += np.outer(X[i, :],P)
        for j in xrange(num_classes):
            if j == y[i]:
                dW[:,j] += (P[j] - 1)*X[i]
            else:
                dW[:,j] += P[j] *X[i]

    loss /= num_train
    loss += 0.5 * reg * np.sum(W*W)    
    dW /= num_train
    dW += reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    '''
    X.shape is (N,D)
    W.shape is (D,C)
    scores = X.dot(W) shape is (N,C)
    P.shape is (N,C)
    '''
    num_train = X.shape[0]
    
    scores = X.dot(W)
    exp_scores = np.exp(scores - np.max(scores,axis=1,keepdims=True))
    exp_sum = np.sum(exp_scores,axis = 1,keepdims=True)
    P = exp_scores / exp_sum

    loss += np.sum(-np.log(P[np.arange(num_train),y]))
    loss /= num_train
    loss += 0.5*reg*np.sum(W*W)

    ind = np.zeros_like(P)
    ind[np.arange(num_train),y] = 1
    
    dW = X.T.dot(P - ind)
    dW /= num_train
    dW += reg*W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
