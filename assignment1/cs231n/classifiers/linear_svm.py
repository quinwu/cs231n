import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    '''
    W.shape is (D,C)
    X[i].shape is (1,D)
    X[i].dot(W).shape is (1,C)
    '''
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                dW[:,j] += X[i]
                dW[:,y[i]] += - X[i]
            

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    loss += 0.5 *  reg * np.sum(W * W)
    dW += reg*W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    num_classes = W.shape[1]
    num_train = X.shape[0]
    # print X.shape
    # print y.shape
    '''
    W.shape is (D,C)
    X.shape is (N,D)
    X.dot(W).shape is (N,C)
    '''
    scores = X.dot(W)
    '''
    correct_class_scores = (1*n)
    '''
    correct_class_scores = scores[np.arange(num_train),y].reshape(-1,1)
    margin = np.maximum(0,scores - correct_class_scores + 1)
    margin[np.arange(num_train),y] = 0
    loss = np.sum(margin)
    loss /= num_train
    loss += 0.5 * reg * np.sum(W*W)


    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    flag = np.zeros_like(margin)
    flag[margin > 0] = 1
    
    incorrect_count = np.sum(flag,axis = 1)

    flag[np.arange(num_train),y] = -incorrect_count

    '''
    flag.shape = margin.shape = scores.shape is (N,C)
    X.shape is (N,D)
    '''
    dW = X.T.dot(flag)
    dW /= num_train
    dW += reg*W
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
