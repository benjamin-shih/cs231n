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
  dW = np.zeros(W.shape)
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, y[i]] = dW[:, y[i]] - X[i]
        dW[:,j] = dW[:,j] + X[i]
  loss /= num_train
  dW = dW / num_train
  loss += reg * np.sum(W * W)
  dW = dW + reg * 2 * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W) #why is it X dot W rather than W dot X, in the equation it is W dot X
  correct_class_score = scores[np.arange(num_train), y].reshape(num_train,1) #find out what the rearrangement of matrix is (into waht shape)
  margin = np.maximum(0, scores - correct_class_score + 1)
  margin[np.arange(num_train), y] = 0 #ignore the correct class in loss *****
  loss = margin.sum() / num_train
  loss += reg * np.sum(W * W) #why is it np.sum(W^2) instead of np.square(W)
  
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

  
  margin[margin > 0] = 1
  valid_margin_count = margin.sum(axis =1)
  margin[np.arange(num_train), y] -= valid_margin_count
  dW = (X.T).dot(margin) / num_train
  dW = dW + reg * 2 * W
  '''
  margin[margin > 0] = 1
  valid_margin_count = margin.sum(axis = 1) #axis = 1 is columns
  margin[np.arange(num_train),y ] -= valid_margin_count
  dW = (X.T).dot(margin) / num_train #why is it the transapose of X
  
  dW = dW + reg * 2 * W
  '''
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
