from builtins import range
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #initialize the variable
    num_train = X.shape[0]
    num_classes = W.shape[1]
    
    for i in range(num_train):
        e_term = 0
        scores = X[i].dot(W)
        for j in range(num_classes):
            e_term += np.exp(scores[j])
            
        loss += -scores[y[i]] + np.log(e_term)
        #calculate dW
        for j in range(num_classes):
            if j != y[i]:
                dW[:,j] += (np.exp(scores[j])/e_term)*X[i]
            if j == y[i]:
                dW[:,j] += (np.exp(scores[j])/e_term - 1)*X[i]
    loss /= num_train
    loss += 0.5*reg*np.sum(W*W)#怎么确定要不要×2或1/2
    dW = dW / num_train + reg * W#系数怎么定0.0
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #initialize the variable
    num_train = X.shape[0]
    num_classes = W.shape[1]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # loss = [sum(-f_yi) + sum(log(sum(exp(f_j))))] / num_train + R(W)
    # loss = term_1   +  term_2            +  term_3
    '''
    scores_all = X.dot(W)#scores_all's shape:(train_num , classes_num)
    
    e_term_column_vector = np.sum(np.exp(scores_all) , 1)#np.exp(scores_all)'s shape:(train_num , classes_num)
    term_2 = np.sum(np.log(e_term_column_vector))
    term_1 = -np.sum(scores_all[range(num_train),list(y)])
    term_3 = 0.5*reg*np.sum(W*W)
    loss = (term_1 + term_2) / num_train + term_3
    '''
    
    scores = X.dot(W)#[num_train , num_classes]
    shift_scores = scores - np.max(scores, axis = 1).reshape(-1,1)
    softmax_output = np.exp(shift_scores)/np.sum(np.exp(shift_scores), axis = 1).reshape(-1,1)
    loss = -np.sum(np.log(softmax_output[range(num_train), list(y)]))
    loss /= num_train 
    loss +=  0.5* reg * np.sum(W * W)
    dS = softmax_output.copy()
    dS[range(num_train), list(y)] += -1
    dW = (X.T).dot(dS)
    dW = dW/num_train + reg* W

    
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
