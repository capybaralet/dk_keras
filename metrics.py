from keras import backend as K
from keras.utils.generic_utils import get_from_module

import numpy
np = numpy

def hinge(y_true, y_pred):
    return K.mean(K.maximum(1. - y_true * y_pred, 0.))

# FIXME! what is happening when loss = 0 and accuracy = 1 ???


#------------------------------------------------------------------
# STRUCTURED SVM LOSS
def get_softmax_hinge(margin=1.):
    return lambda yt, yp: softmax_hinge(yt, yp, margin)

# TODO: rm log_domain param??
def softmax_hinge(y_true, y_pred, margin=1., log_domain=False):
    p_true = y_true * y_pred # the probability assigned the target
    if log_domain:
        log_p_true = (K.sum(p_true, axis=-1, keepdims=True))
    else:
        log_p_true = K.log(K.sum(p_true, axis=-1, keepdims=True))
    #return log_p_true
    #return K.maximum(margin - (log_p_true - K.log(y_pred)), 0.) * (1 - y_true)
    # the difference between the logits for the correct class vs. others must exceed the margin
    # FIXME: this isn't the right loss... here I am penalizing ALL differences, not just the max...
    return K.mean(K.sum(K.maximum(margin - (log_p_true - K.log(y_pred)), 0.) * (1 - y_true), axis=-1))

def np_softmax_hinge(y_true, y_pred, margin=1., log_domain=False):
    p_true = y_true * y_pred # the probability assigned the target
    if log_domain:
        log_p_true = (np.sum(p_true, axis=-1, keepdims=True))
    else:
        log_p_true = np.log(np.sum(p_true, axis=-1, keepdims=True))
    log_p_true = np.log(np.sum(p_true, axis=-1, keepdims=True))
    #return log_p_true
    #return K.maximum(margin - (log_p_true - K.log(y_pred)), 0.) * (1 - y_true)
    # the difference between the logits for the correct class vs. others must exceed the margin
    return np.mean(np.sum(np.maximum(margin - (log_p_true - np.log(y_pred)), 0.) * (1 - y_true), axis=-1))

#------------------------------------------------------------------
# ONE-vs-ALL SVM LOSS
def get_onevsall_hinge(margin=1.):
    return lambda yt, yp: onevsall_hinge(yt, yp, margin)

def onevsall_hinge(y_true, y_pred, margin=1.):
    p_true = y_true * y_pred # the probability assigned the target
    log_p_true = K.log(K.sum(p_true, axis=-1, keepdims=True))
    return K.mean(K.sum(K.maximum(margin - log_p_true, 0.) * y_true, axis=-1))

def np_onevsall_hinge(y_true, y_pred, margin=1.):
    p_true = y_true * y_pred # the probability assigned the target
    log_p_true = np.log(np.sum(p_true, axis=-1, keepdims=True))
    return np.mean(np.sum(np.maximum(margin - log_p_true, 0.) * y_true, axis=-1))


#------------------------------------------------------------------
# TESTS

# TODO: verbose options of functions

# these arrays should have zero cost
y_true = np.array([[1,0,0], [1,0,0]])
y_pred = np.array([[1 - 2e-9, 1e-9, 1e-9], [1 - 2e-9, 1e-9, 1e-9]])
print np_softmax_hinge(y_true, y_pred)

# these should not
y_true = np.array([[1,0,0], [1,0,0]])
y_pred = np.array([[1e-99, 1 - 2e-99, 1e-99], [1 - 2e-9, 1e-9, 1e-9]])
print np_softmax_hinge(y_true, y_pred)
y_true = np.array([[1,0,0], [1,0,0]])
y_pred = np.array([[1e-99,1,0], [1e-99,0,1]])
print np_softmax_hinge(y_true, y_pred)

# what is the right margin?
y_true = np.array([[1,0,0], [1,0,0]])
y_pred = np.array([[.8, .1, .1], [.8, .1, .1]])
print np_softmax_hinge(y_true, y_pred)
y_true = np.array([[1,0,0], [1,0,0]])
y_pred = np.array([[.5, .25, .25], [.5, .25, .25]])
print np_softmax_hinge(y_true, y_pred)

#------------------------------------------------------------------
# RM below?

def get(identifier):
    return get_from_module(identifier, globals(), 'metric')


def np_onevsall_hinge(y_true, y_pred, margin=1.):
    p_true = y_true * y_pred # the probability assigned the target
    log_p_true = np.log(np.sum(p_true, axis=-1, keepdims=True))
    return np.mean(np.sum(np.maximum(margin - log_p_true, 0.) * y_true, axis=-1))
