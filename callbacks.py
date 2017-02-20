from __future__ import absolute_import
from __future__ import print_function

import os
import csv

import numpy as np
import time
import json
import warnings

from keras import backend as K
from keras.callbacks import Callback

try:
    import requests
except ImportError:
    requests = None

if K.backend() == 'tensorflow':
    import tensorflow as tf


class ConvergenceStopping(Callback):
    """Stop training when a monitored quantity has reached a certain value (default = 0).

    # Arguments
        monitor: quantity to be monitored.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
    """

    def __init__(self, monitor='loss', value=0,
                 verbose=0, mode='auto'):
        super(ConvergenceStopping, self).__init__()

        self.monitor = monitor
        self.value = value
        self.verbose = verbose
        self.wait = 0
        self.stopped_epoch = 0

        # MODE STUFF (todo: exact)
        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ConvergenceStopping mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less_equal
        elif mode == 'max':
            self.monitor_op = np.greater_equal
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater_equal
            else:
                self.monitor_op = np.less_equal

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('convergence stopping requires %s available!' %
                          (self.monitor), RuntimeWarning)

        if self.monitor_op(current, self.value):
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: convergence stopping' % (self.stopped_epoch))

