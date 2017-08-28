import tensorflow as tf
tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops
from tensorflow.contrib.compiler import jit
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

class ConvLSTMCell(tf.contrib.rnn.RNNCell):

  def __init__(self, shape, num_filters, kernel_size, forget_bias=1.0, 
               input_size=None, state_is_tuple=True, activation=tf.nn.tanh, reuse=None):
    self._shape = shape
    self._num_filters = num_filters
    self._kernel_size = kernel_size
    self._size = tf.TensorShape(shape+[self._num_filters])

    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation
    self._reuse = reuse 
  @property
  def state_size(self):
    return (tf.contrib.rnn.LSTMStateTuple(self._size, self._size)
            if self._state_is_tuple else 2 * self._num_units)

  @property
  def output_size(self):
    return self._size  

  def __call__(self, inputs, state, scope=None):
    # we suppose inputs to be [time, batch_size, row, col, channel]
    with tf.variable_scope(scope or "basic_convlstm_cell", reuse=self._reuse):
      if self._state_is_tuple:
        c, h = state
      else:
        c, h = array_ops.split(value=state, num_or_size_splits=2, axis=3)

      inp_channel = inputs.get_shape().as_list()[-1]+self._num_filters
      out_channel = self._num_filters * 4
      concat = tf.concat([inputs, h], axis=3)

      kernel = tf.get_variable('kernel', shape=self._kernel_size+[inp_channel, out_channel])
      concat = tf.nn.conv2d(concat, filter=kernel, strides=(1,1,1,1), padding='SAME') 

      i, j, f, o = tf.split(value=concat, num_or_size_splits=4, axis=3)

      new_c = (c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) *
               self._activation(j))
      new_h = self._activation(new_c) * tf.sigmoid(o)
      if self._state_is_tuple:
        new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
      else:
        new_state = tf.concat([new_c, new_h], 3)
      return new_h, new_state
