from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import collections
import contextlib
# import hashlib
# import math
# import numbers

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest


_BIAS_VARIABLE_NAME = "biases"
_WEIGHTS_VARIABLE_NAME = "weights"


@contextlib.contextmanager
def _checked_scope(cell, scope, reuse=None, **kwargs):
  if reuse is not None:
    kwargs["reuse"] = reuse
  with vs.variable_scope(scope, **kwargs) as checking_scope:
    scope_name = checking_scope.name
    if hasattr(cell, "_scope"):
      cell_scope = cell._scope  # pylint: disable=protected-access
      if cell_scope.name != checking_scope.name:
        raise ValueError(
            "Attempt to reuse RNNCell %s with a different variable scope than "
            "its first use.  First use of cell was with scope '%s', this "
            "attempt is with scope '%s'.  Please create a new instance of the "
            "cell if you would like it to use a different set of weights.  "
            "If before you were using: MultiRNNCell([%s(...)] * num_layers), "
            "change to: MultiRNNCell([%s(...) for _ in range(num_layers)]).  "
            "If before you were using the same cell instance as both the "
            "forward and reverse cell of a bidirectional RNN, simply create "
            "two instances (one for forward, one for reverse).  "
            "In May 2017, we will start transitioning this cell's behavior "
            "to use existing stored weights, if any, when it is called "
            "with scope=None (which can lead to silent model degradation, so "
            "this error will remain until then.)"
            % (cell, cell_scope.name, scope_name, type(cell).__name__,
               type(cell).__name__))
    else:
      weights_found = False
      try:
        with vs.variable_scope(checking_scope, reuse=True):
          vs.get_variable(_WEIGHTS_VARIABLE_NAME)
        weights_found = True
      except ValueError:
        pass
      if weights_found and reuse is None:
        raise ValueError(
            "Attempt to have a second RNNCell use the weights of a variable "
            "scope that already has weights: '%s'; and the cell was not "
            "constructed as %s(..., reuse=True).  "
            "To share the weights of an RNNCell, simply "
            "reuse it in your second calculation, or create a new one with "
            "the argument reuse=True." % (scope_name, type(cell).__name__))

    # Everything is OK.  Update the cell's scope and yield it.
    cell._scope = checking_scope  # pylint: disable=protected-access
    yield checking_scope


class AttentionBasedGRUCell(RNNCell):
    """an Attention Based Gated Recurrent Unit cell
    (cf. https://arxiv.org/abs/1603.01417).
    """

    def __init__(self, num_units, input_size=None, activation=tanh, reuse=None):
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self._activation = activation
        self._reuse = reuse

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with _checked_scope(
                self, scope or "attention_based_gru_cell", reuse=self._reuse):
            with vs.variable_scope("gates"):
                # We start with bias of 1.0 to not reset and not update.
                inputs, g_t = array_ops.split(
                    inputs, num_or_size_splits=[self._num_units, 1], axis=1)
                reset_gate = sigmoid(_linear(
                    [inputs, state], self._num_units, True, 1.0))
            with vs.variable_scope("candidate"):
                h_tilde = self._activation(_linear(
                    [inputs, reset_gate * state], self._num_units, True))
                new_h = g_t * h_tilde + (1 - g_t) * state
        return new_h, new_h


def _linear(args, output_size, bias, bias_start=0.0):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  scope = vs.get_variable_scope()
  with vs.variable_scope(scope) as outer_scope:
    weights = vs.get_variable(
        _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size], dtype=dtype)
    if len(args) == 1:
      res = math_ops.matmul(args[0], weights)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), weights)
    if not bias:
      return res
    with vs.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      biases = vs.get_variable(
          _BIAS_VARIABLE_NAME, [output_size],
          dtype=dtype,
          initializer=init_ops.constant_initializer(bias_start, dtype=dtype))
    return nn_ops.bias_add(res, biases)
