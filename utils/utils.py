from sys import stdout
import functools
from numpy import ndarray, allclose, array as nparray
from numpy.core import overrides
from numpy.lib.npyio import _savez

__all__ = ['Verbose', 'allclose_ndarray', 'allclose_array', 'savez']



class Verbose:

  def __init__(self, *args):
    self.verbose = args[0]
    self.lenStr = 0

  def __call__(self, string):
    if self.verbose:
      stdout.write('\r'+' '*self.lenStr+'\r')
      stdout.flush()
      self.lenStr = 0 if string[-2:]=='\n' else len(string)
      stdout.write(string)
      stdout.flush()


class allclose_ndarray(ndarray):

  def __new__(obj, *args, **kwargs):
    return super().__new__(obj, *args, **kwargs)

  def __eq__(self, other):
    if isinstance(other, ndarray):
      if self.shape == other.shape:
        return allclose(self, other)
    return False

  def __ne__(self, other):
    if isinstance(other, ndarray):
      if self.shape == other.shape:
        return not allclose(self, other)
    return True


def allclose_array(obj):
  arr = obj if isinstance(obj, ndarray) else nparray(obj, copy=False)
  ret = allclose_ndarray(
    *arr.shape,
    dtype=arr.dtype,
  )
  ret[:] = arr[:]
  return ret


array_function_dispatch = functools.partial(
  overrides.array_function_dispatch,
  module='numpy'
)


def _savez_dispatcher(file, *args, **kwds):
  yield from args
  yield from kwds.values()


@array_function_dispatch(_savez_dispatcher)
def savez(file, *args, **kwds):
  _savez(file, args, kwds, False, allow_pickle=False)
