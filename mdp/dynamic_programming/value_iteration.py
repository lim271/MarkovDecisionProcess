from multiprocessing import Array, Value, Process, cpu_count
from time import time
import numpy as np
from scipy import sparse as sp
from ..utils import Verbose



class ValueIteration:

  def __init__(self, mdp):

    self.mdp = mdp
    self.values = np.max(
      self.mdp.rewards.toarray(),
      axis=1
    ) if mdp.values is None else mdp.values


  def solve(self, max_iteration=1e3, tolerance=1e-8, verbose=True, callback=None, parallel=True):

    self.verbose = Verbose(verbose)
    start_time = time()
    last_time = time()
    current_time = time()

    if parallel:
      self.shared = np.frombuffer(
        Array(
          np.ctypeslib.as_ctypes_type(self.mdp.rewards.dtype),
          int(self.mdp.states.n * self.mdp.actions.n)
        ).get_obj(),
        dtype=self.mdp.rewards.dtype
      )
      self.values = np.frombuffer(
        Array(
          np.ctypeslib.as_ctypes_type(self.mdp.rewards.dtype),
          self.values
        ).get_obj(),
        dtype=self.mdp.rewards.dtype
      )
      def _worker(P, key, flag):
        while True:
          if flag.value > 0:
            self.shared[key] = P.dot(self.values)
            flag.value = 0
          elif flag.value < 0:
            break
      chunksize = np.ceil(self.mdp.states.n * self.mdp.actions.n / cpu_count())
      self.workers = []
      for pid in range(cpu_count()):
        key = slice(
          int(chunksize * pid),
          int(
            min(
              chunksize * (pid+1),
              self.mdp.states.n * self.mdp.actions.n
            )
          ),
          None
        )
        flag = Value('i', 0)
        self.workers.append(
          (
            Process(
              target=_worker,
              args=(self.mdp.state_transition_probability[key], key, flag)
            ),
            flag
          )
        )
        self.workers[-1][0].start()

    for iter in range(int(max_iteration)):
      value_diff = self.update(parallel=parallel)
      current_time = time()
      self.verbose(
        'Iter.: %d, Value diff.: %f, Step time: %f (sec).\n'
        %(iter+1, value_diff, current_time - last_time)
      )
      last_time = current_time

      if callback is not None:
        callback(self)

      if value_diff is np.nan or value_diff is np.inf:
        raise OverflowError('Divergence detected.')

      if value_diff < tolerance:
        break

    for p, flag in self.workers:
      flag.value = -1
      p.join()
    self.verbose('Time elapsed: %f (sec).\n'%(current_time - start_time))
    del self.verbose
    

  def update(self, parallel=True):

    self.verbose(
      'Computing action values...'
    )
    if parallel:
      for _, flag in self.workers:
        flag.value = 1
      for _, flag in self.workers:
        while True:
          if flag.value==0:
            break
      q = self.mdp.rewards.toarray() + np.multiply(
        self.mdp.discount,
        self.shared.reshape(self.mdp.rewards.shape)
      )
    else:
      q = self.mdp.rewards.toarray() + np.multiply(
        self.mdp.discount,
        self.mdp.state_transition_probability.dot(
          self.values
        ).reshape(self.mdp.rewards.shape)
      )

    self.verbose(
      'Updating policy...'
    )
    policy = np.argmax(q, axis=1).astype(self.mdp.policy.dtype)
    new_values = np.take_along_axis(q, policy[:, np.newaxis], axis=1).ravel()
    self.mdp.policy.update(
      policy
    )

    value_diff = self.values[:] - new_values[:]
    value_diff = np.sqrt(np.dot(value_diff, value_diff) / self.mdp.states.n)

    self.values[:] = new_values.copy()

    return value_diff
