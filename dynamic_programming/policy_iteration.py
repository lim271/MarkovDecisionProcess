from time import time
import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve, bicgstab, lgmres
from utils import Verbose



class PolicyIteration:

  def __init__(self, mdp):

    self.mdp = mdp
    self.values = None
    self.terminal_state = False
    # Identity matrix $I_{|s|}$ for computation
    self.__I = sp.identity(
      self.mdp.states.n,
      dtype=np.float32, format='csr'
    )
    self.__innerloop_maxiter = max(
      int(np.sqrt(self.mdp.states.n)),
      100
    )
    self.__innerloop_maxiter = int(self.mdp.states.n)


  def solve(self, max_iteration=1e3, tolerance=1e-8, verbose=True, callback=None):

    self.verbose = Verbose(verbose)
    start_time = time()
    last_time = time()
    current_time = time()

    for iter in range(int(max_iteration)):

      value_diff = self.update()

      current_time = time()
      self.verbose(
        'Iter.: %d, Value diff.: %f, Step time: %f (sec).\n'
        %(iter+1, value_diff, current_time - last_time)
      )
      last_time = current_time

      if callback is not None:
        callback(self)
      if iter > 0:
        if value_diff is np.nan or value_diff is np.inf:
          raise OverflowError('Divergence detected.')

      if value_diff < tolerance:
        break

    current_time = time()
    self.verbose('Time elapsed: %f (sec).\n'%(current_time - start_time))
    del self.verbose


  def update(self, direct_method=True):
        
    # Compute the value difference $|\V_{k}-V_{k+1}|\$ for check the convergence
    if self.values is None:
      self.verbose(
        'Computing initial values...'
      )
      policy = np.argmax(
        self.mdp.rewards.toarray(),
        axis=1
      ).astype(self.mdp.policy.dtype)
      self.mdp.policy.update(
        np.argmax(
          self.mdp.rewards.toarray(),
          axis=1
        ).astype(self.mdp.policy.dtype)
      )
      self.values = np.take_along_axis(
        self.mdp.rewards.toarray(),
        policy[:, np.newaxis],
        axis=1
      ).ravel()
      self.mdp.policy.update(
        policy
      )
      value_diff = np.inf
    else:
      # Compute the value $V(s)$ via solving the linear system $(I-\gamma P^{\pi}), R^{\pi}$
      self.verbose(
        'Constructing linear system...'
      )
      A = self.__I - self.mdp.discount * sp.vstack(
        [self.mdp.state_transition_probability[s, a, :] for s, a in enumerate(self.mdp.policy)],
        format='csr'
      )
      if np.all(A.diagonal()):
        b = self.mdp.rewards[self.mdp.policy.one_hot()]
        if self.mdp.rewards.issparse:
          b = b.T
        if direct_method:
          self.verbose(
            'Solving linear system (SuperLU)...'
          )
          new_values = spsolve(A, b)
        else:
          self.verbose(
            'Solving linear system (BiCGstab)...'
          )
          new_values, info = bicgstab(
            A, b, x0=self.values,
            tol=1e-8,
            maxiter=self.__innerloop_maxiter
          )
          if info < 0:
            self.verbose(
              'BiCGstab failed. Call LGMRES...'
            )
            new_values, info = lgmres(
              A, b, x0=new_values,
              tol=1e-8,
              maxiter=int(
                max(np.sqrt(self.__innerloop_maxiter), 10)
              )
            )

        self.verbose(
          'Updating policy...'
        )
        self.mdp.policy.update(
          np.argmax(
            self.mdp.rewards.toarray() + np.multiply(
              self.mdp.discount,
              self.mdp.state_transition_probability.dot(new_values)
            ).reshape(self.mdp.rewards.shape),
            axis=1
          ).astype(self.mdp.policy.dtype)
        )

      else:
        self.verbose(
          'det(A) is zero. Use value iteration update instead...'
        )
        q = self.mdp.rewards.toarray() + np.multiply(
          self.mdp.discount,
          self.mdp.state_transition_probability.dot(self.values)
        ).reshape(self.mdp.rewards.shape)
        self.verbose(
          'Updating policy...'
        )
        policy = np.argmax(q, axis=1).astype(self.mdp.policy.dtype)
        new_values = np.take_along_axis(q, policy[:, np.newaxis], axis=1).ravel()
        self.mdp.policy.update(
          policy
        )

      value_diff = self.values[:] - new_values[:]
      #if np.mean(value_diff)>0:
      #  self.values = spsolve(A, b) # use spsolve if value function does not improved
      value_diff = np.sqrt(np.dot(value_diff, value_diff) / self.mdp.states.n)

      self.values = new_values.copy()

    return value_diff


  def save(self, filename):
    np.savez(filename, values=self.values, policy=self.mdp.policy.toarray())
