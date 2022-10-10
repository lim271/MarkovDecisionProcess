from time import time
import numpy as np
from scipy import sparse as sp
from multiprocessing import Pool, Manager, cpu_count
from .__utils import allclose_array, Verbose, savez
from .__dynamic_programming import ValueIteration

__all__ = [
  'States',
  'Actions',
  'Rewards',
  'Policy',
  'StateTransitionProbability',
  'MarkovDecisionProcess'
]



class States:

  def __init__(self, state_list, terminal_state_list=None):

    self.__data = []
    for item in state_list:
      self.__data.append(allclose_array(item) if isinstance(item, np.ndarray) else item)

    self.__terminal_states = []
    if terminal_state_list is None:
      self.__terminal_states.append(None)
    else:
      for item in terminal_state_list:
        self.__terminal_states.append(allclose_array(item) if isinstance(item, np.ndarray) else item)


  def __getitem__(self, key):
    return self.__data[key]


  def __setitem__(self, key, val):
    self.__data[key] = val


  def __iter__(self):
    return self.__data.__iter__()


  @property
  def data(self):
    return self.__data


  @property
  def terminal_states(self):
    return self.__terminal_states


  @property
  def shape(self):
    return (self.n,)


  @property
  def n(self):
    return len(self.__data)


  def index(self, state):
    return self.__data.index(state)


  def tolist(self):
    return list(self.__data)


  def toarray(self):
    return self.__data.copy()


  # End of class States 



class Actions:

  def __init__(self, action_list):

    self.__data = []
    for item in action_list:
      self.__data.append(allclose_array(item) if isinstance(item, np.ndarray) else item)


  def __getitem__(self, key):
    return self.__data[key]


  def __setitem__(self, key, val):
    self.__data[key] = val


  def __iter__(self):
    return self.__data.__iter__()


  @property
  def data(self):
    return self.__data


  @property
  def shape(self):
    return (self.n,)


  @property
  def n(self):
    return len(self.__data)


  def index(self, state):
    return self.__data.index(state)


  def tolist(self):
    return list(self.__data)


  def toarray(self):
    return self.__data.copy()

  # End of class Actions



class Rewards:

  def __init__(self, states, actions, dtype=np.float32, sparse=False):

    shape = (states.n, actions.n)
    if sparse:
      self.__data = sp.dok_matrix(shape, dtype=dtype)
    else:
      self.__data = np.zeros(shape, dtype=dtype)


  def __setitem__(self, key, val):

    self.__data[key] = val


  def __getitem__(self, key):

    return self.__data[key]


  def __iter__(self):

    return self.__data.__iter__()


  @property
  def dtype(self):
    return self.__data.dtype


  @property
  def shape(self):
    return self.__data.shape


  @property
  def issparse(self):
    return isinstance(self.__data, sp.spmatrix)


  def tocsr(self):

    if self.issparse:
      if not isinstance(self.__data, sp.csr_matrix):
        self.__data = self.__data.tocsr()
    else:
      self.__data = sp.csr_matrix(self.__data, dtype=self.dtype)
    return self.__data


  def todok(self):

    if self.issparse:
      if not isinstance(self.__data, sp.dok_matrix):
        self.__data = self.__data.todok()
    else:
      self.__data = sp.dok_matrix(self.__data, dtype=self.dtype)
    return self.__data


  def toarray(self, copy=False):

    if self.issparse:
      return self.__data.toarray()
    else:
      if copy:
        return self.__data.copy()
      else:
        return self.__data


  def update(self, data):
    self.__data = data


  def load(self, filename):
    filetype = filename.split('.')[-1]
    if filetype=='npz':
      self.__data = sp.load_npz(filename)
    elif filetype=='npy':
      self.__data = np.load(filename)


  def save(self, filename):
    if self.issparse:
      self.__data = sp.save_npz(filename, self.__data)
    else:
      self.__data = np.save(filename, self.__data)

  # End of class Rewards



class StateTransitionProbability:

  def __init__(self, states, actions, dtype=np.float32):

    self.__data = sp.dok_matrix(
      (states.n * actions.n, states.n),
      dtype=dtype
    )


  def __setitem__(self, key, val):

    if isinstance(key, tuple):
      if len(key)==1:
        return self.__data[key[0]]
      elif len(key)==2:
        self.__data[key] = val
      elif len(key)==3:
        if isinstance(key[0], slice):
          for idx1, idx2 in enumerate(
            range(
              0 if key[0].start is None else key[0].start,
              self.shape[0] if key[0].stop is None else key[0].stop,
              1 if key[0].step is None else key[0].step
            )
          ):
            if isinstance(key[1], slice):
              self.__data[
                slice(
                  np.ravel_multi_index(
                    (idx2, 0 if key[1].start is None else key[1].start),
                    self.shape[:2]
                  ),
                  np.ravel_multi_index(
                    (idx2, self.shape[1] - 1 if key[1].stop is None else key[1].stop - 1),
                    self.shape[:2]
                  ) + 1,
                  1 if key[1].step is None else key[1].step
                ),
                key[2]
              ] = val[idx1]
            else:
              self.__data[
                np.ravel_multi_index((idx2, key[1]), self.shape[:2]),
                key[2]
              ] = val[idx1]
        else:
          if isinstance(key[1], slice):
            self.__data[
              slice(
                np.ravel_multi_index(
                  (key[0], 0 if key[1].start is None else key[1].start),
                  self.shape[:2]
                ),
                np.ravel_multi_index(
                  (key[0], self.shape[1] - 1 if key[1].stop is None else key[1].stop - 1),
                  self.shape[:2]
                ) + 1,
                1 if key[1].step is None else key[1].step
              ),
              key[2]
            ] = val
          else:
            self.__data[
              np.ravel_multi_index(key[:2], self.shape[:2]),
              key[2]
            ] = val
      else:
        raise IndexError('Indices mismatch.')
    elif isinstance(key, int) or isinstance(key, slice):
      self.__data[key] = val
    else:
      raise IndexError('Indices mismatch.')


  def __getitem__(self, key):

    if isinstance(key, tuple):
      if len(key)==1:
        return self.__data[key[0]]
      elif len(key)==2:
        return self.__data[key]
      elif len(key)==3:
        if isinstance(key[0], slice):
          val = []
          for idx in range(
            0 if key[0].start is None else key[0].start,
            self.shape[0] if key[0].stop is None else key[0].stop,
            1 if key[0].step is None else key[0].step
          ):
            if isinstance(key[1], slice):
              val.append(
                self.__data[
                  slice(
                    np.ravel_multi_index(
                      (idx, 0 if key[1].start is None else key[1].start),
                      self.shape[:2]
                    ),
                    np.ravel_multi_index(
                      (idx, self.shape[1] - 1 if key[1].stop is None else key[1].stop - 1),
                      self.shape[:2]
                    ) + 1,
                    1 if key[1].step is None else key[1].step
                  ),
                  key[2]
                ]
              )
            else:
              val.append(
                self.__data[
                  np.ravel_multi_index((idx, key[1]), self.shape[:2]),
                  key[2]
                ]
              )
          return val
        else:
          if isinstance(key[1], slice):
            return self.__data[
              slice(
                np.ravel_multi_index(
                  (key[0], 0 if key[1].start is None else key[1].start),
                  self.shape[:2]
                ),
                np.ravel_multi_index(
                  (key[0], self.shape[1] - 1 if key[1].stop is None else key[1].stop - 1),
                  self.shape[:2]
                ) + 1,
                1 if key[1].step is None else key[1].step
              ),
              key[2]
            ]
          else:
            return self.__data[
              np.ravel_multi_index(key[:2], self.shape[:2]),
              key[2]
            ]
    elif isinstance(key, int) or isinstance(key, slice):
      return self.__data[key]
    else:
      raise IndexError('Indices mismatch.')


  def __iter__(self):
    return self.__data.__iter__()


  @property
  def dtype(self):
    return self.__data.dtype


  @property
  def shape(self):
    sa, s = self.__data.shape
    return (s, sa // s, s)


  def dot(self, other):
    return self.__data.dot(other)


  def tocsr(self):
    if not sp.isspmatrix_csr(self.__data):
      self.__data = self.__data.tocsr()
  

  def todok(self):
    if not sp.isspmatrix_dok(self.__data):
      self.__data = self.__data.todok()


  def tospmat(self):
    return self.__data


  def toarray(self):
    return self.__data.toarray()


  def update(self, data):
    self.__data = data


  def load(self, filename):
    self.__data = np.load(filename, allow_pickle=True)


  def save(self, filename):
    np.save(filename, self.__data, allow_pickle=True)

  # End of class StateTransitionProbability 



class Policy:

  def __init__(self, states, actions, dtype=int):
    self.__states = states
    self.__actions = actions
    self.__data  = None
    self.__I = np.eye(self.__actions.n, dtype=bool)
    self.reset(dtype=dtype)


  def __setitem__(self, key, val):
    if isinstance(self.dtype, int):
      self.__data[key] = min(val, self.__actions.n - 1)
    else:
      self.__data[key] = np.clip(val, 0, 1)


  def __getitem__(self, key):
    return self.__data[key]


  def __iter__(self):
    return self.__data.__iter__()


  def __str__(self):
    return self.__data.__str__()


  @property
  def dtype(self):
    return self.__data.dtype


  @property
  def shape(self):
    return self.__data.shape


  @property
  def one_hot_array(self):
    if isinstance(self.dtype, int):
      return self.__I[self.__data]
    else:
      return self.__I[self.__data.argmax(axis=1)]


  def get_action(self, state):
    return self.__actions[self.get_action_index(state)]


  def get_action_index(self, state):
    if isinstance(self.dtype, int):
      idx = self.__states.index(state)
    else:
      idx = self.__states.index(state)
    return int(self.__data[idx, :].argmax())


  def update(self, data):
    if isinstance(data.dtype, int):
      self.__data = data
    else:
      self.__data = sp.csr_matrix(data)


  def reset(self, dtype=int):
    if isinstance(dtype, int):
      self.__data = np.random.randint(
        0, self.__actions.n, self.__states.n, dtype=dtype
      )
    else:
      self.__data = sp.csr_matrix(
        (
          np.ones((self.__states.n,)),
          (
            np.arange(0, self.__states.n, dtype=int),
            np.random.randint(
              0, self.__actions.n, self.__states.n, dtype=dtype
            )
          )
        ),
        shape=(self.__states.n, self.__actions.n)
      )


  def toarray(self, copy=False):
    if isinstance(self.dtype, int):
      if copy:
        return self.__data.copy()
      else:
        return self.__data
    else:
      return self.__data.toarray()


  def load(self, filename):
    self.__data = np.load(filename, allow_pickle=True)


  def save(self, filename):
    np.save(filename, self.__data, allow_pickle=True)

  # End of class Policy



class MarkovDecisionProcess:

  def __init__(self, states, actions,
    rewards=None,
    state_transition_probability=None,
    policy=None,
    discount = 0.99,
  ):

    self.states  = states
    self.actions = actions
    self.rewards = Rewards(self.states, self.actions) if rewards is None else rewards
    self.discount = max(
      min(
        np.array(discount, dtype=self.rewards.dtype).item(),
        np.array(1, dtype=self.rewards.dtype).item() - np.finfo(self.rewards.dtype).epsneg
      ),
      np.array(0, dtype=self.rewards.dtype).item()
    )
    self.state_transition_probability = StateTransitionProbability(
      self.states, self.actions
    ) if state_transition_probability is None else state_transition_probability
    self.policy = Policy(self.states, self.actions) if policy is None else policy
    self.values = None
    self.__sampler = None
    self.__sample_reward = False


  def _worker(self, queue, state):
    spmat = sp.dok_matrix(
      self.state_transition_probability.shape[1:],
      dtype=self.state_transition_probability.dtype
    )
    if self.__sample_reward:
      arr = np.zeros(self.rewards.shape[1:], dtype=self.rewards.dtype)
    if not allclose_array(state) in self.states.terminal_states:
      for action_idx, action in enumerate(self.actions):
        if self.__sample_reward:
          next_states, probs, reward = self.__sampler(state, action)
          arr[action_idx] += reward
        else:
          next_states, probs = self.__sampler(state, action)
        next_state_indices = [
          self.states.index(next_state) for next_state in next_states
        ]
        for next_state_idx, prob in zip(next_state_indices, probs):
          spmat[action_idx, next_state_idx] += prob
    queue.put(1)
    if self.__sample_reward:
      return np.array([spmat.tocsr(), arr], dtype=object)
    else:
      return spmat.tocsr()


  def sample(self, sampler, sample_reward=False, verbose=True):

    verbose = Verbose(verbose)
    verbose('Start sampling...')
    start_time = time()
    self.__sampler = sampler
    self.__sample_reward = sample_reward
    queue = Manager().Queue()
    with Pool(cpu_count()) as p:
      data = p.starmap_async(
        self._worker,
        [(queue, state) for state in self.states]
      )
      counter = 0
      tic = time()
      while counter < self.states.n:
        counter += queue.get()
        if time()-tic > 0.1:
          progress = counter / self.states.n
          rt = (time() - start_time) * (1 - progress) / progress
          rh = rt // 3600
          rt %= 3600
          rm = rt // 60
          rs = rt % 60
          progress *= 100
          verbose('Sampling progress: %5.1f %%... (%dh %dm %ds rem.)'%(progress, rh, rm, rs))
          tic = time()
      if self.__sample_reward:
        data = np.array(
          data.get(),
          dtype=object
        )
        self.state_transition_probability.update(
          sp.vstack(data[:, 0])
        )
        self.rewards.update(
          np.array(
            data[:, 1].tolist(),
            dtype=self.rewards.dtype
          )
        )
      else:
        self.state_transition_probability.update(
          sp.vstack(
            data.get()
          )
        )
    self.__sampler = None
    end_time = time()
    verbose('Sampling is done. %f (sec) elapsed.\n'%(end_time - start_time))


  def solve(self, max_iteration=1e3, tolerance=1e-8, verbose=True, callback=None, parallel=True):

    solver = ValueIteration(self)
    solver.solve(max_iteration=max_iteration, tolerance=tolerance, verbose=verbose, callback=callback, parallel=parallel)
    self.values = solver.values


  def load(self, filename):

    data = np.load(filename, allow_pickle=False)
    self.states = States(
      data['states.data'],
      terminal_states = data['states.terminal_states']
    )
    self.actions = Actions(data['actions.data'])
    self.rewards = Rewards(
      self.states,
      self.actions,
      sparse=data['rewards.issparse'].item()
    )
    if self.rewards.issparse:
      self.rewards.update(
        sp.csr_matrix(
          (
            data['rewards.data'],
            data['rewards.indices'],
            data['rewards.indptr']
          ),
          shape=(self.states.n, self.actions.n)
        )
      )
    else:
      self.rewards.update(data['rewards.data'])

    self.state_transition_probability = StateTransitionProbability(
      self.states,
      self.actions
    )
    self.state_transition_probability.update(
      sp.csr_matrix(
        (
          data['state_transition_probability.data'],
          data['state_transition_probability.indices'],
          data['state_transition_probability.indptr']
        ),
        shape=(
          self.states.n * self.actions.n,
          self.states.n
        )
      )
    )
    self.policy = Policy(
      self.states,
      self.actions
    )
    self.policy.update(
      data['policy.data']
    )
    self.discount = data['discount'].item()
    

  def save(self, filename):

    if not sp.isspmatrix_csr(self.state_transition_probability.tospmat()):
      self.state_transition_probability.tocsr()

    kwargs = {
      'states.data': self.states.toarray(),
      'states.terminal_states': self.states.terminal_states,
      'actions.data': self.actions.toarray(),
      'rewards.issparse': self.rewards.issparse,
      'state_transition_probability.data': self.state_transition_probability.tospmat().data,
      'state_transition_probability.indices': self.state_transition_probability.tospmat().indices,
      'state_transition_probability.indptr': self.state_transition_probability.tospmat().indptr,
      'policy.data': self.policy.toarray(),
      'discount': self.discount
    }
    if self.rewards.issparse:
      kwargs['rewards.data'] = self.rewards.tocsr().data,
      kwargs['rewards.indices'] = self.rewards.tocsr().indices,
      kwargs['rewards.indptr'] = self.rewards.tocsr().indptr,
    else:
      kwargs['rewards.data'] = self.rewards.toarray()

    savez(filename, **kwargs)

  # End of class MarkovDecisionProcess


if __name__=="__main__":

  states = States(np.linspace(0, 1, 100))

  actions = Actions(np.linspace(0, 1, 10))

  rewards = Rewards(states, actions)

  state_transition_prob = StateTransitionProbability(states, actions)
  for s in range(states.n):
    for a in range(actions.n):
      state_transition_prob[s, a, s] = 1.

  policy = Policy(states, actions)

  mdp = MarkovDecisionProcess(
    states=states,
    actions=actions,
    rewards=rewards,
    state_transition_probability=state_transition_prob,
    policy=policy,
    discount=0.99
  )
