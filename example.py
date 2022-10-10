import numpy as np
from mdp import (
  States,
  Actions,
  Rewards,
  Policy,
  StateTransitionProbability,
  MarkovDecisionProcess
)
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

N_R = 20
N_ALPHA = 36



r_list = np.linspace(20.0/N_R, 20.0, N_R, dtype=np.float32)
alpha_list = np.linspace(-np.pi + np.pi/N_ALPHA, np.pi - np.pi/N_ALPHA, N_ALPHA, dtype=np.float32)



def sampler(state, action):
  sol = solve_ivp(
    lambda t, y: np.array(
      [-np.cos(y[1]), np.sin(y[1]) / y[0] - action]
    ),
    [0.0, 0.1],
    state
  )
  r, alpha = sol.y[:, -1]
  if r <= r_list[0]:
    rs = [r_list[0]]
  elif r >= r_list[-1]:
    rs = [r_list[-1]]
  else:
    ri = np.searchsorted(r_list, r)
    if r_list[ri] == r:
      rs = [r_list[ri]]
    elif r_list[ri] > r:
      rs = [*r_list[ri-1:ri+1]]
    else:
      rs = [*r_list[ri:ri+2]]
  drs = []
  for item in rs:
    drs.append(abs(item - r))
  prs = drs[::-1] / np.sum(drs)

  if alpha <= alpha_list[0]:
    alphas = [alpha_list[0]]
  elif alpha >= alpha_list[-1]:
    alphas = [alpha_list[-1]]
  else:
    alphai = np.searchsorted(alpha_list, alpha)
    if alpha_list[alphai] == alpha:
      alphas = [alpha_list[alphai]]
    elif alpha_list[alphai] > alpha:
      alphas = [*alpha_list[alphai-1:alphai+1]]
    else:
      alphas = [*alpha_list[alphai:alphai+2]]
  dalphas = []
  for item in alphas:
    dalphas.append(abs(item - alpha))
  palphas = dalphas[::-1] / np.sum(dalphas)

  states = []
  probs = []
  for r, pr in zip(rs, prs):
    for alpha, palpha in zip(alphas, palphas):
      states.append(np.array([r, alpha]))
      probs.append(pr * palpha)

  reward = -(r - 10.0) ** 2

  return states, probs, reward



if __name__=="__main__":

  state_list = np.vstack([np.repeat(r_list, len(alpha_list)), np.tile(alpha_list, len(r_list))]).T
  states = States(state_list)

  action_list = np.linspace(-1, 1, 11)
  actions = Actions(action_list)

  mdp = MarkovDecisionProcess(states=states, actions=actions, discount=0.99)
  mdp.sample(sampler, sample_reward=True, verbose=True)

  mdp.solve()

  plt.figure(1)
  plt.imshow(mdp.values.reshape((len(r_list), len(alpha_list))))

  plt.figure(2)
  plt.imshow(mdp.policy.toarray().reshape((len(r_list), len(alpha_list))))

  plt.show()
