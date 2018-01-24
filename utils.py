import numpy as np
import math
from collections import namedtuple

Step = namedtuple('Step','cur_state action next_state reward done')


def normalize(vals):
  """
  normalize to (0, max_val)
  input:
    vals: 1d array
  """
  min_val = np.min(vals)
  max_val = np.max(vals)
  mean_val = np.mean(vals)
  return (vals - min_val) / (max_val - min_val)
  #return vals #/ mean_val


def sigmoid(xs):
  """
  sigmoid function
  inputs:
    xs      1d array
  """
  return [1 / (1 + math.exp(-x)) for x in xs]


def compute_next_state(policy, curr_state, H, W):
  # self.dirs = {0: 'r', 1: 'l', 2: 'd', 3: 'u', 4: 's'}
  action = policy[curr_state[0], curr_state[1]]
  if action == 0:
    if curr_state[1] == W-1:
      next_s = curr_state
    else:
      next_s = [curr_state[0], curr_state[1] + 1]
  elif action == 1:
    if curr_state[1] == 0:
      next_s = curr_state
    else:
      next_s = [curr_state[0], curr_state[1] - 1]
  elif action == 2:
    if curr_state[0] == H-1:
      next_s = curr_state
    else:
      next_s = [curr_state[0] + 1, curr_state[1]]
  elif action == 3:
    if curr_state[0] == 0:
      next_s = curr_state
    else:
      next_s = [curr_state[0] - 1, curr_state[1]]
  elif action == 4:
    next_s = curr_state

  return next_s

def compute_expected_value_difference(policy, rewards_gt, policy_gt, gamma, H, W):
  rewards_gt_np = np.reshape(rewards_gt, (H,W), order='F')
  policy_gt_np = np.reshape(policy_gt.numpy(), (H, W), order='F')
  policy_np = np.reshape(policy.numpy(), (H, W), order='F')
  init_state = [0,0]
  terminate_state = [0,6]
  count = 0
  gt_curr_state = init_state
  learn_curr_state = init_state
  evd_gt = pow(gamma, count) * rewards_gt_np[gt_curr_state[0],gt_curr_state[1]]
  evd_learn = pow(gamma, count) * rewards_gt_np[learn_curr_state[0],learn_curr_state[1]]
  done = True

  while done:
    count += 1
    gt_next_state = compute_next_state(policy_gt_np, gt_curr_state, H, W)
    learn_next_state = compute_next_state(policy_np, learn_curr_state, H, W)

    evd_gt = evd_gt + pow(gamma, count) * rewards_gt_np[gt_next_state[0],gt_next_state[1]]
    evd_learn = evd_learn + pow(gamma, count) * rewards_gt_np[learn_next_state[0],learn_next_state[1]]

    gt_curr_state = gt_next_state
    learn_curr_state = learn_next_state

    if gt_curr_state == terminate_state:
      done = False

  return abs(evd_gt - evd_learn)


def generateObstacle(r_map_1, H, W, Min, Max, obs_type):

  if obs_type == 1:
    for idx in range(H-1):
      r_map_1[H - (H - idx), W - (W - 1)] = Min
      r_map_1[H - (H - idx), W - (W - 2)] = Min
      if W > 5:
        r_map_1[H - (H - idx), W - (W - 3)] = Min
    r_map_1[H - H, W - 1] = Max

  elif obs_type == 2:
    if W == 9:
      r_map_1[H - 9, W - 5] = Min
      r_map_1[H - 8, W - 5] = Min
      r_map_1[H - 7, W - 5] = Min
      r_map_1[H - 5, W - 5] = Min
      r_map_1[H - 4, W - 5] = Min
      r_map_1[H - 3, W - 5] = Min
      r_map_1[H - 2, W - 5] = Min
      r_map_1[H - 1, W - 5] = Min

      r_map_1[H - 9, W - 4] = Min
      r_map_1[H - 8, W - 4] = Min
      r_map_1[H - 7, W - 4] = Min
      r_map_1[H - 5, W - 4] = Min
      r_map_1[H - 4, W - 4] = Min
      r_map_1[H - 3, W - 4] = Min
      r_map_1[H - 2, W - 4] = Min
      r_map_1[H - 1, W - 4] = Min

      r_map_1[H - 9, W - 3] = Min
      r_map_1[H - 8, W - 3] = Min
      r_map_1[H - 7, W - 3] = Min
      r_map_1[H - 5, W - 3] = Min
      r_map_1[H - 4, W - 3] = Min
      r_map_1[H - 3, W - 3] = Min
      r_map_1[H - 2, W - 3] = Min
      r_map_1[H - 1, W - 3] = Min

    elif W == 7:
      r_map_1[H - 7, W - 3] = Min
      r_map_1[H - 6, W - 3] = Min
      r_map_1[H - 5, W - 3] = Min
      r_map_1[H - 3, W - 3] = Min
      r_map_1[H - 2, W - 3] = Min
      r_map_1[H - 1, W - 3] = Min

      r_map_1[H - 7, W - 2] = Min
      r_map_1[H - 6, W - 2] = Min
      r_map_1[H - 5, W - 2] = Min
      r_map_1[H - 3, W - 2] = Min
      r_map_1[H - 2, W - 2] = Min
      r_map_1[H - 1, W - 2] = Min

    elif W == 5:
      r_map_1[H - 5, W - 3] = Min
      r_map_1[H - 4, W - 3] = Min
      r_map_1[H - 3, W - 3] = Min
      r_map_1[H - 1, W - 3] = Min

    r_map_1[H - 1, W - 1] = Max

  elif obs_type == 3:
    for idx in range(H - 1):
      r_map_1[H - (H - idx), W - (W - 1)] = Min
      r_map_1[H - (H - idx), W - (W - 2)] = Min
      if W > 5:
        r_map_1[H - (H - idx), W - (W - 3)] = Min

  return r_map_1

