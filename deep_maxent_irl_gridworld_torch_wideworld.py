import matplotlib.pyplot as plt
import argparse
from collections import namedtuple


import img_utils
from mdp import gridworld
from mdp import value_iteration_wideworld
from deep_maxent_irl_torch_wideworld import *
from maxent_irl import *
from utils import *
from lp_irl import *
import numpy as np

Step = namedtuple('Step','cur_state action next_state reward done')


PARSER = argparse.ArgumentParser(description=None)
PARSER.add_argument('-hei', '--height', default=7, type=int, help='height of the gridworld')
PARSER.add_argument('-wid', '--width', default=7, type=int, help='width of the gridworld')
PARSER.add_argument('-g', '--gamma', default=0.9, type=float, help='discount factor')
PARSER.add_argument('-a', '--act_random', default=0.3, type=float, help='probability of acting randomly')
PARSER.add_argument('-t', '--n_trajs', default=10000, type=int, help='number of expert trajectories')
PARSER.add_argument('-l', '--l_traj', default=100, type=int, help='length of expert trajectory')
PARSER.add_argument('--rand_start', dest='rand_start', action='store_true', help='when sampling trajectories, randomly pick start positions')
PARSER.add_argument('--no-rand_start', dest='rand_start',action='store_false', help='when sampling trajectories, fix start positions')
PARSER.set_defaults(rand_start=False)
PARSER.add_argument('-lr', '--learning_rate', default=0.02, type=float, help='learning rate')
PARSER.add_argument('-ni', '--n_iters', default=5000, type=int, help='number of iterations')
ARGS = PARSER.parse_args()
print ARGS


GAMMA = ARGS.gamma
ACT_RAND = ARGS.act_random
R_MAX = 10# the constant r_max does not affect much the recoverred reward distribution
R_MIN = -30
H = ARGS.height
W = ARGS.width
N_TRAJS = ARGS.n_trajs
L_TRAJ = ARGS.l_traj
RAND_START = ARGS.rand_start
LEARNING_RATE = ARGS.learning_rate
N_ITERS = ARGS.n_iters


def numpyToTensor(x):
  x = torch.from_numpy(x)
  x = Variable(x).float()

  return x

def generate_demonstrations(gw, policy, n_trajs=100, len_traj=20, rand_start=False, start_pos=[0,0]):
  """gatheres expert demonstrations

  inputs:
  gw          Gridworld - the environment
  policy      Nx1 matrix
  n_trajs     int - number of trajectories to generate
  rand_start  bool - randomly picking start position or not
  start_pos   2x1 list - set start position, default [0,0]
  returns:
  trajs       a list of trajectories - each element in the list is a list of Steps representing an episode
  """

  trajs = []
  #rand_start = True
  for i in range(n_trajs):
    if rand_start:
      # override start_pos
      start_pos = [np.random.randint(0, gw.height), np.random.randint(0, gw.width)]

    episode = []
    gw.reset(start_pos)
    cur_state = start_pos
    cur_state, action, next_state, reward, is_done = gw.step(int(policy[gw.pos2idx(cur_state)]))
    episode.append(Step(cur_state=gw.pos2idx(cur_state), action=action, next_state=gw.pos2idx(next_state), reward=reward, done=is_done))
    # while not is_done:
    for _ in range(len_traj):
        cur_state, action, next_state, reward, is_done = gw.step(int(policy[gw.pos2idx(cur_state)]))
        episode.append(Step(cur_state=gw.pos2idx(cur_state), action=action, next_state=gw.pos2idx(next_state), reward=reward, done=is_done))
        if is_done:
            break
        if reward == R_MAX:
          break
    trajs.append(episode)
  return trajs




def main():
  N_STATES = H * W
  N_ACTIONS = 5

  rmap_gt = - np.ones([H, W])
  rmap_gt[H-7, W-4] = R_MIN
  rmap_gt[H-6, W-4] = R_MIN
  rmap_gt[H-5, W-4] = R_MIN
  rmap_gt[H-4, W-4] = R_MIN
  rmap_gt[H-3, W-4] = R_MIN
  rmap_gt[H-2, W-4] = R_MIN
  rmap_gt[H-7, W-5] = R_MIN
  rmap_gt[H-6, W-5] = R_MIN
  rmap_gt[H - 5, W - 5] = R_MIN
  rmap_gt[H - 4, W - 5] = R_MIN
  rmap_gt[H - 3, W - 5] = R_MIN
  rmap_gt[H - 2, W - 5] = R_MIN
  rmap_gt[H-7, W-6] = R_MIN
  rmap_gt[H-6, W-6] = R_MIN
  rmap_gt[H-5, W-6] = R_MIN
  rmap_gt[H-4, W-6] = R_MIN
  rmap_gt[H-3, W-6] = R_MIN
  rmap_gt[H-2, W-6] = R_MIN
  #rmap_gt[0, W-1] = R_MAX
  #rmap_gt[H-1, 0] = R_MAX
  rmap_gt[H-7, W-1] = R_MAX


  gw = gridworld.GridWorld(rmap_gt, {}, 1 - ACT_RAND)


  rewards_gt = np.reshape(rmap_gt, H*W, order = 'F')
  rewards_gt = numpyToTensor(rewards_gt)
  P_a = gw.get_transition_mat()

  values_gt, policy_gt = value_iteration_wideworld.value_iteration(P_a, rewards_gt, GAMMA, error=0.01, deterministic=True, npy=False)
  
  # use identity matrix as feature
  feat_map = torch.eye(N_STATES)
  #feat_map = torch.rand(N_STATES, N_STATES)- 0.5

  trajs = generate_demonstrations(gw, policy_gt, n_trajs=N_TRAJS, len_traj=L_TRAJ, rand_start=RAND_START)
  
  print 'Deep Max Ent IRL training ..'
  Trainer = trainer(feat_map, P_a, GAMMA, trajs, LEARNING_RATE, N_ITERS)
  rewards, mu_D, mu_exp = Trainer.deep_maxent_irl(feat_map, P_a, GAMMA, trajs, ARGS.learning_rate, N_ITERS, rewards_gt)

  values, _ = value_iteration_wideworld.value_iteration(P_a, rewards, GAMMA, error=0.01, deterministic=True, npy = True)

  rewards_gt = rewards_gt.data.numpy()
  values = values.numpy()
  values_gt = values_gt.numpy()

  mu_D = mu_D.numpy()
  mu_exp = mu_exp.numpy()

  # plots
  plt.figure(figsize=(20,4))
  plt.subplot(1, 6, 1)
  img_utils.heatmap2d(np.reshape(rewards_gt, (H,W), order='F'), 'Rewards Map - Ground Truth', block=False)
  plt.subplot(1, 6, 2)
  img_utils.heatmap2d(np.reshape(values_gt, (H,W), order='F'), 'Value Map - Ground Truth', block=False)
  plt.subplot(1, 6, 3)
  img_utils.heatmap2d(np.reshape(rewards, (H,W), order='F'), 'Reward Map - Recovered', block=False)
  plt.subplot(1, 6, 4)
  img_utils.heatmap2d(np.reshape(values, (H,W), order='F'), 'Value Map - Recovered', block=False)
  plt.subplot(1, 6, 5)
  img_utils.heatmap2d(np.reshape(mu_D, (H,W), order='F'), 'SVF - Ground Truth', block=False)
  plt.subplot(1, 6, 6)
  img_utils.heatmap2d(np.reshape(mu_exp, (H,W), order='F'), 'SVF - Recovered', block=False)
  plt.show()


if __name__ == "__main__":
  main()
