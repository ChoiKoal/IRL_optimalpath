
import mdp.gridworld as gridworld
import mdp.value_iteration as value_iteration
import img_utils
import tf_utils
from utils import *
import torch
import torch.nn as nn
from torch import optim
import torchvision
from torch.autograd import Variable
from network import build_c_network
import numpy as np


R_MAX= 10
H = 5
W = 5
R_MIN = -100
ACT_RAND = 0.2
rmap_gt = np.ones([H, W])
rmap_gt[H - 5, W - 4] = R_MIN
rmap_gt[H - 4, W - 4] = R_MIN
rmap_gt[H - 3, W - 4] = R_MIN
rmap_gt[H - 2, W - 4] = R_MIN

# rmap_gt[0, W-1] = R_MAX
# rmap_gt[H-1, 0] = R_MAX
rmap_gt[H - 5, W - 1] = R_MAX
gw = gridworld.GridWorld(rmap_gt, {}, 1 - ACT_RAND)

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
        if cur_state == [H - 5, W - 1]:
          break
    trajs.append(episode)
  return trajs

def compute_state_visition_freq(P_a, gamma, trajs, policy, deterministic=True):
  """compute the expected states visition frequency p(s| theta, T) 
  using dynamic programming

  inputs:
    P_a     NxNxN_ACTIONS matrix - transition dynamics
    gamma   float - discount factor
    trajs   list of list of Steps - collected from expert
    policy  Nx1 vector (or NxN_ACTIONS if deterministic=False) - policy

  
  returns:
    p       Nx1 vector - state visitation frequencies
  """
  N_STATES, _, N_ACTIONS = np.shape(P_a)

  T = len(trajs[0])
  # mu[s, t] is the prob of visiting state s at time t
  mu = torch.zeros([N_STATES, T])

  for traj in trajs:
    mu[traj[0].cur_state, 0] += 1
  mu[:,0] = mu[:,0]/len(trajs)

  P_a = torch.from_numpy(P_a)

  if deterministic:
    policy = policy
  else:
    policy = torch.transpose(policy, 0, 1)
  for s in range(N_STATES):
    for t in range(T-1):
      if deterministic:
        for pre_s in range(N_STATES):
          mu[s, t+1] += mu[pre_s, t]*P_a[pre_s, s, int(policy[pre_s])]
      else:
        for pre_s in range(N_STATES):
          mu[s, t+1] += sum([mu[pre_s, t]*P_a[pre_s, s, a1]*policy[pre_s, a1] for a1 in range(N_ACTIONS)])
  p = torch.sum(mu, 1)
  return p


def demo_svf(trajs, n_states):
  """
  compute state visitation frequences from demonstrations
  
  input:
    trajs   list of list of Steps - collected from expert
  returns:
    p       Nx1 vector - state visitation frequences   
  """

  p = torch.zeros(n_states)
  for traj in trajs:
    for step in traj:
      p[step.cur_state] += 1
  p = p/len(trajs)
  return p

class trainer:
  def __init__(self, n_input, lr, n_h1=25, n_h2=20, l2=10, name='deep_irl_fc'):
    self.n_input = n_input
    self.lr = lr
    self.n_h1 = n_h1
    self.n_h2 = n_h2
    self.name = name
    self.networks = build_c_network()
    # self.network = self.network.cuda()

    self.finetune(allow=True)

    # self.sess = tf.Session()
    # self.input_s, self.reward, self.theta = self.build_network(self.name)
    self.optimizer = optim.SGD(self.networks.parameters(), lr=0.001)
    # self.optimizer = tf.train.GradientDescentOptimizer(lr)

    self.l2_loss = nn.MSELoss()

  def finetune(self, allow=True):
    for param in self.networks.parameters():
      param.requires_grad = True

  def deep_maxent_irl(self, feat_map, P_a, gamma, trajs, lr, n_iters, rewards_gt):
    """
    Maximum Entropy Inverse Reinforcement Learning (Maxent IRL)

    inputs:
      feat_map    NxD matrix - the features for each state
      P_a         NxNxN_ACTIONS matrix - P_a[s0, s1, a] is the transition prob of
                                         landing at state s1 when taking action
                                         a at state s0
      gamma       float - RL discount factor
      trajs       a list of demonstrations
      lr          float - learning rate
      n_iters     int - number of optimization steps

    returns
      rewards     Nx1 vector - recoverred state rewards
    """
    self.finetune(allow = True)
    # tf.set_random_seed(1)
    self.rewards_gt = rewards_gt
    N_STATES, _, N_ACTIONS = np.shape(P_a)
    self.lr= lr

    # init nn model
    feat_map = feat_map.view(1, 1, 25, 25)
    feat_map = Variable(feat_map)


    value = torch.zeros(H, W)
    # find state visitation frequencies using demonstrations
    mu_D = demo_svf(trajs, N_STATES)
    self.networks.train()
    iteration  = 0
    # training
    for iteration in range(n_iters):
    #while True:


      # compute the reward matrix
      rewards = self.networks(feat_map)

      # compute policy
      rewards = rewards.view(25, 1)
      value_new, policy = value_iteration.value_iteration(P_a, rewards, gamma, error=0.01, deterministic=True, npy=False)

      #dyna algorithm
      if iteration % (n_iters/10) ==0 and iteration/ (n_iters/10) > 30:
        # get new trajs
        trajs_new = generate_demonstrations(gw, policy, n_trajs=10, len_traj=20, rand_start=True)
        trajs+=trajs_new

        # update gt_svf
        mu_D = demo_svf(trajs, N_STATES)

      # compute expected svf
      mu_exp = compute_state_visition_freq(P_a, gamma, trajs, policy, deterministic=True)

      # compute gradients on rewards:
      rewards = rewards.view(25)

      # Set Gradient
      torch.autograd.backward([rewards], [-(mu_D-mu_exp)*self.lr])

      # Update / Optimizer: SGD
      self.optimizer.step()

      #iteration += 1

      # break
      if torch.mean(torch.abs(value - value_new)) < 1e-3:
        break

      # print
      if iteration % (50) == 0:
        print ('iteration: %.d , value_diff: %.4f, svf_diff: %.4f'
               % (iteration, torch.mean(torch.abs(value-value_new)), torch.mean(mu_D-mu_exp)))
      self.networks.zero_grad()

      value = value_new
      # apply gradients to the neural network
      # grad_theta, l2_loss, grad_norm = nn_r.apply_grads(feat_map, grad_r)

    # get output
    self.networks.eval()
    rewards = self.networks(feat_map)

    # return sigmoid(normalize(rewards))
    rewards = rewards.view(25, 1)
    rewards = rewards.data.numpy() # for normalize
    return normalize(rewards), mu_D, mu_exp

  def numpyToTensor(self, x):
    x = torch.from_numpy(x)
    x = Variable(x).float()

    return x




