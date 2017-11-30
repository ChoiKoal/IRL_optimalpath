
import mdp.gridworld as gridworld
import mdp.value_iteration_wideworld as value_iteration
import img_utils
import tf_utils
from utils import *
import torch
import torch.nn as nn
from torch import optim
import torchvision
from torch.autograd import Variable
from network import build_network_c_wideworld
import numpy as np



def compute_state_visition_freq(P_a, gamma, trajs, policy, deterministic=False):
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

  for s in range(N_STATES):
    for t in range(T-1):
      if deterministic:
        for pre_s in range(N_STATES):
          mu[s, t+1] += mu[pre_s, t]*P_a[pre_s, s, int(policy[pre_s])]
      else:
        mu[s, t+1] = torch.sum(torch.sum([mu[pre_s, t]*P_a[pre_s, s, a1]*policy[pre_s, a1] for a1 in range(N_ACTIONS)]) for pre_s in range(N_STATES))
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
  def __init__(self, n_input, lr, n_h1=25, n_h2=20, l2=10, name='deep_irl_fc_wideworld'):
    self.n_input = n_input
    self.lr = lr
    self.n_h1 = n_h1
    self.n_h2 = n_h2
    self.name = name
    self.networks = build_network_c_wideworld()
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

    # init nn model
    feat_map = feat_map.view(1, 1, 49, 49)
    feat_map = Variable(feat_map)



    # find state visitation frequencies using demonstrations
    mu_D = demo_svf(trajs, N_STATES)
    self.networks.train()

    # training
    for iteration in range(n_iters):
      if iteration % (n_iters/10) == 0:
        print 'iteration: {}'.format(iteration)
      self.networks.zero_grad()

      # compute the reward matrix
      rewards = self.networks(feat_map)

      # compute policy
      rewards = rewards.view(49, 1)
      _, policy = value_iteration.value_iteration(P_a, rewards, gamma, error=0.01, deterministic=True, npy=False)

      # compute expected svf
      mu_exp = compute_state_visition_freq(P_a, gamma, trajs, policy, deterministic=True)

      # compute gradients on rewards:
      rewards = rewards.view(49)

      # Set Gradient
      torch.autograd.backward([rewards], [-(mu_D-mu_exp)])

      # Update / Optimizer: SGD
      self.optimizer.step()

      # apply gradients to the neural network
      # grad_theta, l2_loss, grad_norm = nn_r.apply_grads(feat_map, grad_r)

    # get output
    self.networks.eval()
    rewards = self.networks(feat_map)
    rewards = rewards.view(49, 1)
    _, policy = value_iteration.value_iteration(P_a, rewards, gamma, error=0.01, deterministic=True, npy=False)
    mu_exp = compute_state_visition_freq(P_a, gamma, trajs, policy, deterministic=True)

    # return sigmoid(normalize(rewards))
    rewards = rewards.view(49, 1)
    rewards = rewards.data.numpy() # for normalize
    return rewards, mu_D, mu_exp

  def numpyToTensor(self, x):
    x = torch.from_numpy(x)
    x = Variable(x).float()

    return x




