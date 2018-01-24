import matplotlib.pyplot as plt
import random
import argparse
from collections import namedtuple

import img_utils
from mdp import gridworld
from mdp import value_iteration
from deep_maxent_irl_torch_9x9 import *
#from maxent_irl import *
from utils import *
#from lp_irl import *
import os
import time
import numpy as np
from numpy import linalg as LA


Step = namedtuple('Step', 'cur_state action next_state reward done')

PARSER = argparse.ArgumentParser(description=None)
#PARSER.add_argument('-hei', '--height', default=5, type=int, help='height of the gridworld')
#PARSER.add_argument('-wid', '--width', default=5, type=int, help='width of the gridworld')
PARSER.add_argument('-g', '--gamma', default=0.9, type=float, help='discount factor')
PARSER.add_argument('-a', '--act_random', default=0.3, type=float, help='probability of acting randomly')
#PARSER.add_argument('-t', '--n_trajs', default=128, type=int, help='number of expert trajectories')
#PARSER.add_argument('-l', '--l_traj', default=15, type=int, help='length of expert trajectory')
PARSER.add_argument('--rand_start', dest='rand_start', action='store_true', help='when sampling trajectories, randomly pick start positions')
PARSER.add_argument('--no-rand_start', dest='rand_start', action='store_false', help='when sampling trajectories, fix start positions')
PARSER.set_defaults(rand_start=False)
PARSER.add_argument('-lr', '--learning_rate', default=0.03, type=float, help='learning rate')
PARSER.add_argument('-mc', '--Montecarlo_trial', default=10, type=int, help='Montecarlo_trial')
#PARSER.add_argument('-ni', '--n_iters', default=100, type=int, help='number of iterations')
ARGS = PARSER.parse_args()
print ARGS

GAMMA = ARGS.gamma
ACT_RAND = ARGS.act_random

R_MAX = 1  # the constant r_max does not affect much the recoverred reward distribution
R_MIN = 0
#H = ARGS.height
#W = ARGS.width
#N_TRAJS = ARGS.n_trajs
#L_TRAJ = ARGS.l_traj
RAND_START = ARGS.rand_start
LEARNING_RATE = ARGS.learning_rate
#N_ITERS = ARGS.n_iters


def numpyToTensor(x):
    x = torch.from_numpy(x)
    x = Variable(x).float()
    return x

def generate_demonstrations(gw, policy, n_trajs=100, len_traj=20, rand_start=False, start_pos=[0, 0]):
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
    # rand_start = True
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

def generateRewardMapGT(W=9, H=9, R=0.5, R_MIN=0.0, R_MAX=1.0, coordGoal=(8, 8), nBlocks=15, mode='random'):
    # Initialize Variables
    if mode == 'random':
        #rmap_gt = np.zeros([H, W]) + R + (np.random.rand(H, W)-0.5)*0.2  # Uniform [R-0.2, R+0.2)
        rmap_gt = np.zeros([H, W]) + R + (np.random.rand(H, W) - 0.5) * 0.4  # Uniform [R-0.2, R+0.2)
    else:
        rmap_gt = np.zeros([H, W]) + R
    #rmap_gt[0, 0] = R_MIN
    N_STATES = W*H
    # Select Random Coordinates
    random_index = np.random.permutation(range(W+1, N_STATES-W-1))
    random_index = random_index[:nBlocks]
    # Set Blocks
    for idx in random_index:
        idxH = int(idx / W)
        idxW = idx - idxH*W
        rmap_gt[(idxH, idxW)] = R_MIN
    # Set the Goal
    rmap_gt[coordGoal] = R_MAX
    return rmap_gt, N_STATES




def draw_plot(data1, data2, edge_color, fill_color):
    bp1 = ax.boxplot(data1, patch_artist=True)
    bp2 = ax.boxplot(data2, patch_artist=True)

    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp1[element], color=edge_color[0])
        plt.setp(bp2[element], color=edge_color[1])

    for patch in bp1['boxes']:
        patch.set(facecolor=fill_color[0])

    for patch in bp2['boxes']:
        patch.set(facecolor=fill_color[1])

    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Baseline', 'Proposed'], loc='lower right')




def term_project(obstacle_type, trial, N_ITERS, N_TRAJS, L_TRAJ, H, W, r_weight, p_weight, dir):
    N_STATES = H * W
    N_ACTIONS = 5

    if obstacle_type == 3:
        if N_STATES == 25:
            block = 5
        elif N_STATES == 49:
            block = 10
        elif N_STATES == 81:
            block = 20

        rmap_gt, N_STATES = generateRewardMapGT(W=W, H=H, R=0.7, R_MIN=R_MIN, R_MAX=R_MAX, coordGoal=(H - H, W - 1), nBlocks=block, mode='constant')
        rmap_gt = generateObstacle(rmap_gt, H, W, R_MIN, R_MAX, obstacle_type)
    elif obstacle_type == 4:
        rmap_gt, N_STATES = generateRewardMapGT(W=W, H=H, R=0.7, R_MIN=R_MIN, R_MAX=R_MAX, coordGoal=(H - 1, W - 1), nBlocks=0, mode='random')
    else:
        rmap_gt = np.zeros([H, W]) + 0.7
        rmap_gt = generateObstacle(rmap_gt, H, W, R_MIN, R_MAX, obstacle_type)

    gw = gridworld.GridWorld(rmap_gt, {}, 1 - ACT_RAND)
    rewards_gt = np.reshape(rmap_gt, H * W, order='F')
    rewards_gt = numpyToTensor(rewards_gt)
    P_a = gw.get_transition_mat()

    values_gt, policy_gt = value_iteration.value_iteration(P_a, rewards_gt, GAMMA, N_STATES, error=0.01, deterministic=True, npy=False)

    # use identity matrix as feature
    feat_map = torch.eye(N_STATES)
    # feat_map = torch.rand(N_STATES, N_STATES)- 0.5

    trajs = generate_demonstrations(gw, policy_gt, n_trajs=N_TRAJS, len_traj=L_TRAJ, rand_start=RAND_START)

    #print 'Baseline(Deep Max Ent IRL) training ..'
    Trainer = trainer(W, H, feat_map, P_a, GAMMA, trajs, LEARNING_RATE, N_ITERS)
    rewards, mu_D, mu_exp = Trainer.deep_maxent_irl(feat_map, P_a, GAMMA, trajs, ARGS.learning_rate, N_ITERS, rewards_gt, policy_gt, N_STATES, H, W, r_weight, p_weight, proposed=0)
    #print 'Proposed method training ..'
    Trainer2 = trainer(W, H, feat_map, P_a, GAMMA, trajs, LEARNING_RATE, N_ITERS)
    rewards_1, mu_D_1, mu_exp_1 = Trainer2.deep_maxent_irl(feat_map, P_a, GAMMA, trajs, ARGS.learning_rate, N_ITERS, rewards_gt, policy_gt, N_STATES, H, W, r_weight, p_weight, proposed=1)
    Trainer3 = trainer(W, H, feat_map, P_a, GAMMA, trajs, LEARNING_RATE, N_ITERS)
    rewards_2, mu_D_2, mu_exp_2 = Trainer3.deep_maxent_irl(feat_map, P_a, GAMMA, trajs, ARGS.learning_rate, N_ITERS, rewards_gt, policy_gt, N_STATES, H, W, r_weight, p_weight, proposed=3)

    values, policy = value_iteration.value_iteration(P_a, rewards, GAMMA, N_STATES, error=0.01, deterministic=True, npy=True)
    values_1, policy_1 = value_iteration.value_iteration(P_a, rewards_1, GAMMA, N_STATES, error=0.01, deterministic=True, npy=True)
    values_2, policy_2 = value_iteration.value_iteration(P_a, rewards_2, GAMMA, N_STATES, error=0.01, deterministic=True, npy=True)

    rewards_gt = rewards_gt.data.numpy()

    #evd_base = compute_expected_value_difference(policy, rewards_gt, policy_gt, GAMMA, H, W)
    #evd_proposed = compute_expected_value_difference(policy_1, rewards_gt, policy_gt, GAMMA, H, W)

    #baseline = np.abs(np.reshape(rewards_gt, (H * W, 1)) - rewards)
    baseline = LA.norm(np.reshape(rewards_gt, (H * W, 1)) - rewards)
    #proposed_1 = np.abs(np.reshape(rewards_gt, (H * W, 1)) - rewards_1)
    proposed_1 = LA.norm(np.reshape(rewards_gt, (H * W, 1)) - rewards_1)
    #proposed_2 = np.abs(np.reshape(rewards_gt, (H * W, 1)) - rewards_2)
    proposed_2 = LA.norm(np.reshape(rewards_gt, (H * W, 1)) - rewards_2)

    #print 'Sum of reward difference(Baseline : {},'.format(np.sum(baseline)), 'Proposed : {})'.format(np.sum(proposed))
    #print '# of exceed 0.3(Baseline : {},'.format(len(np.extract(baseline > 0.3, baseline))), 'Proposed : {})'.format(len(np.extract(proposed > 0.3, proposed)))
    #print 'Expected Value Difference(Baseline : {},'.format(evd_base), 'Proposed : {})'.format(evd_proposed)

    # plots
    plt.figure(figsize=(20, 10))
    plt.subplot(2, 4, 1)
    img_utils.heatmap2d(np.reshape(rewards_gt, (H, W), order='F'), 'Rewards Map - Ground Truth', block=False)
    plt.subplot(2, 4, 2)
    img_utils.heatmap2d(np.reshape(rewards, (H, W), order='F'), 'Reward Map - Baseline', block=False)
    plt.subplot(2, 4, 3)
    img_utils.heatmap2d(np.reshape(rewards_1, (H, W), order='F'), 'Reward Map - Proposed_1', block=False)
    plt.subplot(2, 4, 4)
    img_utils.heatmap2d(np.reshape(rewards_2, (H, W), order='F'), 'Reward Map - Proposed_2', block=False)
    plt.subplot(2, 4, 5)
    img_utils.heatmap2d(np.reshape(values_gt.numpy(), (H, W), order='F'), 'Value Map - Ground Truth', block=False)
    plt.subplot(2, 4, 6)
    img_utils.heatmap2d(np.reshape(values.numpy(), (H, W), order='F'), 'Value Map - Baseline', block=False)
    plt.subplot(2, 4, 7)
    img_utils.heatmap2d(np.reshape(values_1.numpy(), (H, W), order='F'), 'Value Map - Proposed_1', block=False)
    plt.subplot(2, 4, 8)
    img_utils.heatmap2d(np.reshape(values_2.numpy(), (H, W), order='F'), 'Value Map - Proposed_2', block=False)

    file_name = 'reward{}.png'.format(trial)
    plt.savefig(dir + file_name)
    plt.close()
    #plt.savefig('result/type1/reward{}.png'.format(trial))

    plt.figure(figsize=(10, 10))
    plt.subplot(3, 2, 1)
    img_utils.heatmap2d(np.reshape(mu_D.numpy(), (H, W), order='F'), 'SVF - GT_Baseline', block=False)
    plt.subplot(3, 2, 2)
    img_utils.heatmap2d(np.reshape(mu_exp.numpy(), (H, W), order='F'), 'SVF - Baseline', block=False)
    plt.subplot(3, 2, 3)
    img_utils.heatmap2d(np.reshape(mu_D_1.numpy(), (H, W), order='F'), 'SVF - GT_Proposed_1', block=False)
    plt.subplot(3, 2, 4)
    img_utils.heatmap2d(np.reshape(mu_exp_1.numpy(), (H, W), order='F'), 'SVF - Proposed_!', block=False)
    plt.subplot(3, 2, 5)
    img_utils.heatmap2d(np.reshape(mu_D_2.numpy(), (H, W), order='F'), 'SVF - GT_Proposed_2', block=False)
    plt.subplot(3, 2, 6)
    img_utils.heatmap2d(np.reshape(mu_exp_2.numpy(), (H, W), order='F'), 'SVF - Proposed_2', block=False)

    file_name = 'svf{}.png'.format(trial)
    plt.savefig(dir + file_name)
    plt.close()
    #plt.savefig('result/type1/svf{}.png'.format(trial))

    return baseline**2, proposed_1**2 , proposed_2**2


if __name__ == "__main__":

    size = [7] #[5, 7, 9]
    r_weights = [1.75]
    p_weights = [0.4]

    control_v = p_weights

    total_mean = np.zeros([len(control_v), 3, 4])
    total_std = np.zeros([len(control_v), 3, 4])

    for idx in range(len(control_v)):

        # control values
        gw_size = 7# size[idx] # need to experiment : [5, 7, 9]
        reward_weight = 1.75
        p_weight = control_v[idx]

        MC_number = ARGS.Montecarlo_trial
        height = gw_size
        width = gw_size
        n_iters = 500
        n_trajs = 128
        l_traj = gw_size * 5

        baseline_result = np.zeros(4)
        baseline_result_std = np.zeros(4)
        proposed_1_result = np.zeros(4)
        proposed_1_result_std = np.zeros(4)
        proposed_2_result = np.zeros(4)
        proposed_2_result_std = np.zeros(4)
        total_seconds = 0
        for i in range(0, 1): # Number of obstacle type
            monte_seconds = 0
            # obs_type stand for type of obstacles in reward map
            # 1 : near start point(familiar one)
            # 2 : near goal
            # 3 : type 1 + random obstacle
            # 4 : No obstacle, random reward
            obs_type = i+1

            script_dir = os.path.dirname('/home/choikoal/IRL_optimalpath_final/500iter/')
            results_dir = os.path.join(script_dir, 'control{}'.format(control_v[idx]), 'type{}/'.format(obs_type))

            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)

            baseline_trial = np.zeros(MC_number)
            proposed_1_trial = np.zeros(MC_number)
            proposed_2_trial = np.zeros(MC_number)
            print '[[Option #{}'.format(obs_type), 'reward weight : {}'.format(reward_weight-p_weight), 'previous weight : {}'.format(p_weight),'height : {}'.format(height), 'width : {}'.format(
                width), 'iter : {}'.format(n_iters), 'trajs_# : {}'.format(n_trajs), 'traj_length : {}]]'.format(
                l_traj)
            for k in range(0, MC_number):
                #print '[MC trial #{}]'.format(k+1)
                instant_init_time = time.time()
                baseline_trial[k], proposed_1_trial[k], proposed_2_trial[k] = term_project(obs_type, k, n_iters, n_trajs, l_traj, height, width, reward_weight, p_weight, results_dir)
                instant_end_time = time.time()

                instant_time = instant_end_time - instant_init_time
                monte_seconds += instant_time
                total_seconds += instant_time

                m_i, instant_time = int(instant_time // 60), instant_time % 60
                s_i = instant_time

                print '[MC #{:3d}(base/proposed_1/ proposed_2)]'.format(k+1), '{} /'.format(baseline_trial[k]), '{} /'.format(proposed_1_trial[k]),'{} /'.format(proposed_2_trial[k]), 'time : {:d}m'.format(m_i), '{0:.3f}s'.format(s_i)

            baseline_result[i] = np.mean(baseline_trial)
            baseline_result_std[i] = np.std(baseline_trial)
            proposed_1_result[i] = np.mean(proposed_1_trial)
            proposed_1_result_std[i] = np.std(proposed_1_trial)
            proposed_2_result[i] = np.mean(proposed_2_trial)
            proposed_2_result_std[i] = np.std(proposed_2_trial)

            plt.figure()
            a = np.arange(MC_number) + 1
            plt.plot(a, baseline_trial, label = 'Baseline')
            plt.plot(a, proposed_1_trial, label = 'Proposed_1')
            plt.plot(a, proposed_2_trial, label='Proposed_2')
            plt.xticks(range(1,MC_number+1))
            plt.xlabel('MC trial')
            plt.ylabel('Sum of reward difference')
            plt.legend(loc = 'best')
            plt.grid(True)
            file_name = 'result.png'
            plt.savefig(results_dir + file_name)
            plt.close()

            print 'Averate result(Baseline/Proposed_1/Proposed_2) : {} /'.format(baseline_result[i]), '{} /'.format(proposed_1_result[i]), '{}'.format(proposed_2_result[i])

            h_m, monte_seconds = int(monte_seconds // 3600), monte_seconds % 3600
            m_m, monte_seconds = int(monte_seconds // 60), monte_seconds % 60
            s_m = monte_seconds

            print 'Processing time : {}h'.format(h_m), '{}m'.format(m_m), '{0:.3f}s'.format(s_m), '\n'


        plt.figure()
        x = np.array([1, 2, 3, 4])
        plt.errorbar(x, baseline_result, baseline_result_std, linestyle='None', linewidth =2, marker='o', markerfacecolor = 'blue', markersize = 5, label = 'Baseline', color = 'blue')
        plt.errorbar(x, proposed_1_result, proposed_1_result_std, linestyle='None', linewidth =2, marker='o', markerfacecolor = 'green', markersize = 5, label = 'Proposed_1', color= 'green')
        plt.errorbar(x, proposed_2_result, proposed_2_result_std, linestyle='None', linewidth=2, marker='o', markerfacecolor='red', markersize=5, label='Proposed_2', color='red')
        #plt.plot(x, baseline_result, label = 'Baseline', color = 'blue')
        #plt.plot(x, proposed_result, label = 'Proposed', color = 'green')
        #plt.fill_between(x, baseline_result + baseline_result_std, baseline_result - baseline_result_std, facecolor = 'blue', alpha = 0.3)
        #plt.fill_between(x, proposed_result + proposed_result_std, proposed_result - proposed_result_std, facecolor = 'green', alpha = 0.3)
        plt.xticks([0, 1, 2, 3, 4, 5])
        plt.xlabel('Obstacle option')
        plt.ylabel('Sum of reward difference')
        plt.legend(loc='best')
        plt.grid(True)

        file_name = 'result_{}.png'.format(gw_size)
        plt.savefig(os.path.join(script_dir, 'control{}'.format(control_v[idx])) + file_name)
        plt.close()

        print 'Baseline   : ', baseline_result
        print 'Proposed_1 : ', proposed_1_result
        print 'Proposed_2 : ', proposed_2_result

        h_t, total_seconds = int(total_seconds // 3600), total_seconds % 3600
        m_t, total_seconds = int(total_seconds // 60), total_seconds % 60
        s_t = total_seconds

        total_mean[idx, 0, :] = baseline_result
        total_mean[idx, 1, :] = proposed_1_result
        total_mean[idx, 2, :] = proposed_2_result
        total_std[idx, 0, :] = baseline_result_std
        total_std[idx, 1, :] = proposed_1_result_std
        total_std[idx, 2, :] = proposed_2_result_std

        print 'done, {}h'.format(h_t), '{}m'.format(m_t), '{0:.3f}s'.format(s_t), '\n'

    print 'total_result\n', total_mean
    print 'total_std\n', total_std

    plt.figure()
    x = np.array([1, 2, 3, 4])

    #for idx2 in range(len(control_v)):
    #    plt.errorbar(x, total_baseline[idx2], total_baseline_std[idx2], linestyle='None', linewidth =2, marker='o', markerfacecolor='blue', markersize=5, label='Baseline{}'.format(control_v[idx2]))
    #    plt.errorbar(x, total_propsed_1[idx2], total_propsed_1_std[idx2], linestyle='None', linewidth =2, marker='D', markerfacecolor='green', markersize=5, label='Proposed_1{}'.format(control_v[idx2]))
    #    plt.errorbar(x, total_propsed_2[idx2], total_propsed_2_std[idx2], linestyle='None', linewidth=2, marker='s', markerfacecolor='red', markersize=5, label='Proposed_2{}'.format(control_v[idx2]))

    plt.xticks([0, 1, 2, 3, 4, 5])
    plt.xlabel('Obstacle option')
    plt.ylabel('Sum of reward difference')
    plt.legend(loc='best')
    plt.grid(True)

    plt.savefig('result/total_result.png')

    plt.show()

    '''
    # box plot example
    baseline_data = [baseline_trial[0], baseline_trial[1], baseline_trial[2], baseline_trial[3]]
    proposed_data = [proposed_trial[0], proposed_trial[1], proposed_trial[2], proposed_trial[3]]

    # example_data1 = [[1,2,0.8], [0.5,2,2], [3,2,1]]
    # example_data2 = [[5,3, 4], [6,4,3,8], [6,4,9]]

    fig, ax = plt.subplots()
    edgecolor = ['red', 'blue']
    boxColors = ['tan', 'cyan']
    draw_plot(baseline_data, proposed_data, edgecolor, boxColors)
    ax.set_xlabel('Obstacle option')
    ax.set_ylabel('Sum of reward difference')
    ax.grid(True)
    # ax.set_ylim(0, 10)
    '''


