#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on July 2018

@author: nicola ancona
"""

import tensorflow as tf
import numpy as np
import math
import csv
import os.path
import models
from ReplayBuffer import ReplayBuffer
from actorNetwork import Actor
from criticNetwork import Critic
import restore_model
from net import net

# ==========================
#   Training Parameters
# ==========================
# Max training steps
MAX_EPISODES = 40000
# Max episode length
MAX_EPISODE_LENGTH = 1010
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Discount factor
GAMMA = 0.99
# Soft target update param
TAU = 0.001
# ===========================
#   Utility Parameters
# ===========================
RANDOM_SEED = 1234
# Size of replay buffer
TRAINING_SIZE = 2000
BUFFER_SIZE = 300000
MINIBATCH_SIZE = 64
MIN_BUFFER_SIZE = 1000
# Environment Parameters
ACTION_DIMENSION = 6
ACTION_DIMENSION_GRL = 9
STATE_DIMS = 18
ACTION_BOUND = 1
ACTION_BOUND_REAL = 8.6
# Noise Parameters
NOISE_MEAN = 0
NOISE_VAR = 1
# Ornstein-Uhlenbeck variables
OU_THETA = 0.15
OU_MU = 0
OU_SIGMA = 0.2
# Restore Session
INPUT_DIM = 24
OUTPUT_DIM = 18
KEEP_PROB = 1.0
model_path = '/home/nancona/PycharmProjects/NN_LEO/results_model_nn/20k_100/29_10_2018_10_7/model_restore/model_20k_100.ckpt'
# Test flag
TEST = False


def compute_ou_noise(noise):
    # Solve using Euler-Maruyama method
    noise = noise + OU_THETA * (OU_MU - noise) + OU_SIGMA * np.random.randn(ACTION_DIMENSION)
    return noise


def compute_action(actor, s, noise):
    if TEST:
        action = actor.predict(np.reshape(s, (1, STATE_DIMS)))
    else:
        action = actor.predict(np.reshape(s, (1, STATE_DIMS))) + compute_ou_noise(noise)
    action = np.reshape(action, (ACTION_DIMENSION,))
    action = np.clip(action, -1, 1)
    # action = action*ACTION_BOUND_REAL
    return action


def write_csv_learn(episode, steps, s0, a, s2, t, r, tr):

    file_exists = os.path.isfile('learn.csv')
    file_exists_gen = os.path.isfile('gen_learn.csv')
    a = np.ndarray.tolist(a)
    episode = [episode]
    steps = [steps]
    r = [r]
    terminal = [t]
    tr = [tr]

    x1 = episode + steps + [s2[0]] + tr
    x = s0 + a + s2 + r + terminal + tr

    if t:
        with open('gen_learn.csv', 'a') as csvfile1:
            headers = ['Episode', 'Steps', 'Falling Position', 'Total Reward']
            wr = csv.writer(csvfile1, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

            if not file_exists_gen:
                wr.writerow(headers)  # file doesn't exist yet, write a header

            wr.writerow(x1)

    with open('learn.csv', 'a') as csvfile2:
        headers = ['trsxp', 'trszp', 'trsa', 'lha', 'rha', 'lka', 'rka', 'laa', 'raa', 'a1', 'a2', 'a3', 'a4', 'a5',
                   'a6', 'trsxp+1', 'trszp+1', 'trsa+1', 'lha+1', 'rha+1', 'lka+1', 'rka+1', 'laa+1', 'raa+1',
                   'reward', 'terminal', 'tot_reward']
        # headers = ['trsxp', 'trsyp', 'trsa', 'lha', 'rha', 'lka', 'rka', 'laa', 'raa', 'trsxv', 'trzv', 'trso', 'lho',
        #           'rho', 'lko', 'rko', 'lao', 'rao', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'trsxp+1', 'trsyp+1',
        #           'trsa+1', 'lha+1', 'rha+1', 'lka+1', 'rka+1', 'laa+1', 'raa+1', 'trsxv+1', 'trzv+1', 'trso+1',
        #           'lho+1', 'rho+1', 'lko+1', 'rko+1', 'lao+1', 'rao+1', 'reward', 'terminal', 'tot_reward']
        wr = csv.writer(csvfile2, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        if not file_exists:
            wr.writerow(headers)  # file doesn't exist yet, write a header

        wr.writerow(x)


def write_csv_test(episode, steps, s0, a, s2, t, r, tr):
    file_exists = os.path.isfile('test.csv')
    file_exists_gen = os.path.isfile('gen_test.csv')
    a = np.ndarray.tolist(a)
    episode = [episode]
    steps = [steps]
    r = [r]
    terminal = [t]
    tr = [tr]

    x1 = episode + steps + [s2[0]] + tr
    x = s0 + a + s2 + r + terminal + tr

    if t:
        with open('gen_test.csv', 'a') as csvfile3:
            headers = ['Episode', 'Steps', 'Falling Position', 'Total Reward']
            wr = csv.writer(csvfile3, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

            if not file_exists_gen:
                wr.writerow(headers)  # file doesn't exist yet, write a header

            wr.writerow(x1)

    with open('test.csv', 'a') as csvfile4:
        headers = ['trsxp', 'trsyp', 'trsa', 'lha', 'rha', 'lka', 'rka', 'laa', 'raa', 'trsxv', 'trzv', 'trso', 'lho',
                   'rho', 'lko', 'rko', 'lao', 'rao', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'trsxp+1', 'trsyp+1',
                   'trsa+1', 'lha+1', 'rha+1', 'lka+1', 'rka+1', 'laa+1', 'raa+1', 'trsxv+1', 'trzv+1', 'trso+1',
                   'lho+1', 'rho+1', 'lko+1', 'rko+1', 'lao+1', 'rao+1', 'reward', 'terminal', 'tot_reward']
        wr = csv.writer(csvfile4, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        if not file_exists:
            wr.writerow(headers)  # file doesn't exist yet, write a header

        wr.writerow(x)


def write_csv_animation_train(time, s0):
    with open('animation_train.csv', 'a') as csvfile5:
        time = [time]
        row = time + s0
        line = ', '.join(["%s" % k for k in row])
        csvfile5.write(line + '\n')


def write_csv_animation_test(time, s0):
    with open('animation_test.csv', 'a') as csvfile6:
        time = [time]
        row = time + s0
        line = ', '.join(["%s" % k for k in row])
        csvfile6.write(line + '\n')


def train(sess, actor, critic, nn, test, train_flag=False):
    t = 0  # test counter

    time = 0
    step = 0.03

    sess.run(tf.global_variables_initializer())

    # initialize actor, critic and replay buffer, and initial state
    actor.update_target_network()
    critic.update_target_network()
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

    # the initial state has to be change also in the init function below
    s2 = [0, 0, -0.101485,
          0.100951, 0.819996, -0.00146549,
          -1.27, 4.11e-6, 4.11e-6,
          0, 0, 0,
          0, 0, 0,
          0, 0, 0]

    # print s.current_state()

    for i in range(MAX_EPISODES):

        if test and not i % 20 and i > 0 and replay_buffer.size() > MIN_BUFFER_SIZE:
            TEST = True
            t += 1
        else:
            TEST = False
        # initialize noise process
        noise = np.zeros(ACTION_DIMENSION)
        total_episode_reward = 0

        for j in range(MAX_EPISODE_LENGTH):
            s0 = s2
            a = compute_action(actor, s0, noise)
            model_input = np.hstack([s0, a])
            s2 = nn.eval({input: model_input, keep_prob: 1})
            # # ======================================
            # # COMPUTING ACIONS S2, TERMINAL, REWARD.
            # # ======================================
            # # checking if single or double support phase
            # # computing first actuated variable
            # # computing hip and knee difference
            # delta_left_hip = s_av[0] - s0[3]
            # delta_right_hip = s_av[1] - s0[4]
            # delta_left_knee = s_av[2] - s0[5]
            # delta_right_knee = s_av[3] - s0[6]
            # # then computing torso
            # s_trs = s.next_ta(s0, delta_left_hip, delta_right_hip, eng)
            # # computing torso difference
            # delta_torso = 0  # s_trs - s0[2]
            # # and non-actuated variables
            # x = s.computes_noactuated_variables(s0, delta_torso, delta_left_hip, delta_right_hip, delta_left_knee,
            #                                 delta_right_knee, eng, phase)

            r = models.calc_reward(s2, s0)
            # print phase, s.current_state()
            terminal = models.calc_terminal()

            # s2_buffer = [s2[0], s2[3], s2[4], s2[5], s2[6]]
            if not TEST:
                replay_buffer.add(np.reshape(s0, (actor.s_dim,)), np.reshape(a, actor.a_dim), r,
                                  terminal, np.reshape(s2, (actor.s_dim,)))

            total_episode_reward += r

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > MIN_BUFFER_SIZE:
                if not TEST:
                    train_flag = True
                    s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(MINIBATCH_SIZE)

                    # calculate targets
                    target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))

                    y_i = []
                    for k in range(MINIBATCH_SIZE):
                        if t_batch[k]:
                            y_i.append(r_batch[k])
                        else:
                            y_i.append(r_batch[k] + GAMMA * target_q[k])

                    # Update the critic given the targets
                    predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))

                    # ep_ave_max_q += np.amax(predicted_q_value)

                    # Update the actor policy using the sampled gradient
                    a_outs = actor.predict(s_batch)
                    grads = critic.action_gradients(s_batch, a_outs)
                    actor.train(s_batch, grads[0])

                    # Update target networks
                    actor.update_target_network()
                    critic.update_target_network()

                    # csv for animation
                if not TEST:
                    write_csv_animation_train(time, s0[0:9])
                    time = time + step
                    if terminal == 2:
                        time = 0
                else:
                    write_csv_animation_test(time, s0[0:9])
                    time = time + step
                    if terminal == 2:
                        time = 0

                if not TEST:
                    write_csv_learn(i-t, j, s0, a, s2, terminal, r, total_episode_reward)

                else:
                    write_csv_test(t, j, s0, a, s2, terminal, r, total_episode_reward)

            if not terminal == 0:
                print train_flag, t, i-t, j, total_episode_reward  # printing n of test, n of train, length of the episode,
                                                                 # tot ep reward
                break

        s2 = models.reset()


def main():
    keep_prob = tf.placeholder("float")
    input = tf.placeholder("float", [None, INPUT_DIM])
    output = tf.placeholder("float", [None, OUTPUT_DIM])
    nn = net(x=input, keep_prob=keep_prob)
    with tf.Session() as sess:
        actor = Actor(sess, STATE_DIMS, ACTION_DIMENSION, 1, ACTOR_LEARNING_RATE, TAU)
        critic = Critic(sess, STATE_DIMS, ACTION_DIMENSION, CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars())
        train(sess, actor, critic, nn, test=True)


if __name__ == "__main__":
    main()
