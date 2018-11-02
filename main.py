#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on July 2018

@author: nicola ancona
"""

import tensorflow as tf
import numpy as np
import csv
import os.path
import models
from ReplayBuffer import ReplayBuffer
from models import Models
import net
from actorNetwork import Actor
from criticNetwork import Critic
import datetime



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
NOISE_VAR = 1.0
# Ornstein-Uhlenbeck variables
OU_THETA = 0.30 #0.15
OU_MU = 0
OU_SIGMA = 0.40 #0.2
# Restore Session
INPUT_DIM = 24
OUTPUT_DIM = 18
KEEP_PROB = 1.0
model_path = '/home/nancona/PycharmProjects/NN_LEO/results_model_nn/20k_100/29_10_2018_10_7/model_restore/model_20k_100.ckpt'
# RL saving folder
rl_results_path = '/home/nancona/PycharmProjects/NN_LEO/rl_results'
now = datetime.datetime.now()
date_folder = '%s_%s_%s_%s_%s_%s' % (now.day, now.month, now.year, now.hour, now.minute, now.second)
rl_results_folder = os.path.join(rl_results_path, date_folder)
os.makedirs(rl_results_folder)
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

    learn_path = os.path.join(rl_results_folder, 'learn.csv')
    file_exists = os.path.isfile(learn_path)
    gen_learn_path = os.path.join(rl_results_folder, 'gen_learn.csv')
    file_exists_gen = os.path.isfile(gen_learn_path)
    a = np.ndarray.tolist(a)
    episode = [episode]
    s2 = np.ndarray.tolist(s2)
    steps = [steps]
    r = [r]
    terminal = [t]
    tr = [tr]
    if not type(s0) == list:
        s0 = np.ndarray.tolist(s0)

    x1 = episode + steps + [s2[0]] + tr
    x = s0 + a + s2 + r + terminal + tr

    if t:
        with open(gen_learn_path, 'a') as csvfile1:
            headers = ['Episode', 'Steps', 'Falling Position', 'Total Reward']
            wr = csv.writer(csvfile1, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

            if not file_exists_gen:
                wr.writerow(headers)  # file doesn't exist yet, write a header

            wr.writerow(x1)

    with open(learn_path, 'a') as csvfile2:
        # headers = ['trsxp', 'trszp', 'trsa', 'lha', 'rha', 'lka', 'rka', 'laa', 'raa', 'a1', 'a2', 'a3', 'a4', 'a5',
        #            'a6', 'trsxp+1', 'trszp+1', 'trsa+1', 'lha+1', 'rha+1', 'lka+1', 'rka+1', 'laa+1', 'raa+1',
        #            'reward', 'terminal', 'tot_reward']
        headers = ['trsxp', 'trsyp', 'trsa', 'lha', 'rha', 'lka', 'rka', 'laa', 'raa', 'trsxv', 'trzv', 'trso', 'lho',
                  'rho', 'lko', 'rko', 'lao', 'rao', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'trsxp+1', 'trsyp+1',
                  'trsa+1', 'lha+1', 'rha+1', 'lka+1', 'rka+1', 'laa+1', 'raa+1', 'trsxv+1', 'trzv+1', 'trso+1',
                  'lho+1', 'rho+1', 'lko+1', 'rko+1', 'lao+1', 'rao+1', 'reward', 'terminal', 'tot_reward']
        wr = csv.writer(csvfile2, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        if not file_exists:
            wr.writerow(headers)  # file doesn't exist yet, write a header

        wr.writerow(x)


def write_csv_test(episode, steps, s0, a, s2, t, r, tr):
    test_path = os.path.join(rl_results_folder, 'test.csv')
    file_exists = os.path.isfile(test_path)
    gen_test_path = os.path.join(rl_results_folder, 'gen_test.csv')
    file_exists_gen = os.path.isfile(gen_test_path)
    a = np.ndarray.tolist(a)
    s2 = np.ndarray.tolist(s2)
    episode = [episode]
    steps = [steps]
    r = [r]
    terminal = [t]
    tr = [tr]
    if not type(s0) == list:
        s0 = np.ndarray.tolist(s0)

    x1 = episode + steps + [s2[0]] + tr
    x = s0 + a + s2 + r + terminal + tr

    if t:
        with open(gen_test_path, 'a') as csvfile3:
            headers = ['Episode', 'Steps', 'Falling Position', 'Total Reward']
            wr = csv.writer(csvfile3, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

            if not file_exists_gen:
                wr.writerow(headers)  # file doesn't exist yet, write a header

            wr.writerow(x1)

    with open(test_path, 'a') as csvfile4:
        headers = ['trsxp', 'trsyp', 'trsa', 'lha', 'rha', 'lka', 'rka', 'laa', 'raa', 'trsxv', 'trzv', 'trso', 'lho',
                   'rho', 'lko', 'rko', 'lao', 'rao', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'trsxp+1', 'trsyp+1',
                   'trsa+1', 'lha+1', 'rha+1', 'lka+1', 'rka+1', 'laa+1', 'raa+1', 'trsxv+1', 'trzv+1', 'trso+1',
                   'lho+1', 'rho+1', 'lko+1', 'rko+1', 'lao+1', 'rao+1', 'reward', 'terminal', 'tot_reward']
        wr = csv.writer(csvfile4, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        if not file_exists:
            wr.writerow(headers)  # file doesn't exist yet, write a header

        wr.writerow(x)


def write_csv_animation_train(time, s0):
    animation_train_path = os.path.join(rl_results_folder, 'animation_train.csv')
    with open(animation_train_path, 'a') as csvfile5:
        time = [time]
        row = time + s0
        line = ', '.join(["%s" % k for k in row])
        csvfile5.write(line + '\n')


def write_csv_animation_test(time, s0):
    animation_test_path = os.path.join(rl_results_folder, 'animation_test.csv')
    with open(animation_test_path, 'a') as csvfile6:
        time = [time]
        row = time + s0
        line = ', '.join(["%s" % k for k in row])
        csvfile6.write(line + '\n')


def train(sess_2, actor, critic, mod, test, train_flag=False):
    t = 0  # test counter

    time = 0
    step = 0.03

    sess_2.run(tf.global_variables_initializer())

    # initialize actor, critic and replay buffer, and initial state
    actor.update_target_network()
    critic.update_target_network()
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

    # the initial state has to be change also in the init function below
    s = Models()

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
            s0 = s.current_state()
            a = compute_action(actor, s0, noise)
            model_input = (np.hstack([s0, a])).reshape(1, 24)
            s2 = mod.prediction(measured_input=model_input)
            s.import_state(s2[0])
            r = s.calc_reward(s2[0], s0)
            # print phase, s.current_state()
            terminal = s.calc_terminal(s2)

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
                    write_csv_learn(i-t, j, s0, a, s2[0], terminal, r, total_episode_reward)

                else:
                    write_csv_test(t, j, s0, a, s2[0], terminal, r, total_episode_reward)

            if not terminal == 0:
                print train_flag, t, i-t, j, total_episode_reward  # printing n of test, n of train, length of the episode,
                                                                 # tot ep reward
                break

        s.reset()


def main():

    # sample_size = 8000
    #
    # # Sample vector initialization
    # position_train = []
    # velocity_train = []
    # action_train = []
    # next_position_train = []
    # next_velocity_train = []
    #
    # position_test = []
    # velocity_test = []
    # action_test = []
    # next_position_test = []
    # next_velocity_test = []
    #
    # input_file = pd.read_csv("rbdl_leo2606_Animation-learn-0.csv")
    # input_file = input_file.values
    #
    # for i in range(sample_size):
    #     if i == 0:
    #         position_train = input_file[i][1:10]
    #         velocity_train = input_file[i][10:19]
    #         action_train = input_file[i][50:56]
    #         next_position_train = input_file[i][29:38]
    #         next_velocity_train = input_file[i][38:47]
    #
    #     if i > 0 and i < 5000:
    #         position_train = np.vstack([position_train, input_file[i][1:10]])
    #         velocity_train = np.vstack([velocity_train, input_file[i][10:19]])
    #         action_train = np.vstack([action_train, input_file[i][50:56]])
    #         next_position_train = np.vstack([next_position_train, input_file[i][29:38]])
    #         next_velocity_train = np.vstack([next_velocity_train, input_file[i][38:47]])
    #
    #     if i == 7000:
    #         position_test = input_file[i][1:10]
    #         velocity_test = input_file[i][10:19]
    #         action_test = input_file[i][50:56]
    #         next_position_test = input_file[i][29:38]
    #         next_velocity_test = input_file[i][38:47]
    #
    #     if i > 7000:
    #         position_test = np.vstack([position_test, input_file[i][1:10]])
    #         velocity_test = np.vstack([velocity_test, input_file[i][10:19]])
    #         action_test = np.vstack([action_test, input_file[i][50:56]])
    #         next_position_test = np.vstack([next_position_test, input_file[i][29:38]])
    #         next_velocity_test = np.vstack([next_velocity_test, input_file[i][38:47]])
    #
    # # Train samples vector
    # train_input = np.hstack([position_train, velocity_train, action_train])
    # train_output = np.hstack([next_position_train, next_velocity_train])
    # # Test samples vector
    # state_input = np.hstack([position_test, velocity_test, action_test])
    # test_output = np.hstack([next_position_test, next_velocity_test])

    tf.reset_default_graph()
    g_1 = tf.Graph()
    with g_1.as_default():
        sess_1 = tf.Session(graph=g_1)
        mod = net.leo_nn(sess_1)
        mod.restore() # model.eval(measured_input=np.zeros(24))
        # prediction = mod.prediction(state_input)
        # print prediction
    g_2 = tf.Graph()
    with g_2.as_default():
        sess_2 = tf.Session(graph=g_2)
    # with tf.Session() as sess:
    #     tf.reset_default_graph()
        actor = Actor(sess_2, STATE_DIMS, ACTION_DIMENSION, 1, ACTOR_LEARNING_RATE, TAU)
        critic = Critic(sess_2, STATE_DIMS, ACTION_DIMENSION, CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars())
        train(sess_2, actor, critic, mod, test=True)


if __name__ == "__main__":
    main()
