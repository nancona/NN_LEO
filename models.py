#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on July 2018

@author: nancona
"""

import math as m
import numpy as np
from net import leo_nn

class Models(object):

    def __init__(self):

        self.reward = 0
        self.terminal = 0
        self.initialize = self.reset()

    def reset(self):
        self.state_value = [0, 0, -0.101485,
                       0.100951, 0.819996, -0.00146549,
                       -1.27, 4.11e-6, 2.26e-7,
                       0, 0, 0,
                       0, 0, 0,
                       0, 0, 0]
        return self.state_value

    def import_state(self, x):
        self.state_value = x

    def current_state(self):
        return self.state_value

    # def DoomedToFall_TorsoConstaint(self):
    #    torsoConstraint = 1
    #    if m.fabs(self.current_state()[2]) > torsoConstraint:
    #        return True
    #    return False

    def DoomedToFall_Stance_TorsoHeight(self, current_state):
        torsoConstraint = 1
        stanceConstraint = 0.36*m.pi
        torsoHeightConstraint = -0.15
        if m.fabs(self.state_value[2]) > torsoConstraint or m.fabs(self.state_value[7] > stanceConstraint) or \
                m.fabs(self.state_value[8]) > stanceConstraint or self.state_value[5] > 0 or \
                self.state_value[6] > 0 or self.state_value[1] < torsoHeightConstraint:
            return True
        return False

    def calc_reward(self, next_state, current_state):
        reward = 0
        RwDoomedToFall_Stance_TorsoHeight = -75
        RwDoomedToFall_TorsoConstaint = -75
        RwTime = -1.5
        RwForward = 150
        reward = RwTime
        reward += RwForward*(next_state[0] - current_state[0])
        if self.DoomedToFall_Stance_TorsoHeight(current_state):
            reward += RwDoomedToFall_Stance_TorsoHeight
        # if self.DoomedToFall_TorsoConstaint():
        #    self.reward += RwDoomedToFall_TorsoConstaint
        return reward

    def calc_terminal(self, current_state):
        if self.DoomedToFall_Stance_TorsoHeight(current_state):  # or self.DoomedToFall_TorsoConstaint():
            terminal = 2
        else:
            terminal = 0
        return terminal

