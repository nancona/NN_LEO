#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on July 2018

@author: nancona
"""

import math as m


def reset():
    reset_state = [0, 0, -0.101485,
                   0.100951, 0.819996, -0.00146549,
                   -1.27, 4.11e-6, 2.26e-7,
                   0, 0, 0,
                   0, 0, 0,
                   0, 0, 0]
    return reset_state

# def DoomedToFall_TorsoConstaint(self):
#    torsoConstraint = 1
#    if m.fabs(self.current_state()[2]) > torsoConstraint:
#        return True
#    return False

def DoomedToFall_Stance_TorsoHeight(current_state):
    torsoConstraint = 1
    stanceConstraint = 0.36*m.pi
    torsoHeightConstraint = -0.15
    if m.fabs(current_state()[2]) > torsoConstraint or m.fabs(current_state()[7] > stanceConstraint) or \
            m.fabs(current_state()[8]) > stanceConstraint or current_state()[5] > 0 or \
            current_state()[6] > 0 or current_state()[1] < torsoHeightConstraint:
        return True
    return False

def calc_reward(next_state, current_state):
    reward = 0
    RwDoomedToFall_Stance_TorsoHeight = -75
    RwDoomedToFall_TorsoConstaint = -75
    RwTime = -1.5
    RwForward = 300
    reward = RwTime
    reward += RwForward*(next_state[0] - current_state[0])
    if DoomedToFall_Stance_TorsoHeight(current_state):
        reward += RwDoomedToFall_Stance_TorsoHeight
    # if self.DoomedToFall_TorsoConstaint():
    #    self.reward += RwDoomedToFall_TorsoConstaint
    return reward

def calc_terminal(current_state):
    if DoomedToFall_Stance_TorsoHeight(current_state):  # or self.DoomedToFall_TorsoConstaint():
        terminal = 2
    else:
        terminal = 0
    return terminal

