# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 00:04:57 2016

@author: jdouma
"""

import numpy as np
from robot import Distribution

def careful_log(x):
    # computes the natural log of a non-negative real number
    if x == 0:
        return -np.inf
    else:
        return np.log(x)
        
def pYgivenX(x, y):
    """
    Compute the probability that x occurs given that y occurs
    Inputs:
        x - hidden state 'Fair' or 'Biased'
        y - observed state 'H' or 'T'
        
    Output: P(y|x)
    """
    
    prob = 1
    if y == 'T':
        if x ==  'Fair':
            prob = .5
        elif x == 'Biased':
            prob = .75
    elif y == 'H':
        if x ==  'Fair':
            prob = .5
        elif x == 'Biased':
            prob = .25
            
    return prob
    

def pX2givenX1(x1, x2):
    """
    Compute the probability that x2 occurs given that x1 occurs
    Inputs:
        x1 - either 'Fair' or 'Biased'
        x2 - either 'Fair' or 'Biased'
        
    Output: P(x2 | x1)
    """
    
    if x1 == x2:
        return .75
        
    return .25
    
def initial_distribution():
    distro = Distribution()
    distro['Fair'] = .5
    distro['Biased'] = .5
    
    return distro
    
def message(x, phi, psi, prior):
    """
    Calculate minimum over inputs of -log(phi(input)) -log(psi(input, x)) + prior[x]
    Inputs:
        x - destination value of message
        phi - non-negative function of input
        psi - non-negative function of input and x
        prior - Distribution that takes x as a key
    """
    
    minval = np.inf
    minarg = None
    for state in phi:
        val = -careful_log(phi[state])
        if pis != None:
            val -= careful_log(psi(state, x))
        if prior != None:
            val += prior[x]
        if val < minval:
            minval = val
            minarg = state
            
    return (minarg, minval)
    
    
def Viterbi(observations, hidden_states, phi, psi):
    """
    Inputs:
        observations: a list of observations, one per hidden state
        hidden_states: a list of all of the possible hidden states
        phi: function phi(x,y) = P(y|x)
        psi: function psi(current, next) = P(next|current)

    Output:
        A list of esimated hidden states
    """
    
    num_observations = len(observations)
    minima = [None] * (num_observations-1)
    args = [None] * (num_observations-1)
    
    for i in range(num_observations-1):
        minima[i] = Distribution()
        args[i] = Distribution()
        for x2 in hidden_states:
            minvalue = np.inf
            minarg = None
            for x1 in hidden_states:
                val = -careful_log(phi(x1, observations[i]))
                if i > 0:
                    val += minima[i-1][x1]
                if i < num_observations - 1:
                    val -= careful_log(psi(x1, x2))
                if val < minvalue:
                    minvalue = val
                    minarg = x1
            minima[i][x2] = minvalue
            args[i][x2] = minarg
            
    # find minimum at the root
          
    minvalue = np.inf
    for state in hidden_states:
        if minima[num_observations-2][state] -careful_log(phi(state, observations[num_observations-1])) < minvalue:
            minarg = state
    return_states = [None] * num_observations
    return_states[num_observations-1] = minarg        
    for i in range(num_observations-2, -1, -1):
        return_states[i] = args[i][return_states[i+1]]
        
    return return_states
