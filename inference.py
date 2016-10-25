#!/usr/bin/env python
# inference.py
# Base code by George H. Chen (georgehc@mit.edu) -- updated 10/18/2016
import collections
import sys

import graphics
import numpy as np
import robot


# Throughout the code, we use these variables.
# Do NOT change these (but you'll need to use them!):
# - all_possible_hidden_states: a list of possible hidden states
# - all_possible_observed_states: a list of possible observed states
# - prior_distribution: a distribution over states
# - transition_model: a function that takes a hidden state and returns a
#     Distribution for the next state
# - observation_model: a function that takes a hidden state and returns a
#     Distribution for the observation from that hidden state
all_possible_hidden_states = robot.get_all_hidden_states()
all_possible_observed_states = robot.get_all_observed_states()
prior_distribution = robot.initial_distribution()
transition_model = robot.transition_model
observation_model = robot.observation_model


# You may find this function helpful for computing logs without yielding a
# NumPy warning when taking the log of 0.
def careful_log(x):
    # computes the log of a non-negative real number
    if x == 0:
        return -np.inf
    else:
        return np.log(x)
        
def final_distribution():
    # returns a Distribution for the final hidden state
    final = robot.Distribution()
    for x in range(robot.GRID_WIDTH):
        for y in range(robot.GRID_HEIGHT):
            final[(x, y, 'stay')] = 1
    return final
    
def final_distribution2():
    final = robot.Distribution()
    for state in robot.get_all_hidden_states():
        final[state] = 1
    return final
        
        
def pretransition(state):
    x,y,action = state
    prev_states = robot.Distribution()
    
    for s in all_possible_hidden_states:
        candidates = transition_model(s)
        if state in candidates:
            if state in prev_states:
                prev_states[s] += candidates[state]
            else:
                prev_states[s] = candidates[state]
    prev_states.renormalize()
    
    return prev_states
    
def compute_phi(observations):
    """
    Compute phi = P(observations[i]|state) for each hidden state
    
    Inputs:
        observations - list of locations given as (x, y) tuples
        
    Outputs:
        robot.Distribution for each of the observations
    """
    
    num_observations = len(observations)
    phi = [None] * num_observations
    
    for i in range(num_observations):
        phi[i] = robot.Distribution()
        observation = observations[i]
        for state in all_possible_hidden_states:
            possible_locations = observation_model(state)
            if observation in possible_locations:
                phi[i][state] = possible_locations[observation]
                
    return phi
        
        
def compute_forwards(phis):
    """
    Compute forward message alpha_i_to_j for for all timesteps.
    Inputs:
        phis - list of conditional probability distributions, 
               one for each time step.
               phi[i] = P(yi|x) where yi is the ith observation and x is any
               possible state
    Output:
        forwards - list of forward distributions.
        forwards[i] is alpha_(i-1)_to_i for i > 0
        forwards[0] is initial distribution for the system
    """
    
    num_phis = len(phis)
    forwards = [None] * num_phis
    forwards[0] = prior_distribution
    for i in range(1, num_phis):
        forwards[i] = robot.Distribution()
        for state in phis[i-1]:
            prob = forwards[i-1][state] * phis[i-1][state]
            if prob != 0:
                next_states = transition_model(state)
                for next_state in next_states:
                    if next_state in forwards[i]:
                        forwards[i][next_state] += (prob*next_states[next_state])
                    else:
                        forwards[i][next_state] = (prob*next_states[next_state])
        forwards[i].renormalize()
    
    return forwards


# -----------------------------------------------------------------------------
# Functions for you to implement
#

def compute_backwards(phis):
    """
    Compute backward message beta_i_to_j for for all timesteps.
    Inputs:
        phis - list of conditional probability distributions, 
               one for each time step.
               phi[i] = P(yi|x) where yi is the ith observation and x is any
               possible state
    Output:
        backwards - list of backward distributions.
        backwards[i] is beta_(i+1)_to_i for i < <last-element>
        backwards[<last-element>] is final distribution for the system
    """
    
    num_phis = len(phis)
    backwards = [None] * num_phis
    backwards[num_phis - 1] = final_distribution()
    for i in range (num_phis-2, -1, -1):
        backwards[i] = robot.Distribution()
        for state in phis[i+1]:
            prob = backwards[i+1][state]*phis[i+1][state]
            if prob != 0:
                prev_states = pretransition(state)
                for prev_state in prev_states:
                    if prev_state in backwards[i]:
                        backwards[i][prev_state] += (prob * transition_model(prev_state)[state])
                    else:
                        backwards[i][prev_state] = (prob * transition_model(prev_state)[state])
        backwards[i].renormalize()
        
    return backwards

def forward_backward(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    robot.py and see how it is used in both robot.py and the function
    generate_data() above, and the i-th Distribution should correspond to time
    step i
    """

    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE
    #

    phis = compute_phi(observations)
    forward_messages = compute_forwards(phis)

    backward_messages = compute_backwards(phis)

    num_observations = len(observations)
    marginals = [None] * num_observations
    for i in range(num_observations):
        marginals[i] = robot.Distribution()
        for state in phis[i]:
            marginals[i][state] = phis[i][state]*forward_messages[i][state]*backward_messages[i][state]
        marginals[i].renormalize()

    return marginals


def Viterbi(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """

    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE
    #


    num_time_steps = len(observations)
    estimated_hidden_states = [None] * num_time_steps # remove this

    return estimated_hidden_states


def second_best(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """

    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE
    #


    num_time_steps = len(observations)
    estimated_hidden_states = [None] * num_time_steps # remove this

    return estimated_hidden_states


# -----------------------------------------------------------------------------
# Generating data from the hidden Markov model
#

def generate_data(num_time_steps, make_some_observations_missing=False,
                  random_seed=None):
    # generate samples from this project's hidden Markov model
    hidden_states = []
    observations = []

    # if the random seed is not None, then this makes the randomness
    # deterministic, which may be helpful for debug purposes
    np.random.seed(random_seed)

    # draw initial state and emit an observation
    initial_state = prior_distribution.sample()
    initial_observation = observation_model(initial_state).sample()

    hidden_states.append(initial_state)
    observations.append(initial_observation)

    for time_step in range(1, num_time_steps):
        # move the robot
        prev_state = hidden_states[-1]
        new_state = transition_model(prev_state).sample()

        # maybe emit an observation
        if not make_some_observations_missing:
            new_observation = observation_model(new_state).sample()
        else:
            if np.random.rand() < .1:  # 0.1 prob. of observation being missing
                new_observation = None
            else:
                new_observation = observation_model(new_state).sample()

        hidden_states.append(new_state)
        observations.append(new_observation)

    return hidden_states, observations


# -----------------------------------------------------------------------------
# Main
#

def main():
    # flags
    make_some_observations_missing = False
    use_graphics = True
    need_to_generate_data = True

    # parse command line arguments
    for arg in sys.argv[1:]:
        if arg == '--missing':
            make_some_observations_missing = True
        elif arg == '--nographics':
            use_graphics = False
        elif arg.startswith('--load='):
            filename = arg[7:]
            hidden_states, observations = robot.load_data(filename)
            need_to_generate_data = False
            num_time_steps = len(hidden_states)

    # if no data is loaded, then generate new data
    if need_to_generate_data:
        num_time_steps = 100
        hidden_states, observations = \
            generate_data(num_time_steps,
                          make_some_observations_missing)

    print('Running forward-backward...')
    marginals = forward_backward(observations)
    print("\n")

    timestep = 2
    print("Most likely parts of marginal at time %d:" % (timestep))
    if marginals[timestep] is not None:
        print(sorted(marginals[timestep].items(),
                     key=lambda x: x[1],
                     reverse=True)[:10])
    else:
        print('*No marginal computed*')
    print("\n")

    print('Running Viterbi...')
    estimated_states = Viterbi(observations)
    print("\n")

    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10 - 1, num_time_steps):
        if estimated_states[time_step] is None:
            print('Missing')
        else:
            print(estimated_states[time_step])
    print("\n")

    print('Finding second-best MAP estimate...')
    estimated_states2 = second_best(observations)
    print("\n")

    print("Last 10 hidden states in the second-best MAP estimate:")
    for time_step in range(num_time_steps - 10 - 1, num_time_steps):
        if estimated_states2[time_step] is None:
            print('Missing')
        else:
            print(estimated_states2[time_step])
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states[time_step] != hidden_states[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between MAP estimate and true hidden " +
          "states:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states2[time_step] != hidden_states[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between second-best MAP estimate and " +
          "true hidden states:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states[time_step] != estimated_states2[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between MAP and second-best MAP " +
          "estimates:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    # display
    if use_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()


if __name__ == '__main__':
    main()
